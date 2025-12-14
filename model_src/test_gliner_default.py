# READ THIS FIRST: This is an attempt to use GLiNER to identify PII entities. 
# However, the model is not performing well. f_score of 0.65 is not acceptable for production use.

import json
import os
from gliner import GLiNER
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from tqdm import tqdm

class GlinerV7Evaluator:
    def __init__(self, model_path, dataset_path):
        print(f"Loading model: {model_path}...")
        self.model = GLiNER.from_pretrained(model_path)
        self.dataset_path = dataset_path
        
        # 1. TARGETS: Broadened prompts to improve Recall
        self.target_mapping = {
            "NAME_STUDENT": "person's name (first and last)", 
            "EMAIL": "email address",
            "PHONE_NUM": "phone number", 
            "STREET_ADDRESS": "private address or residence",
            "URL_PERSONAL": "personal profile url, linkedin, or portfolio", 
            "USERNAME": "username, handle, or login",
            "ID_NUM": "alphanumeric identification number, government id, or social security",
        }
        
        # 2. DISTRACTORS: Added more distractors to catch false positives
        self.distractor_mapping = {
            "O_Capitalized_Word": "capitalized word or proper noun",  
            "O_CELEBRITY": "famous person, celebrity",
            "O_HISTORICAL_FIGURE": "historical figure or notable person",
            "O_WEBSITE": "public website, company site, or platform",
            "O_ROLE": "job title, profession, or role",    
            "O_ORG": "company, organization, or business",  
        }
        
        self.all_labels = list(self.target_mapping.values()) + list(self.distractor_mapping.values())
        self.reverse_mapping = {v.lower(): k for k, v in self.target_mapping.items()}

    def load_data(self, limit=None):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        return data[:limit] if limit else data

    def predict_with_windowing(self, text, threshold, window_size=512, overlap=50):
        text_length = len(text)
        all_entities = []
        
        for start in range(0, text_length, max(1, window_size - overlap)):
            end = min(start + window_size, text_length)
            chunk = text[start:end]
            entities = self.model.predict_entities(chunk, self.all_labels, threshold=threshold)
            
            for ent in entities:
                ent["start"] += start
                ent["end"] += start
                all_entities.append(ent)
            if end == text_length: break
        
        # Deduplicate
        unique_entities = {}
        for ent in all_entities:
            key = (ent["start"], ent["end"])
            if key not in unique_entities or ent["score"] > unique_entities[key]["score"]:
                unique_entities[key] = ent
        return list(unique_entities.values())

    def _convert_pred_spans_to_bio_greedy(self, tokens, trailing_whitespaces, pred_entities):
        token_spans = []
        cursor = 0
        for token, has_space in zip(tokens, trailing_whitespaces):
            start = cursor
            end = cursor + len(token)
            token_spans.append((start, end))
            cursor = end + (1 if has_space else 0)

        bio_tags = ["O"] * len(tokens)
        pred_entities = sorted(pred_entities, key=lambda x: x["start"])

        # 1. Apply Predictions
        for entity in pred_entities:
            label_prompt = entity["label"].lower()
            
            # Filter Distractors
            if label_prompt not in self.reverse_mapping:
                continue 

            schema_label = self.reverse_mapping[label_prompt]
            start_char = entity["start"]
            end_char = entity["end"]

            for i, (t_start, t_end) in enumerate(token_spans):
                if t_start < end_char and t_end > start_char:
                    if bio_tags[i] == "O":
                        if i == 0 or bio_tags[i-1] == "O" or bio_tags[i-1][2:] != schema_label:
                             bio_tags[i] = f"B-{schema_label}"
                        else:
                             bio_tags[i] = f"I-{schema_label}"


        for i in range(1, len(bio_tags)):
            token_text = tokens[i]
            prev_tag = bio_tags[i-1]
            curr_tag = bio_tags[i]
            
            # Condition: Previous was Address
            if prev_tag.endswith("STREET_ADDRESS"):
                
                # Heuristics for Zip Codes / States
                is_zip = (token_text.isdigit() and len(token_text) == 5)
                is_state = (len(token_text) == 2 and token_text.isupper())
                is_sep = (token_text in [",", ".", "\n", "-"])
                
                # If current token is "O", OR "ID_NUM", OR "PHONE" (common confusion)
                # AND it looks like a Zip/State/Separator -> Force it to be Address
                if curr_tag == "O" or "ID_NUM" in curr_tag or "PHONE" in curr_tag:
                    if is_zip or is_state or is_sep:
                        bio_tags[i] = "I-STREET_ADDRESS"

        return bio_tags

    def calculate_fbeta(self, precision, recall, beta=5):
        if precision == 0 and recall == 0: return 0.0
        beta_sq = beta ** 2
        numerator = (1 + beta_sq) * (precision * recall)
        denominator = (beta_sq * precision) + recall
        return numerator / denominator if denominator != 0 else 0.0

    def evaluate_kaggle(self, num_samples=10, threshold=0.4):
        data = self.load_data(limit=num_samples)
        true_labels_list = []
        pred_labels_list = []

        print(f"\n--- Running KAGGLE V7 Evaluation (Threshold={threshold}) ---")
        
        for entry in tqdm(data):
            preds = self.predict_with_windowing(entry["full_text"], threshold=threshold)
            predicted_bio = self._convert_pred_spans_to_bio_greedy(entry["tokens"], entry["trailing_whitespace"], preds)
            
            true_labels_list.append(entry["labels"])
            pred_labels_list.append(predicted_bio)

        report = classification_report(true_labels_list, pred_labels_list, scheme=IOB2, mode='strict', output_dict=True, zero_division=0)
        p = report['micro avg']['precision']
        r = report['micro avg']['recall']
        f5 = self.calculate_fbeta(p, r, beta=5)

        print("\n" + "="*30)
        print(f"Winner F5 Score:   {f5:.4f}")
        print("="*30)
        print(classification_report(true_labels_list, pred_labels_list, scheme=IOB2, mode='strict', zero_division=0))

    def analyze_errors(self, num_samples=10, threshold=0.4):
        data = self.load_data(limit=num_samples)
        print(f"\n\n--- 🔍 V7 ERROR ANALYSIS ---")
        for entry in data:
            preds = self.predict_with_windowing(entry["full_text"], threshold=threshold)
            pred_labels = self._convert_pred_spans_to_bio_greedy(entry["tokens"], entry["trailing_whitespace"], preds)
            if entry["labels"] != pred_labels:
                print(f"\n📄 Document ID: {entry.get('document', 'Unknown')}")
                has_printed = False
                for i, (tok, true_l, pred_l) in enumerate(zip(entry["tokens"], entry["labels"], pred_labels)):
                    if true_l != "O" or pred_l != "O" or true_l != pred_l:
                        if not has_printed:
                            print(f"{'IDX':<4} | {'TOKEN':<20} | {'TRUE LABEL':<20} | {'PRED LABEL':<20} | STATUS")
                            print("-" * 85)
                            has_printed = True
                        status = "Correct" if true_l == pred_l else "Incorrect"
                        display_tok = (tok[:18] + '..') if len(tok) > 18 else tok
                        print(f"{i:<4} | {display_tok:<20} | {true_l:<20} | {pred_l:<20} | {status}")

if __name__ == "__main__":
    DATASET_PATH = "data/default_dataset/train.json"
    evaluator = GlinerV7Evaluator("gliner-community/gliner_large-v2.5", DATASET_PATH)
    
    evaluator.evaluate_kaggle(num_samples=20, threshold=0.5)
    evaluator.analyze_errors(num_samples=20, threshold=0.5)