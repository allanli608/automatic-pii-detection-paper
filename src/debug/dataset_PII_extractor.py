import json
from enum import Enum

class PIIType(Enum):
    O = "O"
    NAME_STUDENT = "NAME_STUDENT"
    EMAIL = "EMAIL"
    ID_NUM = "ID_NUM"
    PHONE_NUM = "PHONE_NUM"
    URL_PERSONAL = "URL_PERSONAL"
    STREET_ADDRESS = "STREET_ADDRESS"
    USERNAME = "USERNAME"

def extract_all_entities(tokens, labels):
    results = {}
    for pii_type in PIIType:
        if pii_type == PIIType.O:
            continue
        entities = extract_entities(tokens, labels, pii_type.value)
        results[pii_type.value] = entities
    return results


def extract_entities(tokens, labels, target_type):
    results = []
    i = 0

    while i < len(labels):
        label = labels[i]

        # Match BIO labels (B-TYPE or I-TYPE)
        if label.startswith("B-") and label.endswith(target_type):
            entity_tokens = [tokens[i]]
            i += 1

            # collect subsequent I-type tokens
            while i < len(labels) and labels[i].startswith("I-") and labels[i].endswith(target_type):
                entity_tokens.append(tokens[i])
                i += 1

            results.append(" ".join(entity_tokens))
        else:
            i += 1

    return results


if __name__ == "__main__":

    
    dataset_json_path = "data/synthetic_dataset/train.json"
    output_txt_path = "data_utils/pii_extraction_output.txt"

    unique_labels = {}

    try:
        with open(dataset_json_path, "r") as f:
            data = json.load(f)

        with open(output_txt_path, "w") as out:
            for entry in data:
                document = entry.get("document", "")
                tokens = entry.get("tokens", [])
                labels = entry.get("labels", [])

                # Collect all labels
                for label in labels:
                    unique_labels[label] = unique_labels.get(label, 0) + 1

                # Extract BIO entities of selected PII type
                found_entities = extract_all_entities(tokens, labels)


                for pii_type, entities in found_entities.items():
                    for ent in entities:
                        out.write(f"Found {pii_type}: {ent} in document {document}\n")

        print(f"PII extraction completed. Output saved to {output_txt_path}")
        print(f"Unique PII labels found: {unique_labels.keys()}")

    except Exception as e:
        print(f"Error reading dataset: {e}")
