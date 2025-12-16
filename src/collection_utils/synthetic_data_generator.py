import json
import random
import argparse
import os
import re
import spacy
import time
import threading
from faker import Faker
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np


DATA_CORRUPTION_PROBABILITY = 0.3
NAME_CASE_CHANGE_PROBABILITY = 0.6
PHONE_FORMAT_CHANGE_PROBABILITY = 1.0
EMAIL_CASE_CHANGE_PROBABILITY = 0.3
USERNAME_MODIFICATION_PROBABILITY = 0.5
ADDRESS_FORMAT_CHANGE_PROBABILITY = 0.5
ID_FORMAT_CHANGE_PROBABILITY = 0.5
WHITESPACE_CORRUPTION_PROBABILITY = 0.1
NAME_FIRST_ONLY_PROBABILITY = 0.1

MAX_TEXTS_PER_MINUTE = 20
RATE_LIMIT_INTERVAL = 60.0 / MAX_TEXTS_PER_MINUTE

load_dotenv()

# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "mistralai/devstral-2512:free"
DATASET_PATH = "data/default_dataset/train.json"
OUTPUT_PATH = "data/synthetic_dataset/train.json"


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

fake = Faker()

class RateLimiter:
    def __init__(self, interval):
        self.interval = interval
        self.last_call = 0
        self.lock = threading.Lock()

    def wait_for_slot(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call
            wait_time = self.interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self.last_call = time.time()

limiter = RateLimiter(RATE_LIMIT_INTERVAL)

def random_id_num():
    patterns = [
        lambda: "".join(
            random.choices("0123456789", k=random.randint(8, 12))
        ),  # Simple numeric
        lambda: f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{''.join(random.choices('0123456789', k=8))}",  # Letter prefix
        lambda: f"{''.join(random.choices('0123456789', k=3))}-{ ''.join(random.choices('0123456789', k=2))}-{''.join(random.choices('0123456789', k=4))}",  # SSN style
        lambda: f"{''.join(random.choices('0123456789', k=9))}",  # 9-digit
        lambda: f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))}",  # Alphanumeric
    ]
    return random.choice(patterns)()


PII_MAPPING = {
    "<NAME_STUDENT>": {"label": "NAME_STUDENT", "generator": fake.name},
    "<EMAIL>": {"label": "EMAIL", "generator": fake.email},
    "<PHONE_NUM>": {"label": "PHONE_NUM", "generator": fake.phone_number},
    "<STREET_ADDRESS>": {"label": "STREET_ADDRESS", "generator": fake.address},
    "<URL_PERSONAL>": {"label": "URL_PERSONAL", "generator": fake.url},
    "<USERNAME>": {"label": "USERNAME", "generator": fake.user_name},
    "<ID_NUM>": {"label": "ID_NUM", "generator": random_id_num},
}


def get_random_sample(data):
    return random.choice(data)


def generate_prompt(sample_text):
    placeholders = ", ".join(PII_MAPPING.keys())
    prompt = (
        f"Here is a sample text:\n\n{sample_text}\n\n"
        f"Please generate a new text that imitates the style and structure of the sample text. "
        f"However, replace any specific personal information (names, emails, addresses, etc.) "
        f"with placeholders from this list: {placeholders}. "
        f"If the sample text does not contain personal information, please invent some plausible details "
        f"(like an author name, email, address, or mentions of other people) and use placeholders for them. "
        f"CRITICAL: To make the text more realistic and challenging, please also include NON-PII entities that should NOT be replaced by placeholders. "
        f"For example:\n"
        f"- Mention famous people (e.g., 'Albert Einstein', 'Taylor Swift') or fictional characters (e.g., 'Harry Potter', 'Sherlock Holmes') - do NOT use <NAME_STUDENT> for these.\n"
        f"- Mention well-known cities or countries (e.g., 'Paris', 'New York', 'France') in general contexts (e.g., 'I dream of visiting Paris') - do NOT use <STREET_ADDRESS> for these.\n"
        f"- Mention public websites (e.g., 'google.com', 'wikipedia.org') - do NOT use <URL_PERSONAL> for these.\n"
        f"- Mention dates (e.g., 'January 1st, 2024', '12/05/2023') or monetary amounts (e.g., '$500', '100 USD') - do NOT use placeholders for these.\n"
        f"Only use the placeholders for information that would be considered sensitive PII in a real context (e.g., the student's own name, their specific home address, their personal phone number).\n"
        f"Do not use any real PII. The text should be coherent and realistic. "
        f"Output ONLY the generated text with placeholders. Do not include any introductory text like 'Here is the text' or markdown code blocks."
    )
    return prompt

def call_llm(prompt):
    # Enforce global rate limit before calling API
    limiter.wait_for_slot()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )

    # RETRY LOGIC
    max_retries = 10
    base_delay = 2  # Start waiting 2 seconds

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
            
        except Exception as e:
            # Check if it's a rate limit error (usually 429)
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait_time = base_delay * (2 ** attempt) # Exponential backoff: 2, 4, 8, 16...
                print(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # If it's a different error (e.g. invalid key), fail immediately
                print(f"Error calling LLM: {e}")
                return None
    
    print("Max retries exceeded.")
    return None


def corrupt_pii(value, label_type):
    if random.random() > DATA_CORRUPTION_PROBABILITY:
        return value

    # whitespace corruption
    if random.random() < WHITESPACE_CORRUPTION_PROBABILITY:
        idx = random.randint(0, len(value))
        value = value[:idx] + " " + value[idx:]

    if label_type == "NAME_STUDENT":
        if random.random() > NAME_CASE_CHANGE_PROBABILITY:
            return value.lower()
        elif random.random() < NAME_CASE_CHANGE_PROBABILITY:
            return value.upper()
        elif random.random() < NAME_FIRST_ONLY_PROBABILITY:
            # First name only
            return value.split()[0]

    elif label_type == "PHONE_NUM":
        if random.random() < PHONE_FORMAT_CHANGE_PROBABILITY:
            formats = [
                lambda x: re.sub(r"\D", "", x),  # 1234567890
                lambda x: re.sub(r"\D", "", x)[:3]
                + "."
                + re.sub(r"\D", "", x)[3:6]
                + "."
                + re.sub(r"\D", "", x)[6:],  # 123.456.7890
                lambda x: re.sub(r"\D", "", x)[:3]
                + " "
                + re.sub(r"\D", "", x)[3:6]
                + " "
                + re.sub(r"\D", "", x)[6:],  # 123 456 7890
                lambda x: "("
                + re.sub(r"\D", "", x)[:3]
                + ") "
                + re.sub(r"\D", "", x)[3:6]
                + "-"
                + re.sub(r"\D", "", x)[6:],  # (123) 456-7890
            ]
            try:
                return random.choice(formats)(value)
            except:
                return value

    elif label_type == "EMAIL":
        if random.random() < EMAIL_CASE_CHANGE_PROBABILITY:
            return value.lower()
        if random.random() < EMAIL_CASE_CHANGE_PROBABILITY * 0.33:
            return value.upper()

    elif label_type == "USERNAME":
        if random.random() < USERNAME_MODIFICATION_PROBABILITY:
            if random.random() < 0.5:
                return value + str(random.randint(1, 99))
            else:
                return value.lower()

    elif label_type == "STREET_ADDRESS":
        if random.random() < ADDRESS_FORMAT_CHANGE_PROBABILITY:
            return value.replace("\n", ", ")
        if random.random() < ADDRESS_FORMAT_CHANGE_PROBABILITY * 0.4:
            value = (
                value.replace("Street", "St.")
                .replace("Avenue", "Ave.")
                .replace("Road", "Rd.")
            )

    elif label_type == "ID_NUM":
        if random.random() < ID_FORMAT_CHANGE_PROBABILITY:
            return re.sub(r"[\-\s]", "", value)

    return value


def fill_template_and_label(template, doc_id):
    placeholder_pattern = re.compile("|".join(map(re.escape, PII_MAPPING.keys())))

    matches = list(placeholder_pattern.finditer(template))

    full_text = ""
    last_end = 0
    spans = []

    for match in matches:
        start, end = match.span()
        placeholder = match.group()

        full_text += template[last_end:start]

        fake_value = PII_MAPPING[placeholder]["generator"]()
        fake_value = fake_value.replace("\n", ", ")

        # Apply corruption/noise to make data more realistic
        fake_value = corrupt_pii(fake_value, PII_MAPPING[placeholder]["label"])

        value_start = len(full_text)
        full_text += fake_value
        value_end = len(full_text)

        spans.append((value_start, value_end, PII_MAPPING[placeholder]["label"]))

        last_end = end

    full_text += template[last_end:]

    # Tokenize
    doc = nlp(full_text)

    tokens = [token.text for token in doc]
    trailing_whitespace = [bool(token.whitespace_) for token in doc]
    labels = ["O"] * len(doc)

    # Assign labels
    for span_start, span_end, label_type in spans:
        tokens_in_span = [
            t for t in doc if t.idx >= span_start and t.idx + len(t.text) <= span_end
        ]

        for i, token in enumerate(tokens_in_span):
            if i == 0:
                labels[token.i] = f"B-{label_type}"
            else:
                labels[token.i] = f"I-{label_type}"

    return {
        "document": doc_id,
        "full_text": full_text,
        "tokens": tokens,
        "trailing_whitespace": trailing_whitespace,
        "labels": labels,
    }


def process_single_generation(i, data):
    try:
        # GIVE THE LLM A RANDOM SAMPLE FROM THE DATASET TO IMITATE
        sample = get_random_sample(data)
        sample_text = sample.get("full_text", "")
        if not sample_text:
            return None

        prompt = generate_prompt(sample_text)
        template = call_llm(prompt)

        if template:
            # Clean up potential markdown code blocks from LLM
            template = template.replace("```json", "").replace("```", "").strip()

            result = fill_template_and_label(template, f"synthetic_{i}")
            return result
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PII data.")
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        with open(DATASET_PATH, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    print(f"Generating {args.num_samples} samples with {args.workers} workers...")
    print(f"Rate Limit Enforced: {MAX_TEXTS_PER_MINUTE} texts/min ({RATE_LIMIT_INTERVAL}s interval)")
    print(f"Saving results incrementally to {OUTPUT_PATH}...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("[\n")

        first_entry = True
        count = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_single_generation, i, data)
                for i in range(args.num_samples)
            ]

            for future in as_completed(futures):
                res = future.result()
                if res:
                    if not first_entry:
                        f.write(",\n")

                    json.dump(res, f, indent=2)
                    f.flush()

                    first_entry = False
                    count += 1
                    print(f"Generated sample {count}/{args.num_samples}")

        f.write("\n]")

    print("Done.")


if __name__ == "__main__":
    main()