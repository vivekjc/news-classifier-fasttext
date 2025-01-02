import requests
import sys
from tqdm import tqdm
from utils import normalize_and_clean_text, fetch_datasets_from_config

if not (2 <= len(sys.argv) <= 3):
    print("Usage: python load_and_train.py <num_docs> [<num_negative_docs>]")
    sys.exit(1)

positive_document_count = int(sys.argv[1])
negative_document_count = int(sys.argv[2]) if len(sys.argv) == 3 else positive_document_count

dataset_sources = fetch_datasets_from_config()
news_dataset = dataset_sources["ccnews"]
reference_dataset = dataset_sources["wikipedia"]

positive_documents = []
negative_documents = []

print(f"Extracting {positive_document_count} positive samples from CCNews...")
for record in tqdm(news_dataset, desc="Processing CCNews", total=positive_document_count):
    if record["language"] == 'en':
        positive_documents.append(normalize_and_clean_text(record["plain_text"]))
    if len(positive_documents) >= positive_document_count:
        break

print(f"\nExtracting {negative_document_count} negative samples from Wikipedia...")
for record in tqdm(reference_dataset, desc="Processing Wikipedia", total=negative_document_count):
    negative_documents.append(normalize_and_clean_text(record["text"]))
    if len(negative_documents) >= negative_document_count:
        break

training_request_payload = {
    "positive_documents": positive_documents,
    "negative_documents": negative_documents
}

print("\nSubmitting training request...")
response = requests.post("http://127.0.0.1:8000/train", json=training_request_payload)
print(response.json())