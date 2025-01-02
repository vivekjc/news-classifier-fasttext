import requests
import sys
import os
from tqdm import tqdm
from utils import normalize_and_clean_text, fetch_datasets_from_config

def list_available_models(model_directory="models"):
    return [f for f in os.listdir(model_directory) if f.endswith(".bin")]

available_models = list_available_models()

if not available_models:
    print("No models available. Train a model first.")
    sys.exit(1)

print("Available models:")
for idx, model_name in enumerate(available_models):
    print(f"{idx + 1}. {model_name}")

selected_model_index = int(input("Select a model by number: ")) - 1
if selected_model_index not in range(len(available_models)):
    print("Invalid selection.")
    sys.exit(1)

model_identifier = available_models[selected_model_index].replace("model_", "").replace(".bin", "")

datasets = fetch_datasets_from_config()
news_articles = datasets["ccnews"]
reference_texts = datasets["wikipedia"]

sample_size = 5
positive_documents = []
negative_documents = []

print(f"Extracting {sample_size} positive samples from CCNews...")
for record in tqdm(news_articles, desc="Processing CCNews", total=sample_size):
    if record["language"] == 'en':
        positive_documents.append(normalize_and_clean_text(record["plain_text"]))
    if len(positive_documents) >= sample_size:
        break

print(f"\nExtracting {sample_size} negative samples from Wikipedia...")
for record in tqdm(reference_texts, desc="Processing Wikipedia", total=sample_size):
    negative_documents.append(normalize_and_clean_text(record["text"]))
    if len(negative_documents) >= sample_size:
        break

scoring_request_payload = {
    "model_identifier": model_identifier,
    "samples_to_score": positive_documents + negative_documents
}

response = requests.post("http://127.0.0.1:8000/score", json=scoring_request_payload)

if response.status_code == 200:
    scored_documents = response.json()
    print("Received scores:")
    for index, document in enumerate(scored_documents):
        print(f"Document {index + 1}: {document['predicted_label']} -> {document['prediction_confidence']} : {document['sample'][:50]}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
