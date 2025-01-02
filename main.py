from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
import fasttext
import os
from typing import List, Dict
import numpy as np

app = FastAPI()

MODEL_STORAGE_PATH = "models"
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

registered_models: Dict[str, str] = {}

class DocumentTrainingData(BaseModel):
    positive_documents: List[str]
    negative_documents: List[str]

class ScoringData(BaseModel):
    samples_to_score: List[str]
    model_identifier: str

@app.post("/train")
async def train_text_classifier(training_data: DocumentTrainingData):
    if len(training_data.positive_documents) < 20000:
        raise HTTPException(status_code=400, detail="Minimum 20,000 positive samples required.")

    if len(training_data.negative_documents) < len(training_data.positive_documents):
        raise HTTPException(
            status_code=400,
            detail=f"Negative samples must be at least {len(training_data.positive_documents)}.")

    training_data_file = os.path.join(MODEL_STORAGE_PATH, f"training_{uuid4().hex}.txt")
    with open(training_data_file, "w") as file:
        for sample in training_data.positive_documents:
            file.write(f"__label__positive {sample}\n")
        for sample in training_data.negative_documents:
            file.write(f"__label__negative {sample}\n")

    model_identifier = uuid4().hex
    model_file_path = os.path.join(MODEL_STORAGE_PATH, f"model_{model_identifier}.bin")
    classifier_model = fasttext.train_supervised(
        input=training_data_file, epoch=5, lr=1.0, wordNgrams=2, bucket=200000, dim=50, loss='softmax')
    classifier_model.save_model(model_file_path)

    os.remove(training_data_file)

    registered_models[model_identifier] = model_file_path
    
    return {"model_identifier": model_identifier}

@app.post("/score")
async def score_text_samples(scoring_data: ScoringData):
    model_file_path = os.path.join(MODEL_STORAGE_PATH, f"model_{scoring_data.model_identifier}.bin")

    if not os.path.exists(model_file_path):
        raise HTTPException(status_code=500, detail="Specified model not found.")
    
    classifier_model = fasttext.load_model(model_file_path)

    scored_results = []
    for sample in scoring_data.samples_to_score:
        labels, probabilities = classifier_model.predict(sample, k=1)
        probabilities = np.asarray(probabilities)
        scored_results.append({
            "sample": sample,
            "predicted_label": labels[0] if labels else "unknown",
            "prediction_confidence": probabilities[0] if len(probabilities) > 0 else 0.0
        })
    
    return scored_results
