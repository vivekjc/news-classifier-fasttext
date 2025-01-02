# FastText Model for Classifying News Articles

This project utilizes FastText to classify documents as either news or non-news. The project exposes endpoints through a FastAPI server for model training and scoring.

## Endpoints Overview

- **`POST /train`** – Trains a FastText model using datasets from CCNews and Wikipedia.
- **`POST /score`** – Scores documents against a trained model.

After launching the FastAPI server, you can interact with the endpoints through the interactive docs at:
**http://127.0.0.1:8000/docs**

---

## How to Use

### 1. Install Dependencies
Ensure all dependencies are installed by running:
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Server
Run the server with:
```bash
uvicorn main:app --reload
```

---

### 3. Train the Model
To train the model, use the following command:
```bash
python3 train.py <number_of_documents> <optional_negative_documents>
```
- The minimum required number of positive documents is **20,000**.
- If the second argument is omitted, the number of negative documents defaults to the same as the positive count.
- Datasets are streamed from CCNews and Wikipedia.
- The endpoint will return a **UUID** for the trained model. Keep this ID for scoring.

**Training Data Sources:**
- **Positive Samples** – Extracted from CCNews.
- **Negative Samples** – Extracted from Wikipedia.

---

### 4. Score Documents
To score documents using the trained model:
```bash
python3 score.py
```
- You will be prompted to select a model from the available models.
- The script loads **five samples** each from CCNews and Wikipedia.
- Scoring results show the predicted label and confidence score for each document.

Alternatively, you can manually run:
```bash
python3 score.py <model_uuid>
```

---

## Dataset Configuration
Datasets are specified in `datasets_config.yaml`:
```yaml
datasets:
  ccnews:
    source: "stanford-oval/ccnews"
    name: "2022"
    split: "train"
    streaming: true
  wikipedia:
    source: "wikipedia"
    name: "20220301.en"
    split: "train"
    streaming: true
```

---

## Model Training and Scoring Details
- **Training**
  - Positive and negative documents are normalized and cleaned before training.
  - The model is trained with **5 epochs, 1.0 learning rate**, and **wordNgrams=2**.
- **Scoring**
  - The model predicts labels for each document with a confidence score.
  - If the requested model does not exist, a 500 error is returned.

---

For more information about FastAPI, visit the official documentation:
**https://fastapi.tiangolo.com**

