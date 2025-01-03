# Scalable News Classification and Data Curation Pipeline with FastAPI, FastText, and Hugging Face Datasets

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
## Troubleshooting

ValueError: Unable to avoid copy while creating an array as requested.

If you get the above error you can modify the FastText.py on line 239 to
`return labels, np.array(probs)` and rerun the fastapi server.

#### Stack trace
```
Traceback (most recent call last):
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/uvicorn/protocols/http/h11_impl.py", line 403, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.scope, self.receive, self.send
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/fastapi/applications.py", line 1054, in __call__
    await super().__call__(scope, receive, send)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/middleware/errors.py", line 187, in __call__
    raise exc
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/middleware/errors.py", line 165, in __call__
    await self.app(scope, receive, _send)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/middleware/exceptions.py", line 62, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/routing.py", line 715, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/routing.py", line 735, in app
    await route.handle(scope, receive, send)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/routing.py", line 288, in handle
    await self.app(scope, receive, send)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/routing.py", line 76, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/starlette/routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/fastapi/routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
    )
    ^
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/vchacko/news-classifier-fasttext/main.py", line 64, in score_text_samples
    labels, probabilities = classifier_model.predict(sample, k=1)
                            ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/Users/vchacko/news-classifier-fasttext/.venv/lib/python3.13/site-packages/fasttext/FastText.py", line 239, in predict
    return labels, np.array(probs, copy=False)
                   ~~~~~~~~^^^^^^^^^^^^^^^^^^^
ValueError: Unable to avoid copy while creating an array as requested.
If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).
For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.
```


---

For more information about FastAPI, visit the official documentation:
**https://fastapi.tiangolo.com**

