# Endpoints: GET /health, POST /predict
# src/app.py
import os
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import List

from src.io_schemas import ChurnRow
from src.features import TabPreprocess
from src.models import to_keras_feed

ART_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
MODEL_DIR = os.path.join(ART_DIR, "model_keras.keras")

#intance preprocessor
pre = TabPreprocess.load(ART_DIR)

#LOAD MODEL
model = tf.keras.models.load_model(MODEL_DIR)

app = FastAPI(title="Churn Prediction API (Keras)", version="1.0")



@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(rows: List[ChurnRow]):
    try:
        # revert to dataframe
        X = pd.DataFrame([r.model_dump() for r in rows])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    # preprocess
    Xm = pre.transform(X)

    # predict
    probs = model.predict(to_keras_feed(Xm, pre.cat_cols, pre.num_cols), verbose=0).ravel()

    # transfrom to binary predictions
    preds = (probs >= 0.5).astype(int).tolist()

    return {"probabilities": [float(p) for p in probs], "class": preds}

# src/app.py (place below `app = FastAPI(...)`)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Return 400 ONLY for:
      - missing fields ("field required")
      - invalid literals / unknown categories per schema ("unexpected value", literal_error, enum)
    Fall back to FastAPI's 422 for everything else.
    """
    messages = []

    for err in exc.errors():
        loc = err.get("loc", [])
        msg = err.get("msg", "")
        typ = err.get("type", "")
        ctx = err.get("ctx") or {}

        # Try to extract row index and field name from loc like: ('body', 0, 'plan_type')
        row = None
        field = None
        if len(loc) >= 3 and loc[0] == "body" and isinstance(loc[1], int):
            row = loc[1]
            field = str(loc[2])
        elif len(loc) >= 2 and loc[0] == "body":
            field = str(loc[1])

        where = f"row {row}" if row is not None else "payload"

        # Case 1: missing required field
        if "field required" in msg or typ in ("missing",):
            if field:
                messages.append(f"{where}: missing field '{field}'")
            else:
                messages.append(f"{where}: missing required field")
            continue

        # Case 2: invalid literal (bad category)
        is_literal = (
            "unexpected value" in msg or
            typ in ("literal_error", "enum")
        )
        if is_literal:
            # Pydantic v2 often places allowed values in ctx["expected"] / ["permitted"] / ["allowed"]
            allowed = ctx.get("expected") or ctx.get("permitted") or ctx.get("allowed")
            if isinstance(allowed, (list, tuple, set)):
                allowed_str = ", ".join(repr(v) for v in allowed)
            elif allowed is not None:
                allowed_str = repr(allowed)
            else:
                allowed_str = None

            if field and allowed_str:
                messages.append(f"{where}: invalid value for '{field}'. Allowed: {allowed_str}")
            elif field:
                messages.append(f"{where}: invalid value for '{field}'")
            else:
                messages.append(f"{where}: invalid category value")
            continue

    # If we built any helpful messages, return 400; otherwise fall back to default 422
    if messages:
        return JSONResponse(status_code=400, content={"errors": messages})

    # Let FastAPI return its normal 422 details for other validation problems
    raise exc
