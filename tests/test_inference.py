# tests/test_inference.py
# uv run pytest -q tests/test_inference.py
import json
import pathlib
import pytest
from fastapi.testclient import TestClient
from src.app import app

# create a TestClient passing the FastAPI app
client = TestClient(app)

def test_predict_endpoint():
    
    # load sample payload
    sample_path = pathlib.Path(__file__).parent / "sample.json"
    with open(sample_path) as f:
        payload = json.load(f)

    # call the endpoint
    resp = client.post("/predict", json=payload)
    print("STATUS:", resp.status_code)
    print("BODY:", resp.json()) 

    # check status code
    assert resp.status_code == 200

    data = resp.json()

    # check each key in response
    assert "probabilities" in data
    assert "class" in data

    probs = data["probabilities"]
    labels = data["class"]
    
    #check lengths and value ranges
    assert len(probs) == len(payload) == len(labels)
    assert all(0.0 <= float(p) <= 1.0 for p in probs)
    assert all(x in (0, 1) for x in labels)

