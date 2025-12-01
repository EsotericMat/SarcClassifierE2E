from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Optional
import os
from pydantic import BaseModel, Field
from configs.manager import ConfigManager
from inference_utils import load_model, predict_with_model, map_result
from sarcasm_classifier.components.preprocess import Preprocess
from contextlib import asynccontextmanager

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The text to be predicted.",
    )

class PredictResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    details: Optional[dict]

config = ConfigManager('app').config

processor = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading Assets')
    global processor, model
    processor = Preprocess()
    try:
        model = load_model(config.model_file)
    except Exception as e:
        print(f'Cant load model: {e}')
    print('Service is Ready')
    yield

app = FastAPI(title=config.title,  version=config.version, lifespan=lifespan)

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not found') # Service Unavailable

    try:
        signal = processor.run_single_text(req.text[:120], add_punct=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Preprocess Failed: {e}') # Bad Request

    try:
        prediction, proba = predict_with_model(model, signal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction Failed: {e}')

    return PredictResponse(
        prediction=prediction,
        probability=proba,
        details={'classification': map_result(prediction)}
    )

if __name__ == "__main__":
    uvicorn.run("inference.app:app", host="0.0.0.0", port=8080, log_level="info")




