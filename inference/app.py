from fastapi import FastAPI, HTTPException, Depends
import uvicorn
from typing import Optional, Any
import os
import numpy as np
from pydantic import BaseModel, Field
from configs.manager import ConfigManager
from .inference_utils import load_model, predict_with_model, map_result
from sarcasm_classifier.components.preprocess import Preprocess
import logging

logger = logging.getLogger(__name__)

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

def get_sarcasm_model():
    return load_model(config.model_file)

def get_subclass_model():
    return load_model(config.subclass_model_file)

def get_processor():
    return Preprocess()

app = FastAPI(title=config.title,  version=config.version)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict_sarc', response_model=PredictResponse)
def predict_sarc(req: PredictRequest, model: Any = Depends(get_sarcasm_model), processor: Any = Depends(get_processor)):
    try:
        signal = processor.run_single_text(req.text)
    except ValueError as e:
        logger.info(f'Preprocess Failed: {e}')
        raise HTTPException(status_code=400, detail=f'Preprocess Failed: {e}') # Bad Request

    try:
        prediction, proba = predict_with_model(model, signal, threshold=config.threshold)
    except RuntimeError as e:
        logger.error(f'Predict Failed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Prediction Failed: {e}')

    return PredictResponse(
        prediction=prediction,
        probability=np.round(proba, 3),
        details={'classification': map_result(prediction)}
    )

@app.post('/predict_sarc_subclass', response_model=PredictResponse)
def predict_sarc_subclass(req: PredictRequest, model: Any = Depends(get_subclass_model), processor: Any = Depends(get_processor)):
    try:
        signal = processor.run_single_text(req.text)
    except ValueError as e:
        logger.info(f'Predict Failed: {e}')
        raise HTTPException(status_code=400, detail=f'Preprocess Failed: {e}') # Bad Request

    try:
        prediction, proba = predict_with_model(model, signal)
    except RuntimeError as e:
        logger.error(f'Predict Failed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Prediction Failed: {e}')

    return PredictResponse(
        prediction=prediction,
        probability=0.0,
        details={'classification': map_result(prediction)}
    )

if __name__ == "__main__":
    uvicorn.run("inference.app:app", host="0.0.0.0", port=8080, log_level="info")




