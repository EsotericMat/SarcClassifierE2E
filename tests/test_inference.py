import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
import numpy as np
from inference.app import app, PredictRequest, PredictResponse, get_sarcasm_model, get_subclass_model, get_processor


@pytest.fixture
def mock_sarcasm_model():
    model = Mock()
    model.feature_names = ['feature1', 'feature2', 'feature3']
    model.predict.return_value = np.array([0.8])  # > 0.5 threshold = sarcastic
    return model


@pytest.fixture
def mock_not_sarcasm_model():
    model = Mock()
    model.feature_names = ['feature1', 'feature2', 'feature3']
    model.predict.return_value = np.array([0.3])  # < 0.5 threshold = not sarcastic
    return model


@pytest.fixture
def mock_subclass_model():
    model = Mock()
    model.feature_names = ['feature1', 'feature2', 'feature3']
    model.predict.return_value = np.array([0.8])  # class 1
    return model


@pytest.fixture
def mock_not_subclass_model():
    model = Mock()
    model.feature_names = ['feature1', 'feature2', 'feature3']
    model.predict.return_value = np.array([0.3])  # class 0
    return model


@pytest.fixture
def mock_processor():
    processor = Mock()
    processor.run_single_text.return_value = [0.1, 0.2, 0.3]
    return processor


@pytest.fixture
def mock_error_processor():
    processor = Mock()
    processor.run_single_text.side_effect = ValueError("Processing failed")
    return processor


@pytest.fixture
def mock_failing_model():
    model = Mock()
    model.feature_names = ['feature1', 'feature2', 'feature3']
    model.predict.side_effect = RuntimeError("Model prediction failed")
    return model


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPredictSarcEndpoint:
    def test_predict_sarc_success(self, client, mock_sarcasm_model, mock_processor, monkeypatch):

        app.dependency_overrides[get_sarcasm_model] = lambda: mock_sarcasm_model
        app.dependency_overrides[get_processor] = lambda: mock_processor

        response = client.post("/predict_sarc", json={"text": "Oh great, another meeting"})
            
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert data["probability"] == 0.8
        assert data["details"]["classification"] == "Sarcasm"

    def test_predict_sarc_not_sarcasm(self, client, mock_not_sarcasm_model, mock_processor):
        app.dependency_overrides[get_sarcasm_model] = lambda: mock_not_sarcasm_model
        app.dependency_overrides[get_processor] = lambda: mock_processor
        
        try:
            response = client.post("/predict_sarc", json={"text": "This is a normal sentence"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == 0
            assert data["probability"] == 0.3
            assert data["details"]["classification"] == "Not Sarcasm"
        finally:
            app.dependency_overrides.clear()

    def test_predict_sarc_preprocess_error(self, client, mock_sarcasm_model, mock_error_processor):

        app.dependency_overrides[get_sarcasm_model] = lambda: mock_sarcasm_model
        app.dependency_overrides[get_processor] = lambda: mock_error_processor
        
        try:
            response = client.post("/predict_sarc", json={"text": "some text"})
            
            assert response.status_code == 400
            assert "Preprocess Failed" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_predict_sarc_prediction_error(self, client, mock_failing_model, mock_processor):
        app.dependency_overrides[get_sarcasm_model] = lambda: mock_failing_model
        app.dependency_overrides[get_processor] = lambda: mock_processor
        
        try:
            response = client.post("/predict_sarc", json={"text": "Some text"})
            
            assert response.status_code == 500
            assert "Prediction Failed" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()


    def test_predict_sarc_invalid_request(self, client, mock_sarcasm_model, mock_processor):

        app.dependency_overrides[get_sarcasm_model] = lambda: mock_not_sarcasm_model
        app.dependency_overrides[get_processor] = lambda: mock_processor

        response = client.post("/predict_sarc", json={})
        assert response.status_code == 422

    def test_predict_sarc_empty_text(self, client, mock_sarcasm_model, mock_processor):
        app.dependency_overrides[get_sarcasm_model] = lambda: mock_sarcasm_model
        app.dependency_overrides[get_processor] = lambda: mock_processor
        response = client.post("/predict_sarc", json={"text": ""})
        assert response.status_code == 422


class TestPredictSarcSubclassEndpoint:
    def test_predict_sarc_subclass_success(self, client, mock_subclass_model, mock_processor):
        app.dependency_overrides[get_subclass_model] = lambda: mock_subclass_model
        app.dependency_overrides[get_processor] = lambda: mock_processor
        
        try:
            response = client.post("/predict_sarc_subclass", json={"text": "Wow, what a brilliant idea"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == 1
            assert data["probability"] == 0.0  # Subclass endpoint returns 0.0
            assert data["details"]["classification"] == "Sarcasm"
        finally:
            app.dependency_overrides.clear()

    def test_predict_sarc_subclass_not_sarcasm(self, client, mock_not_subclass_model, mock_processor):
        app.dependency_overrides[get_subclass_model] = lambda: mock_not_subclass_model
        app.dependency_overrides[get_processor] = lambda: mock_processor
        
        try:
            response = client.post("/predict_sarc_subclass", json={"text": "Regular text"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == 0
            assert data["probability"] == 0.0
            assert data["details"]["classification"] == "Not Sarcasm"
        finally:
            app.dependency_overrides.clear()

    def test_predict_sarc_subclass_preprocess_error(self, client, mock_subclass_model, mock_error_processor):
        app.dependency_overrides[get_subclass_model] = lambda: mock_subclass_model
        app.dependency_overrides[get_processor] = lambda: mock_error_processor
        
        try:
            response = client.post("/predict_sarc_subclass", json={"text": "invalid input"})
            
            assert response.status_code == 400
            assert "Preprocess Failed" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_predict_sarc_subclass_prediction_error(self, client, mock_failing_model, mock_processor):
        app.dependency_overrides[get_subclass_model] = lambda: mock_failing_model
        app.dependency_overrides[get_processor] = lambda: mock_processor
        
        try:
            response = client.post("/predict_sarc_subclass", json={"text": "Some text"})
            
            assert response.status_code == 500
            assert "Prediction Failed" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_predict_sarc_subclass_invalid_request(self, client):
        app.dependency_overrides[get_subclass_model] = lambda: mock_not_sarcasm_model
        app.dependency_overrides[get_processor] = lambda: mock_processor
        response = client.post("/predict_sarc_subclass", json={})
        assert response.status_code == 422

    def test_predict_sarc_subclass_empty_text(self, client):
        app.dependency_overrides[get_subclass_model] = lambda: mock_subclass_model
        app.dependency_overrides[get_processor] = lambda: mock_processor
        response = client.post("/predict_sarc_subclass", json={"text": ""})
        assert response.status_code == 422


class TestPredictRequestModel:
    def test_valid_request(self):
        request = PredictRequest(text="Valid text input")
        assert request.text == "Valid text input"

    def test_empty_text_validation(self):
        with pytest.raises(ValueError):
            PredictRequest(text="")

    def test_missing_text_validation(self):
        with pytest.raises(ValueError):
            PredictRequest()


class TestPredictResponseModel:
    def test_response_creation(self):
        response = PredictResponse(
            prediction=1,
            probability=0.85,
            details={"classification": "Sarcasm"}
        )
        assert response.prediction == 1
        assert response.probability == 0.85
        assert response.details == {"classification": "Sarcasm"}

    def test_response_with_none_values(self):
        response = PredictResponse(prediction=0, details=None)
        assert response.prediction == 0
        assert response.probability is None
        assert response.details is None