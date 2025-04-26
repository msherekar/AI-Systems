"""
Tests for the model module.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


@pytest.fixture
def mock_model():
    """
    Fixture to mock the model loading.
    """
    with patch('os.path.exists', return_value=True), \
         patch('joblib.load') as mock_load:
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0, 1])
        mock_load.return_value = model
        yield mock_load


class TestModel:
    """
    Test cases for the Model class.
    """

    def test_model_initialization(self, mock_model):
        from src.models.model import Model

        model = Model(model_path='fake_path.pkl')
        mock_model.assert_called_once()
        assert hasattr(model, 'model')

    def test_model_predict_sentiment(self, mock_model):
        from src.models.model import Model

        model = Model(model_path='fake_path.pkl')
        embeddings = np.random.rand(4, 100)
        preds = model.predict_sentiment(embeddings)

        assert preds.shape == (4,)
        assert all(p in [0, 1] for p in preds)

    def test_model_error_handling(self, mock_model):
        from src.models.model import Model

        model = Model(model_path='fake_path.pkl')

        with pytest.raises(TypeError):
            model.predict_sentiment("not an array")

        model.model.predict.side_effect = Exception("Predict error")
        with pytest.raises(Exception):
            model.predict_sentiment(np.random.rand(4, 100))

    def test_model_input_shapes(self, mock_model):
        from src.models.model import Model

        model = Model(model_path='fake_path.pkl')
        single_sample = np.random.rand(100)
        multi_samples = np.random.rand(10, 100)

        model.model.predict.side_effect = lambda x: np.zeros(len(x))

        preds_single = model.predict_sentiment(np.array([single_sample]))
        preds_multi = model.predict_sentiment(multi_samples)

        assert preds_single.shape == (1,)
        assert preds_multi.shape == (10,)
