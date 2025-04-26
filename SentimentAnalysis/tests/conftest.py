"""
Common pytest fixtures for the SentimentAnalysis project.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

@pytest.fixture
def sample_text():
    """
    Fixture that provides a sample text for testing.
    """
    return "This is a sample text for testing. It has some punctuation!"

@pytest.fixture
def sample_dataframe():
    """
    Fixture that provides a sample dataframe for testing.
    """
    return pd.DataFrame({
        'text': [
            "This movie was great! I loved it.",
            "Terrible acting and plot. Waste of time.",
            "Average film, nothing special to see here."
        ],
        'sentiment': [1, 0, 0.5]
    })

@pytest.fixture
def sample_embeddings():
    """
    Fixture that provides sample embeddings for testing.
    """
    # Create 3 samples with embedding dimension of 100
    return np.random.rand(3, 100)

@pytest.fixture
def mock_word2vec():
    """
    Fixture that mocks a Word2Vec model.
    """
    with patch('gensim.models.Word2Vec') as mock_word2vec:
        mock_model = MagicMock()
        mock_model.wv = {}
        mock_model.wv.__getitem__ = lambda self, key: np.random.rand(10)
        mock_model.wv.__contains__ = lambda self, key: True
        mock_model.vector_size = 10
        mock_word2vec.return_value = mock_model
        mock_word2vec.load.return_value = mock_model
        yield mock_word2vec 