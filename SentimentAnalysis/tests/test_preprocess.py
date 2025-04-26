"""
Tests for the preprocess module.
"""

import sys
import pytest
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import contractions

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class TestPreprocess:
    """
    Test cases for text normalization.
    """

    def test_normalization_special_chars(self):
        from src.data.preprocess import normalization

        text = "Hello, world! This is a test. #@$%^&*"
        result = normalization(text)

        assert all(char.isalnum() or char.isspace() for char in result), "Special characters not removed."

    def test_normalization_stopwords(self):
        from src.data.preprocess import normalization

        text = "This is a test with stopwords like the and a"
        result = normalization(text)

        # Check that common stopwords are removed
        forbidden = ["the", "and", "a", "is", "with", "like"]
        for word in forbidden:
            assert word not in result.split()

        assert "test" in result
        assert "stopword" in result or "stopwords" in result

    def test_normalization_lemmatization(self):
        from src.data.preprocess import normalization

        text = "The cats are running quickly."
        result = normalization(text)

        words = result.split()
        assert "cat" in words
        assert "run" in words
        assert "quickly" in words  # <-- allow "quickly", don't expect "quick"


    def test_normalization_lowercase(self):
        from src.data.preprocess import normalization

        text = "This Is A Mixed Case Text"
        result = normalization(text)

        assert result == result.lower()
        assert "this" not in result
        assert "mix" in result
        assert "case" in result
        assert "text" in result
