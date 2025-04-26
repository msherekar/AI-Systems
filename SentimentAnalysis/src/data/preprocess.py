import re
import logging
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK resources...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    logger.info("NLTK resources downloaded successfully")

def normalization(text):
    """
    Normalize text by expanding contractions, removing special characters, 
    removing stopwords, and lemmatizing words with POS tagging.
    """
    try:
        if not isinstance(text, str):
            logger.warning(f"Non-string input received: {type(text)}. Converting to string.")
            text = str(text)
        
        if not text or len(text) < 2:
            logger.warning("Empty or very short text received")
            return ""
        
        expanded_text = contractions.fix(text)
        expanded_text = expanded_text.lower()
        just_text = re.sub(r'[^a-zA-Z\s]', '', expanded_text)

        try:
            word_tokens = word_tokenize(just_text)
        except Exception as e:
            logger.warning(f"Error tokenizing: {str(e)}. Using simple split.")
            word_tokens = just_text.split()

        try:
            stop_words = set(stopwords.words('english'))
            stop_words.update(["like"])  # Add custom stopwords
            filtered_words = [w for w in word_tokens if w.lower() not in stop_words]
        except Exception as e:
            logger.warning(f"Error removing stopwords: {str(e)}. Keeping all words.")
            filtered_words = word_tokens

        try:
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(w, pos='v') for w in filtered_words]
        except Exception as e:
            logger.warning(f"Error lemmatizing: {str(e)}. Using unlemmatized words.")
            lemmatized_words = filtered_words
        
        lemmatized_text = ' '.join(lemmatized_words)
        return lemmatized_text
    except Exception as e:
        logger.error(f"Error in text normalization: {str(e)}")
        return text


if __name__ == "__main__":
    # Set logging level to INFO for the example
    logging.getLogger().setLevel(logging.INFO)
    
    # Ensure NLTK resources are downloaded
    for resource in ['stopwords', 'punkt', 'wordnet']:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)
    
    # Test with various inputs
    examples = [
        "I can't believe it's already 2021. I'm so excited for the new year.",
        "The movie was great! I loved the special effects and the acting was top-notch.",
        "This product is terrible. It broke after just one use.",
        ""  # Test with empty string
    ]
    
    for i, example in enumerate(examples):
        logger.info(f"Example {i+1}:")
        logger.info(f"Original: {example}")
        normalized = normalization(example)
        logger.info(f"Normalized: {normalized}")
        logger.info("-" * 50)


