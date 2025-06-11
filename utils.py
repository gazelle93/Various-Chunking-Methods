import spacy
import nltk
import re


def get_nlp_pipeline(_nlp_pipeline):
    if _nlp_pipeline == "spacy":
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Model 'en_core_web_sm' not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
            # Try loading again after download
            return spacy.load("en_core_web_sm")

def check_nltk():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        print("NLTK 'punkt_tab' resource is already downloaded.")
    except LookupError:
        print("NLTK 'punkt_tab' resource not found. Downloading...")
        nltk.download('punkt_tab')


def word_tokenization(_input_text, nlp, _nlp_pipeline='whitespace', _lower=False):
    if _lower == True:
        _input_text = _input_text.lower()

    if _nlp_pipeline == "spacy":
        doc = nlp(_input_text)
        return [token.text for token in doc]

    elif _nlp_pipeline == "nltk":
        check_nltk()
        return nltk.tokenize.word_tokenize(_input_text)

    else:
        return _input_text.split()


def sent_tokenization(_input_text, nlp, _nlp_pipeline='regex'):
    if _nlp_pipeline == "nltk":
        check_nltk()
        return nltk.sent_tokenize(_input_text)

    elif _nlp_pipeline == "spacy":
        doc = nlp(_input_text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    else:
        # Split on '.', '?', '!' followed by space or newline, keep punctuation
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, _input_text)
        return [s.strip() for s in sentences if s.strip()]

# Helper function to split text by a separator and preserve the separator
def split_by_separator(_input_text, separator):
    if separator == "":
        return list(_input_text)
    
    parts = _input_text.split(separator)
    # Add back the separator to each part for consistency
    return [p.strip() + separator for p in parts if p.strip()]
