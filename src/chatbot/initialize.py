import nltk

def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.download('vader_lexicon')
        print("Successfully downloaded NLTK resources.")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
