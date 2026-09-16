import nltk
import json
import re
import logging
from typing import List, Dict
import uuid

# Setup logging for professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data (run once)
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    raise

class TextSplitter:
    """A simple NLP text splitter for preprocessing text for sign language translation."""
    
    def __init__(self):
        """Initialize the text splitter with NLTK sentence tokenizer."""
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    def clean_text(self, text: str) -> str:
        """Clean input text by removing extra spaces and normalizing."""
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        cleaned_text = self.clean_text(text)
        sentences = self.sentence_tokenizer.tokenize(cleaned_text)
        return sentences
    
    def split_words(self, sentence: str) -> List[str]:
        """Split a sentence into words, removing punctuation."""
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum()]
        return words
    
    def process_text(self, text: str) -> Dict:
        """Process text into sentences and words, return structured data."""
        try:
            sentences = self.split_sentences(text)
            result = {
                'input_text': text,
                'sentences': []
            }
            for i, sentence in enumerate(sentences, 1):
                words = self.split_words(sentence)
                result['sentences'].append({
                    'sentence_id': i,
                    'text': sentence,
                    'words': words
                })
            return result
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {'error': str(e)}
    
    def save_output(self, data: Dict, output_path: str = 'text_split_output.json'):
        """Save processed text to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Output saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")

def main():
    """Demo the text splitter with sample text."""
    splitter = TextSplitter()
    
    # Sample text for demo
    sample_text = "Hello world! This is a test. Let's learn ASL together."
    logger.info("Processing sample text: %s", sample_text)
    
    # Process text
    result = splitter.process_text(sample_text)
    
    # Print results
    print("\n=== Text Splitting Results ===")
    print(f"Input Text: {result['input_text']}")
    for sentence in result['sentences']:
        print(f"\nSentence {sentence['sentence_id']}: {sentence['text']}")
        print(f"Words: {', '.join(sentence['words'])}")
    
    # Save to JSON
    output_file = f"text_split_{uuid.uuid4().hex}.json"
    splitter.save_output(result, output_file)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()