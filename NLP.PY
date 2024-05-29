import spacy
from transformers import pipeline
from bs4 import BeautifulSoup
import requests

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load the transformers summarization pipeline
summarizer = pipeline("summarization")

def preprocess_text(html_doc):
    """
    Preprocess HTML document to extract text.
    """
    soup = BeautifulSoup(html_doc, 'lxml')
    text = soup.get_text()
    return text

def extract_entities(text):
    """
    Extract named entities from text using SpaCy.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_clauses(text):
    """
    Extract clauses (sentences) from text using SpaCy.
    """
    doc = nlp(text)
    clauses = [sent.text for sent in doc.sents]
    return clauses

def summarize_text(text):
    """
    Summarize text using transformers summarization pipeline.
    """
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def extract_legal_information(html_doc):
    """
    Extract and summarize key legal information from an HTML document.
    """
    text = preprocess_text(html_doc)
    entities = extract_entities(text)
    clauses = extract_clauses(text)
    summaries = [summarize_text(clause) for clause in clauses]
    return {
        "entities": entities,
        "clauses": clauses,
        "summaries": summaries
    }

# Example usage
if __name__ == "__main__":
    # Example HTML document (can be loaded from a file or web)
    html_doc = """<html><body><p>Your legal document content here...</p></body></html>"""
    
    # Alternatively, you can load an HTML document from a URL
    # url = "https://www.example.com/legal-document"
    # html_doc = requests.get(url).text

    legal_info = extract_legal_information(html_doc)
    
    print("Entities:")
    for entity in legal_info["entities"]:
        print(entity)

    print("\nClauses:")
    for clause in legal_info["clauses"]:
        print(clause)

    print("\nSummaries:")
    for summary in legal_info["summaries"]:
        print(summary)
