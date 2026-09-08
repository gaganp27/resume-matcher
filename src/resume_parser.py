import PyPDF2
import re
import spacy

nlp = spacy.load("en_core_web_sm")
SKILLS = ["python", "java", "sql", "machine learning", "deep learning", "tensorflow", "pytorch", "nlp", "react", "javascript", "aws", "docker"]

def extract_skills(text):
    text_lower = text.lower()
    return [s for s in SKILLS if s in text_lower]

def extract_email(text):
    emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
    return emails[0] if emails else None

def parse_resume(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join(page.extract_text() for page in reader.pages)
    return {"text": text, "skills": extract_skills(text), "email": extract_email(text)}