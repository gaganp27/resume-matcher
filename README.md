# 📄 AI-Powered Resume Screening & Job Matching

> **An AI system that reads resumes, extracts skills, and matches candidates to job descriptions using NLP.**
> Difficulty: ⭐⭐⭐ Medium | Duration: 4-5 weeks

---

## 🎯 What You Will Learn

1. **What is Natural Language Processing (NLP) + Cosine Similarity?** — NLP helps computers understand human text. We convert resumes and job descriptions into numbers (vectors), then measure how similar they are using cosine similarity.
2. **How to work with real-world data** — Download, clean, and prepare datasets
3. **How to train ML models** — Teach computers to make predictions
4. **How to evaluate results** — Know if your model is good or bad
5. **How to build a web app** — Create an interactive Streamlit dashboard
6. **How to deploy for free** — Put your app online for anyone to use

---

## 1. About the Project

### What is This Project?
An AI system that reads resumes, extracts skills, and matches candidates to job descriptions using NLP.

### Why Does It Matter?
HR teams spend 23 seconds per resume. This system does it in 0.1 seconds.

### What You Will Build:
- Data exploration and visualization
- Machine learning model training
- Interactive Streamlit web application
- Deployed online (free)

---

## 2. Background Knowledge

### What is Natural Language Processing (NLP) + Cosine Similarity?
NLP helps computers understand human text. We convert resumes and job descriptions into numbers (vectors), then measure how similar they are using cosine similarity.

### How Does It Work?
1. **Input**: You provide data (health metrics, text, numbers)
2. **Processing**: The model finds patterns in the data
3. **Output**: The model makes a prediction (disease/no disease, fake/real, etc.)

---

## 3. Prerequisites

### Before Starting:
- [ ] A computer (Windows, Mac, or Linux)
- [ ] Internet connection
- [ ] 4-5 weeks of time
- [ ] Willingness to learn!

### You Do NOT Need:
- ❌ Prior programming experience
- ❌ Math degree
- ❌ Expensive software

### What We Will Use (All Free):
| Tool | Purpose | Cost |
|------|---------|------|
| **Python** | Programming language | Free |
| **VS Code** | Code editor | Free |
| **Streamlit** | Web app framework | Free |
| **Kaggle** | Dataset source | Free |
| **Hugging Face** | Deploy apps | Free |

---

## 4. Setup Guide

### Step 1: Install Python
1. Go to https://www.python.org/downloads/
2. Download Python 3.10+
3. Run installer
4. **Check "Add Python to PATH"** 
5. Click Install Now

### Step 2: Install VS Code
1. Go to https://code.visualstudio.com/
2. Download and install
3. Install Python extension (search "Python" in Extensions)

### Step 3: Create Project
```bash
# Open Terminal (Windows: search "cmd", Mac: search "Terminal")
mkdir 01-resume-screening-job-matching
cd 01-resume-screening-job-matching
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

### Step 4: Install Libraries
```bash
pip install streamlit spacy scikit-learn PyPDF2 sentence-transformers pandas numpy plotly
```

**What each library does:**
| Library | Purpose |
|---------|---------|
| `pandas` | Handle data tables (like Excel) |
| `numpy` | Math operations |
| `scikit-learn` | Machine Learning algorithms |
| `xgboost` | Powerful gradient boosting algorithm |
| `shap` | Explain model predictions |
| `streamlit` | Build web apps easily |
| `plotly` | Interactive charts |
| `matplotlib` | Static charts |
| `seaborn` | Statistical visualizations |

---

## 5. Step-by-Step Implementation

### Step 1: Install Python & VS Code
Download Python 3.10+ from python.org. Install VS Code from code.visualstudio.com. Install Python extension in VS Code.

### Step 2: Create Project & Virtual Environment
Run: mkdir resume-matcher && cd resume-matcher && python -m venv venv && venv\Scripts\activate (Windows) or source venv/bin/activate (Mac/Linux)

### Step 3: Install Libraries
Run: pip install streamlit spacy scikit-learn PyPDF2 pandas numpy plotly && python -m spacy download en_core_web_sm

### Step 4: Download Dataset
Go to kaggle.com, search 'Resume Dataset', download and extract to data/ folder

### Step 5: Build Resume Parser
Create src/resume_parser.py that extracts text from PDF, finds skills using pattern matching, extracts email using regex

### Step 6: Build Matching Engine
Create src/matching.py that uses TF-IDF to convert text to vectors, then cosine similarity to match resumes to jobs

### Step 7: Build Streamlit App
Create app.py with file uploader for resumes, text area for job description, matching results with scores

### Step 8: Run Locally
Run: streamlit run app.py — opens at http://localhost:8501

### Step 9: Deploy Free
Push to GitHub, create Hugging Face Space with Streamlit SDK, upload files, app goes live!



---

## 6. Core Code

### 6.1 Data Exploration
```python
import spacy
import PyPDF2
import re

nlp = spacy.load("en_core_web_sm")

SKILLS = ["python", "java", "sql", "machine learning", "deep learning",
          "tensorflow", "pytorch", "nlp", "react", "javascript", "aws", "docker"]

def extract_skills(text):
    """Find skills mentioned in resume text."""
    text_lower = text.lower()
    return [s for s in SKILLS if s in text_lower]

def extract_email(text):
    """Find email address using regex pattern."""
    emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
    return emails[0] if emails else None

def parse_resume(pdf_path):
    """Parse a PDF resume and extract information."""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = "".join(page.extract_text() for page in reader.pages)
    return {"text": text, "skills": extract_skills(text), "email": extract_email(text)}
```

### 6.2 Model Training
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match(resume_text, job_text, resume_skills, job_skills):
    """Calculate match score between resume and job."""
    # Skill overlap score
    resume_set = set(s.lower() for s in resume_skills)
    job_set = set(s.lower() for s in job_skills)
    skill_score = len(resume_set & job_set) / max(len(job_set), 1)

    # Text similarity score
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    text_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # Combined score (50% skills + 50% text similarity)
    final = 0.5 * skill_score + 0.5 * text_score
    return {
        "overall_score": final,
        "skill_match": skill_score,
        "matched_skills": list(resume_set & job_set),
        "missing_skills": list(job_set - resume_set)
    }
```

### 6.3 Web Application
```python
import streamlit as st
import tempfile
from src.resume_parser import parse_resume
from src.matching import calculate_match

st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 AI Resume Screening & Job Matching")

# Sidebar: Job description input
st.sidebar.header("Job Description")
job_text = st.sidebar.text_area("Paste job description:", height=200)
job_skills = st.sidebar.text_input("Required skills (comma-separated):").split(",")

# Main: Resume upload
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=['pdf'], accept_multiple_files=True)

if uploaded_files and job_text:
    results = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file.read())
            resume = parse_resume(tmp.name)
            match = calculate_match(resume['text'], job_text, resume['skills'], job_skills)
            match['filename'] = file.name
            results.append(match)

    results.sort(key=lambda x: x['overall_score'], reverse=True)

    for r in results:
        with st.expander(f"📁 {r['filename']} — Score: {r['overall_score']:.1%}"):
            st.write(f"**Matched Skills:** {', '.join(r['matched_skills'])}")
            st.write(f"**Missing Skills:** {', '.join(r['missing_skills'])}")
            st.progress(r['overall_score'])
```

---

## 7. Expected Results

| Metric | Target |
|--------|--------|
| Skill matching accuracy | 85%+ |
| Processing time | < 1 second per resume |
| App response time | < 3 seconds |

---

## 8. Common Mistakes & Solutions

| Problem | Solution |
|---------|----------|
| PDF extraction fails | Install pdfplumber: pip install pdfplumber, use as fallback |
| spaCy model not found | Run: python -m spacy download en_core_web_sm |
| Low matching accuracy | Add more skills to the SKILLS list, use sentence-transformers for better similarity |


---

## 9. Free Deployment

### Deploy to Hugging Face Spaces:

1. **Push code to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/01-resume-screening-job-matching.git
git push -u origin main
```

2. **Create Space at huggingface.co/spaces:**
   - Click "Create new Space"
   - Choose **Streamlit** SDK
   - Upload: `app.py`, `requirements.txt`, `models/`

3. **Your app is live!**
   - URL: `https://YOUR_USERNAME-01-resume-screening-job-matching.hf.space`

### requirements.txt:
```
streamlit
spacy
scikit-learn
PyPDF2
sentence-transformers
pandas
numpy
plotly
streamlit>=1.28.0
```

---

## 10. What to Submit

- [ ] GitHub repository with all code
- [ ] README.md documentation
- [ ] Live deployment URL
- [ ] Project report (10-15 pages)
- [ ] Demo video (5 minutes)

### Report Structure:
1. Introduction (1 page)
2. Literature Review (2 pages)
3. Dataset Description (1 page)
4. Methodology (3 pages)
5. Results & Discussion (3 pages)
6. Conclusion (1 page)
7. References

---

## 📁 Project Structure

```
01-resume-screening-job-matching/
├── app.py              ← Streamlit web app
├── exploration.py      ← Data exploration
├── preprocessing.py    ← Data cleaning
├── train_model.py      ← Model training
├── requirements.txt    ← Dependencies
├── README.md           ← Documentation
├── data/               ← Datasets
├── models/             ← Saved models
└── docs/               ← Visualizations
```

---

> **💡 Tip:** Start with 10 sample resumes and 3 job descriptions. Get the pipeline working end-to-end first, then scale up.
