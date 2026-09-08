import streamlit as st
import tempfile
from src.resume_parser import parse_resume
from src.matching import calculate_match

st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 AI Resume Screening & Job Matching")

st.sidebar.header("Job Description")
job_text = st.sidebar.text_area("Paste job description:", height=200)
job_skills_input = st.sidebar.text_input("Required skills (comma-separated):")
job_skills = [s.strip() for s in job_skills_input.split(",")] if job_skills_input else []

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