from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match(resume_text, job_text, resume_skills, job_skills):
    resume_set = set(s.lower() for s in resume_skills)
    job_set = set(s.lower() for s in job_skills)
    skill_score = len(resume_set & job_set) / max(len(job_set), 1)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    text_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    final = 0.5 * skill_score + 0.5 * text_score
    return {
        "overall_score": final,
        "skill_match": skill_score,
        "matched_skills": list(resume_set & job_set),
        "missing_skills": list(job_set - resume_set)
    }