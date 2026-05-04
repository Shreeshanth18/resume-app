from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------
# SKILL DATABASE
# ---------------------------
SKILLS_DB = [
    "python","java","c++","html","css","javascript",
    "sql","machine learning","ai","react",
    "data analysis","node","flask","django"
]

# ---------------------------
# JOB DATABASE
# ---------------------------
JOB_DB = {
    "Data Scientist": "python machine learning data analysis ai statistics",
    "Web Developer": "html css javascript react frontend backend",
    "Software Developer": "java c++ problem solving algorithms",
    "Database Analyst": "sql database data management"
}

# ---------------------------
# PDF → TEXT
# ---------------------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text.lower()


# ---------------------------
# SKILL DETECTION
# ---------------------------
def detect_skills(text):
    return list(set([skill for skill in SKILLS_DB if skill in text]))


# ---------------------------
# SCORE ENGINE
# ---------------------------
def calculate_score(skills, text):
    score = len(skills) * 8

    if "project" in text:
        score += 10

    if "internship" in text:
        score += 10

    return min(score, 100)


# ---------------------------
# SUGGESTION ENGINE
# ---------------------------
def generate_suggestions(skills):
    suggestions = []

    if len(skills) < 5:
        suggestions.append("Add more technical skills")

    if "sql" not in skills:
        suggestions.append("Learn SQL for better opportunities")

    if "project" not in skills:
        suggestions.append("Add project experience")

    return suggestions


# ---------------------------
# AI JOB MATCHING (REAL LOGIC)
# ---------------------------
def match_jobs(text):
    results = {}

    for role, desc in JOB_DB.items():
        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform([text, desc])
        similarity = cosine_similarity(vectors)[0][1]
        results[role] = round(similarity * 100, 2)

    return results


# ---------------------------
# CATEGORY ENGINE
# ---------------------------
def detect_category(skills):
    if "machine learning" in skills:
        return "AI / Data Science"
    elif "html" in skills:
        return "Web Development"
    elif "java" in skills:
        return "Software Engineering"
    return "General Tech"


# ---------------------------
# MAIN ROUTE
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['resume']

        text = extract_text(file)

        skills = detect_skills(text)
        score = calculate_score(skills, text)
        suggestions = generate_suggestions(skills)
        jobs = match_jobs(text)
        category = detect_category(skills)

        return render_template(
            "index.html",
            score=score,
            skills=skills,
            suggestions=suggestions,
            jobs=jobs,
            category=category
        )

    return render_template("index.html")


app.run(host="0.0.0.0", port=10000)