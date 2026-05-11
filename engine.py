"""
Resume Matching Engine — Redrob AI Campus Hackathon
====================================================
Builds TF-IDF vectors for 10 resumes, binary vectors for 3 JDs,
computes cosine similarity, and outputs Top 3 candidates per JD.

Uses only standard libraries (no numpy, pandas, scikit-learn).
"""

import math
import re
from collections import OrderedDict

# ─────────────────────────────────────────────────────────────
# SKILL_ALIASES  (provided verbatim — do NOT modify)
# ─────────────────────────────────────────────────────────────
SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",
    # Web — Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",
    # Web — Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android",
    "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# ─────────────────────────────────────────────────────────────
# RESUME DATASET  — 10 Candidates
# ─────────────────────────────────────────────────────────────
RESUMES = {
    "Arjun Sharma":   "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",
    "Priya Nair":     "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",
    "Rahul Gupta":    "Java, Spring Boot, MySql, Microservices, Docker, kubernates",
    "Sneha Patel":    "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",
    "Vikram Singh":   "C++, Algoritms, Data Structure, competitive programming, python",
    "Ananya Krishnan":"javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",
    "Karan Mehta":    "Python, Sklearn, XGboost, feature engineering, SQL, tableau",
    "Deepika Rao":    "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",
    "Aditya Kumar":   "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",
    "Meera Iyer":     "python, R, statistics, ML, regression, clustering, Power-BI",
}

# ─────────────────────────────────────────────────────────────
# JOB DESCRIPTION DATASET  — 3 JDs
# (Required + Preferred skills combined for matching)
# ─────────────────────────────────────────────────────────────
JOB_DESCRIPTIONS = {
    "JD-1 — Kakao (ML Engineer)": [
        "Python", "Machine Learning", "Deep Learning", "TensorFlow",
        "PyTorch", "SQL", "Data Visualization",
        "NLP", "BERT", "Feature Engineering", "Statistics",
    ],
    "JD-2 — Naver (Backend Engineer)": [
        "Java", "Spring Boot", "MySQL", "PostgreSQL",
        "Microservices", "Docker", "Kubernetes",
        "REST API", "CI/CD", "Redis",
    ],
    "JD-3 — Line (Frontend Engineer)": [
        "JavaScript", "React", "Vue", "TypeScript", "REST API", "HTML/CSS",
        "Node.js", "GraphQL", "Redux", "Jest", "AWS",
    ],
}


# ═════════════════════════════════════════════════════════════
# STEP 1 & 2 — Normalize + Deduplicate Skills
# ═════════════════════════════════════════════════════════════

def _build_multi_word_aliases():
    """
    Separate multi-word alias keys from single-token keys.
    Multi-word phrases must be matched BEFORE token-level processing.
    """
    multi = {}
    single = {}
    for raw, canon in SKILL_ALIASES.items():
        if " " in raw or "/" in raw or "-" in raw:
            multi[raw] = canon
        else:
            single[raw] = canon
    # Sort multi-word by length descending so longer phrases match first
    multi = OrderedDict(
        sorted(multi.items(), key=lambda kv: len(kv[0]), reverse=True)
    )
    return multi, single


MULTI_WORD_ALIASES, SINGLE_TOKEN_ALIASES = _build_multi_word_aliases()


def normalize_skills(raw_skills_str: str) -> list[str]:
    """
    Normalize a raw comma-separated skill string into a deduplicated
    list of canonical skill names.

    Pipeline:
      1. Lowercase the entire string
      2. Match & consume multi-word / special-char phrases first
      3. Split remaining text on commas → single tokens
      4. Map each token through SKILL_ALIASES
      5. Discard anything not in the alias map
      6. Deduplicate (preserve first-seen order)
    """
    text = raw_skills_str.lower().strip()
    canonical_skills = []

    # --- Phase A: extract multi-word phrases ---
    for phrase, canon in MULTI_WORD_ALIASES.items():
        if phrase in text:
            canonical_skills.append(canon)
            text = text.replace(phrase, ",")  # consume the phrase

    # --- Phase B: process remaining single tokens ---
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    for tok in tokens:
        if tok in SINGLE_TOKEN_ALIASES:
            canonical_skills.append(SINGLE_TOKEN_ALIASES[tok])

    # --- Phase C: deduplicate while preserving order ---
    seen = set()
    deduped = []
    for skill in canonical_skills:
        if skill not in seen:
            seen.add(skill)
            deduped.append(skill)

    return deduped


# ═════════════════════════════════════════════════════════════
# STEP 3 — Build Vocabulary
# ═════════════════════════════════════════════════════════════

def build_vocabulary(all_resume_skills: dict[str, list[str]]) -> list[str]:
    """
    Create a shared, alphabetically sorted vocabulary from all
    normalized + deduplicated resume skills.
    """
    vocab = set()
    for skills in all_resume_skills.values():
        vocab.update(skills)
    return sorted(vocab)


# ═════════════════════════════════════════════════════════════
# STEP 4 — Compute TF-IDF Vectors for Resumes
# ═════════════════════════════════════════════════════════════

def compute_tf(skill: str, resume_skills: list[str]) -> float:
    """
    TF(skill, resume) = 1 / N   (after deduplication, each skill
    appears exactly once, so count = 1)
    N = total unique skills in the resume
    """
    if skill in resume_skills:
        return 1.0 / len(resume_skills)
    return 0.0


def compute_idf(skill: str, all_resume_skills: dict[str, list[str]],
                total_resumes: int = 10) -> float:
    """
    IDF(skill) = ln( 10 / df(skill) )
    df = number of resumes containing the skill
    No smoothing.
    """
    df = sum(1 for skills in all_resume_skills.values() if skill in skills)
    if df == 0:
        return 0.0
    return math.log(total_resumes / df)


def build_tfidf_vectors(all_resume_skills: dict[str, list[str]],
                        vocabulary: list[str]) -> dict[str, list[float]]:
    """
    For each resume, compute a TF-IDF vector aligned to the vocabulary.
    """
    vectors = {}
    for name, skills in all_resume_skills.items():
        vec = []
        for term in vocabulary:
            tf = compute_tf(term, skills)
            idf = compute_idf(term, all_resume_skills)
            vec.append(tf * idf)
        vectors[name] = vec
    return vectors


# ═════════════════════════════════════════════════════════════
# STEP 5 — Build JD Binary Vectors
# ═════════════════════════════════════════════════════════════

def normalize_jd_skills(raw_skills: list[str]) -> set[str]:
    """Normalize each JD skill through the alias map."""
    canonical = set()
    for skill in raw_skills:
        low = skill.lower().strip()
        # Try multi-word first
        matched = False
        for phrase, canon in MULTI_WORD_ALIASES.items():
            if phrase == low:
                canonical.add(canon)
                matched = True
                break
        if not matched and low in SINGLE_TOKEN_ALIASES:
            canonical.add(SINGLE_TOKEN_ALIASES[low])
    return canonical


def build_jd_binary_vector(jd_skills: set[str],
                           vocabulary: list[str]) -> list[float]:
    """
    Binary vector: 1.0 if the JD mentions the skill, 0.0 otherwise.
    """
    return [1.0 if term in jd_skills else 0.0 for term in vocabulary]


# ═════════════════════════════════════════════════════════════
# STEP 6 — Cosine Similarity & Ranking
# ═════════════════════════════════════════════════════════════

def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def magnitude(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine(A, B) = (A · B) / (|A| × |B|)
    """
    dot = dot_product(a, b)
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def rank_candidates(resume_vectors: dict[str, list[float]],
                    jd_vector: list[float],
                    top_n: int = 3) -> list[tuple[str, float]]:
    """
    Compute cosine similarity for every resume against a JD,
    sort descending by score, break ties alphabetically by name.
    Return top N.
    """
    scores = []
    for name, vec in resume_vectors.items():
        sim = cosine_similarity(vec, jd_vector)
        scores.append((name, round(sim, 2)))
    # Sort: primary = score descending, secondary = name ascending (ties)
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_n]


# ═════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════

def main():
    # ── Step 1 & 2: Normalize + Deduplicate ──────────────────
    all_resume_skills = {}
    print("=" * 65)
    print("  RESUME MATCHING ENGINE — Redrob AI Campus Hackathon")
    print("=" * 65)

    print("\n── Step 1 & 2: Skill Normalization + Deduplication ──\n")
    for name, raw in RESUMES.items():
        skills = normalize_skills(raw)
        all_resume_skills[name] = skills
        print(f"  {name:20s} → {skills}")

    # ── Step 3: Build Vocabulary ─────────────────────────────
    vocabulary = build_vocabulary(all_resume_skills)
    print(f"\n── Step 3: Shared Vocabulary ({len(vocabulary)} skills) ──\n")
    print(f"  {vocabulary}\n")

    # ── Step 4: Compute TF-IDF Vectors ───────────────────────
    resume_vectors = build_tfidf_vectors(all_resume_skills, vocabulary)
    print("── Step 4: TF-IDF Vectors (sample — first 5 terms) ──\n")
    for name, vec in resume_vectors.items():
        preview = [f"{v:.4f}" for v in vec[:5]]
        print(f"  {name:20s} → [{', '.join(preview)}, ...]")

    # ── Step 5 & 6: JD Vectors → Cosine Similarity → Rank ───
    print("\n" + "=" * 65)
    print("  RESULTS — Top 3 Candidates per Job Description")
    print("=" * 65)

    for jd_name, jd_raw_skills in JOB_DESCRIPTIONS.items():
        jd_canonical = normalize_jd_skills(jd_raw_skills)
        jd_vector = build_jd_binary_vector(jd_canonical, vocabulary)
        top3 = rank_candidates(resume_vectors, jd_vector)

        formatted = ", ".join(f"{name}({score})" for name, score in top3)
        print(f"\n  {jd_name}")
        print(f"  {formatted}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
