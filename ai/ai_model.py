from .model_engine import diff_enc, topic_enc, track_enc, bert_model, xgb, scaler, df ,MODEL_DIR
import numpy as np
import re
import random
import torch
import os


from transformers import (
    AutoTokenizer,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 256


tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_DIR, "bert_grading_model")
)
def extract_keywords(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    stop = {
        "the","a","an","is","are","was","were","be","been","have","has","had",
        "do","does","did","will","would","can","could","should","may","might",
        "of","in","on","at","to","for","with","by","from","up","about","into",
        "it","its","this","that","and","or","not","but","so","if","as","we","you","i"
    }
    return set(text.split()) - stop


def hand_features(row):
    s_kw = extract_keywords(row["student_answer"])
    m_kw = extract_keywords(row["model_answer"])
    q_kw = extract_keywords(row["question_text"])

    s_words = str(row["student_answer"]).split()
    m_words = str(row["model_answer"]).split()

    kw_recall    = len(s_kw & m_kw) / max(len(m_kw), 1)
    kw_precision = len(s_kw & m_kw) / max(len(s_kw), 1)
    q_relevance  = len(s_kw & q_kw) / max(len(q_kw), 1)
    len_ratio    = min(len(s_words) / max(len(m_words), 1), 2.0)
    abs_len      = min(len(s_words) / 50.0, 1.0)

    bigram_s = set(zip(s_words, s_words[1:])) if len(s_words) > 1 else set()
    bigram_m = set(zip(m_words, m_words[1:])) if len(m_words) > 1 else set()
    bigram_overlap = len(bigram_s & bigram_m) / max(len(bigram_m), 1)
    excess = max(len(s_words) - len(m_words), 0) / max(len(m_words), 1)

    difficulty = diff_enc.transform([row["difficulty_level"]])[0] / 3.0
    topic      = topic_enc.transform([row["topic_area"]])[0]      / 10.0
    track      = track_enc.transform([row["track"]])[0]           / 2.0

    return [
        kw_recall, kw_precision, q_relevance,
        len_ratio, abs_len, bigram_overlap, excess,
        difficulty, topic, track
    ]

def build_hand_matrix(subset_df):
    return np.array([hand_features(r) for _, r in subset_df.iterrows()])

def select_questions(track, n_per_topic=2):
    """
    يختار n_per_topic أسئلة randomly من كل topic
    من الـ track المحدد فقط — max 10
    """
    questions_bank = (
        df[["question_id", "question_text", "model_answer",
            "difficulty_level", "topic_area", "track"]]
        .drop_duplicates(subset="question_id")
        .reset_index(drop=True)
    )

    track_qs = questions_bank[questions_bank["track"] == track]

    if len(track_qs) == 0:
        raise ValueError(f"No questions found for track: {track}")

    topics   = track_qs["topic_area"].unique().tolist()
    selected = []

    for topic in topics:
        topic_qs = track_qs[track_qs["topic_area"] == topic]
        sampled  = topic_qs.sample(min(n_per_topic, len(topic_qs)), random_state=None)
        for _, row in sampled.iterrows():
            selected.append(row.to_dict())

    random.shuffle(selected)
    return selected[:10]


def grade_answer(question, model_answer, student_answer, difficulty="medium", topic="General", track="Front-End"):
    """يقيّم إجابة واحدة"""
    text = f"{question} [SEP] {model_answer} [SEP] {student_answer}"
    enc  = tokenizer(text, truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="pt")
    bert_model.eval()
    with torch.no_grad():
        logits = bert_model(**enc).logits
    bert_probs_single = torch.softmax(logits, dim=-1).numpy()[0]

    row = {
        "question_text":    question,
        "model_answer":     model_answer,
        "student_answer":   student_answer,
        "difficulty_level": difficulty if difficulty in diff_enc.classes_ else "medium",
        "topic_area":       topic if topic in topic_enc.classes_           else topic_enc.classes_[0],
        "track":            track if track in track_enc.classes_           else track_enc.classes_[0],
    }
    h_feats = scaler.transform([hand_features(row)])
    X       = np.hstack([bert_probs_single.reshape(1, -1), h_feats])

    pred  = xgb.predict(X)[0]
    proba = xgb.predict_proba(X)[0]

    label_map = {0: "Wrong", 1: "Partial", 2: "Correct"}
    return {
        "prediction": int(pred),
        "label":      label_map[int(pred)],
        "confidence": {
            "wrong":   round(float(proba[0]), 3),
            "partial": round(float(proba[1]), 3),
            "correct": round(float(proba[2]), 3),
        }
    }


def get_question_score(label):
    """
    يحوّل الـ label لـ score من 10 بناءً على الفهم:
    Correct  → random من 9  لـ 10   (فهم كامل)
    Partial  → random من 6  لـ 8    (فهم جزئي)
    Wrong    → random من 0  لـ 3    (مفيش فهم)
    """
    if label == "Correct":
        return round(random.uniform(9, 10), 1)
    elif label == "Partial":
        return round(random.uniform(6, 8), 1)
    else:  # Wrong
        return round(random.uniform(0, 3), 1)


def get_level(percentage):
    """
    percentage هنا من 0 لـ 100 بناءً على الـ weighted score
    """
    if percentage < 40:
        return "Beginner"
    elif percentage < 70:
        return "Intermediate"
    else:
        return "Advanced"
    


def get_topic_results(results: list) -> dict:
    difficulty_weights = {"easy": 1, "medium": 2, "hard": 3}
    topic_scores = {}

    for r in results:
        topic = r["topic_area"]
        weight = difficulty_weights.get(r["difficulty_level"], 2)

        if topic not in topic_scores:
            topic_scores[topic] = {"score": 0, "max": 0}

        topic_scores[topic]["score"] += r["q_score"] * weight
        topic_scores[topic]["max"] += 10 * weight

    topic_results = {}

    for topic, vals in topic_scores.items():
        pct = (vals["score"] / vals["max"] * 100) if vals["max"] > 0 else 0
        topic_results[topic] = {
            "level": get_level(pct),
            "score": round(pct, 1)
        }

    return topic_results




def get_overall_result(results: list, track: str) -> dict:
    difficulty_weights = {"easy": 1, "medium": 2, "hard": 3}
    total_score = 0
    total_max = 0

    for r in results:
        weight = difficulty_weights.get(r["difficulty_level"], 2)
        total_score += r["q_score"] * weight
        total_max += 10 * weight

    overall_pct = (total_score / total_max * 100) if total_max > 0 else 0

    question_breakdown = []

    for i, r in enumerate(results, 1):
        question_breakdown.append({
            "question_number": i,
            "topic": r["topic_area"],
            "difficulty": r["difficulty_level"],
            "label": r["label"],
            "score": round(r["q_score"], 1),
            "confidence": r.get("confidence", {})
        })

    return {
        "track": track,
        "overall": {
            "level": get_level(overall_pct),
            "score": round(overall_pct, 1)
        },
        "question_breakdown": question_breakdown
    }
