from .model_engine import diff_enc, topic_enc, track_enc, bert_model, xgb, scaler, df

import numpy as np
import re
import random
import torch
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
)






AVAILABLE_TRACKS = ["Data Analysis", "Front-End"]

STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","can","could","should","may","might",
    "of","in","on","at","to","for","with","by","from","up","about","into",
    "it","its","this","that","and","or","not","but","so","if","as","we","you","i"
}


# 2. BERT CROSS-ENCODER

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 256
BATCH_SIZE = 16
EPOCHS     = 2
LR         = 2e-5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def build_input(row):
    return (
        str(row["question_text"]) + " [SEP] " +
        str(row["model_answer"])  + " [SEP] " +
        str(row["student_answer"])
    )

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )



bert_model.eval()

# 5. HAND-CRAFTED FEATURES
# ============================================================

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

    len_ratio    = min(len(s_words) / max(len(m_words), 1), 1.0)
    abs_len      = min(len(s_words) / 100.0, 1.0)

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







def is_meaningful_answer(student_answer, min_meaningful_words=3):

    if not student_answer or not isinstance(student_answer, str):
        return False
    text = student_answer.strip().lower()
    if not text:
        return False
    words = re.findall(r"[a-zA-Z]{2,}", text)
    meaningful = [w for w in words if w not in STOP_WORDS]
    return len(meaningful) >= min_meaningful_words


def is_relevant_answer(question, model_answer, student_answer, threshold=0.5):

    ref_text = f"{question} [SEP] {model_answer}"

    ref_enc = tokenizer(ref_text, truncation=True, padding=True,
                        max_length=MAX_LEN, return_tensors="pt")
    stu_enc = tokenizer(student_answer, truncation=True, padding=True,
                        max_length=MAX_LEN, return_tensors="pt")

    bert_model.eval()
    with torch.no_grad():

        ref_emb = bert_model.distilbert(**ref_enc).last_hidden_state[:, 0, :]
        stu_emb = bert_model.distilbert(**stu_enc).last_hidden_state[:, 0, :]

    cos_sim = torch.nn.functional.cosine_similarity(ref_emb, stu_emb).item()
    return cos_sim >= threshold



def select_questions(track, n_per_topic=2):

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








def grade_answer(question, model_answer, student_answer,
                difficulty="medium", topic="General", track="Front-End"):

    if not is_meaningful_answer(student_answer):
        return {
            "prediction": 0,
            "label":      "Wrong",
            "confidence": {"wrong": 1.0, "partial": 0.0, "correct": 0.0},
            "note": "Answer rejected: no meaningful content."
        }


    if not is_relevant_answer(question, model_answer, student_answer, threshold=0.5):
        return {
            "prediction": 0,
            "label":      "Wrong",
            "confidence": {"wrong": 1.0, "partial": 0.0, "correct": 0.0},
            "note": "Answer rejected: not relevant to the question."
        }

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

    if label == "Correct":
        return round(random.uniform(9, 10), 1)
    elif label == "Partial":
        return round(random.uniform(6, 8), 1)
    else:
        return round(random.uniform(0, 3), 1)


def get_level(percentage):
    """percentage من 0 لـ 100"""
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
        topic  = r["topic_area"]
        weight = difficulty_weights.get(r["difficulty_level"], 2)
        if topic not in topic_scores:
            topic_scores[topic] = {"score": 0, "max": 0}
        topic_scores[topic]["score"] += r["q_score"] * weight
        topic_scores[topic]["max"]   += 10 * weight

    topic_results = {}
    for topic, vals in topic_scores.items():
        pct = (vals["score"] / vals["max"] * 100) if vals["max"] > 0 else 0
        topic_results[topic] = {
            "level": get_level(pct),
            "score": round(pct, 1)
        }

    return topic_results



# النتيجة النهائية الكلية


def get_overall_result(results: list, track: str) -> dict:

    difficulty_weights = {"easy": 1, "medium": 2, "hard": 3}
    total_score = 0
    total_max   = 0

    for r in results:
        weight       = difficulty_weights.get(r["difficulty_level"], 2)
        total_score += r["q_score"] * weight
        total_max   += 10 * weight

    overall_pct = (total_score / total_max * 100) if total_max > 0 else 0

    question_breakdown = []
    for i, r in enumerate(results, 1):
        question_breakdown.append({
            "question_number": i,
            "topic":           r["topic_area"],
            "difficulty":      r["difficulty_level"],
            "label":           r["label"],
            "score":           round(r["q_score"], 1),
            "confidence":      r.get("confidence", {})
        })

    return {
        "track": track,
        "overall": {
            "level": get_level(overall_pct),
            "score": round(overall_pct, 1)
        },
        "question_breakdown": question_breakdown
    }
