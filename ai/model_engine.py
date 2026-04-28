from transformers import AutoModelForSequenceClassification
import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_model')
BERT_DIR = os.path.join(MODEL_DIR, 'bert_grading_model')



xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_hybrid.pkl'))
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
diff_enc = joblib.load(os.path.join(MODEL_DIR, 'enc_difficulty.pkl'))
topic_enc = joblib.load(os.path.join(MODEL_DIR, 'enc_topic.pkl'))
track_enc = joblib.load(os.path.join(MODEL_DIR, 'enc_track.pkl'))

DATA_PATH = os.path.join(MODEL_DIR, 'assessment_dataset.csv')
df = pd.read_csv(DATA_PATH, encoding="latin-1")


print("✅ All models loaded successfully")

print(BASE_DIR)


