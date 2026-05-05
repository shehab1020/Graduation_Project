import pandas as pd
import os
import joblib
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_model')


clf = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

DATA_PATH = os.path.join(MODEL_DIR, 'ai_assessment_dataset_enhanced.csv')
df = pd.read_csv(DATA_PATH, encoding="latin-1")


print("✅ All models loaded successfully")

