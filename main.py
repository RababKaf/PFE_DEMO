
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import numpy as np

API_KEY = "U_Have_Mybee"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
custom_objects = {"Orthogonal": tf.keras.initializers.Orthogonal}
model = load_model("notre_modele.keras")

# Load AraBERT tokenizer
arabert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv01")

# Constants
VOCAB_SHIFT = 64000

def preprocess_text(text):
    # Keep only letters and digits
    text = re.sub(r'[^\w\s]', '', text)

    max_vocab = 176000
    max_sequence_length = 160
    tokenizer_wlt = Tokenizer(num_words=max_vocab)
    tokenizer_wlt.fit_on_texts(text)

    encoded = arabert_tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="np"
    )
    arabert_ids = encoded["input_ids"][0]

    seq = tokenizer_wlt.texts_to_sequences([text])
    wlt_padded = pad_sequences(seq, maxlen=max_sequence_length, padding='post')[0]
    wlt_shifted = wlt_padded + VOCAB_SHIFT

    merged = np.concatenate([arabert_ids, wlt_shifted])
    return np.expand_dims(merged, axis=0)

def predict_text(text):
    input_vec = preprocess_text(text)
    pred = model.predict(input_vec)[0][0]
    label = "خبر زائف" if pred > 0.5 else "خبر حقيقي"
    confidence = float(pred)
    return label, confidence

@app.post("/predict")
async def predict(request: Request, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    data = await request.json()
    text = data.get("text", "")
    if not text.strip():
        return {"error": "نص فارغ"}
    label, confidence = predict_text(text)
    return {"label": label}
