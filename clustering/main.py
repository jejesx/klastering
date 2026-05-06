from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json

app = FastAPI(title="PSO-KMeans Clustering API")

# Izinkan request dari React (Firebase Hosting)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti dengan domain Firebase Hosting saat production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model saat server start
with open("model_params.json", "r") as f:
    params = json.load(f)

MEAN = np.array(params["mean"])
STD  = np.array(params["std"])
CENTROIDS = np.array(params["centroids_zscore"])
FEATURE_ORDER = ["Usia", "BMI", "HbA1c", "HOMA2B", "HOMA2IR"]

# Label deskriptif tiap klaster (sesuaikan dengan interpretasi skripsimu)
CLUSTER_LABELS = {
    1: "Kontrol Glikemik Baik, Sensitivitas Insulin Rendah",
    2: "Usia Lanjut, BMI Normal",
    3: "HbA1c Tinggi, Resistensi Insulin Berat",
    4: "Resistensi Insulin Tinggi, Sekresi Beta-Cell Baik",
}

class InputData(BaseModel):
    Usia: float
    BMI: float
    HbA1c: float
    HOMA2B: float
    HOMA2IR: float

class PredictionResult(BaseModel):
    klaster: int
    label: str
    jarak_ke_centroid: dict

@app.get("/")
def root():
    return {"status": "ok", "model": "PSO-KMeans 4 Klaster"}

@app.post("/predict", response_model=PredictionResult)
def predict(data: InputData):
    # 1. Susun input sebagai array
    input_arr = np.array([
        data.Usia, data.BMI, data.HbA1c, data.HOMA2B, data.HOMA2IR
    ])

    # 2. Z-score normalisasi pakai mean & std dari data training
    input_zscore = (input_arr - MEAN) / STD

    # 3. Hitung jarak Euclidean ke tiap centroid
    distances = np.linalg.norm(CENTROIDS - input_zscore, axis=1)

    # 4. Ambil klaster dengan jarak terpendek (index + 1 karena klaster mulai dari 1)
    klaster = int(np.argmin(distances)) + 1

    return PredictionResult(
        klaster=klaster,
        label=CLUSTER_LABELS[klaster],
        jarak_ke_centroid={
            f"klaster_{i+1}": round(float(d), 4)
            for i, d in enumerate(distances)
        }
    )