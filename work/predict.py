import torch
import numpy as np
import os
import joblib
from sklearn.base import BaseEstimator
from config import MODEL_PATH


def predict(model, model_name, X_input, device):
    print(f"Predikció a {model_name} modellel...")

    # Scaler betöltése
    scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_input)
    else:
        print("Figyelmeztetés: scaler.pkl nem található, az adatok nem kerülnek normalizálásra.")
        X_scaled = X_input

    # PyTorch modell esetén
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()

    # sklearn modell esetén
    elif isinstance(model, BaseEstimator):
        predictions = model.predict(X_scaled)

    else:
        raise ValueError("A modell típusa nem támogatott (csak PyTorch vagy sklearn lehet).")

    return predictions
