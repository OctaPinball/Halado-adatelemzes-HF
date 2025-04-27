import torch
import os
import joblib

from config import MODEL_PATH


def predict(model, model_name, X_input, device):
    """
    Új adatokon végzett predikció
    """
    print(f"Predikció a {model_name} modellel...")
    
    model = model.to(device)
    model.eval()
    
    # Scaler betöltése
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    
    # Input normalizálása
    X_scaled = scaler.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    # Előrejelzés
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    return predictions