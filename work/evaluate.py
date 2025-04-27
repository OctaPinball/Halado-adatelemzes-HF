import torch
import numpy as np
import pandas as pd
import os

from config import  LOG_PATH, RESULTS_PATH


def evaluate(model, model_name, X_test, y_test, device):
    """
    Modell kiértékelése
    """
    print(f"{model_name} modell kiértékelése...")
    
    model = model.to(device)
    model.eval()
    
    # Előrejelzés
    with torch.no_grad():
        X_test = X_test.to(device)
        y_pred = model(X_test).cpu().numpy()
        y_true = y_test.cpu().numpy()
    
    # Metrikák számítása
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Eredmények mentése
    results = {
        "model": model_name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }
    
    # Logolás
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    
    results_df = pd.DataFrame([results])
    
    # Ellenőrizzük, hogy létezik-e már log fájl
    log_file = os.path.join(LOG_PATH, "evaluation_results.csv")
    if os.path.exists(log_file):
        old_results = pd.read_csv(log_file)
        # Töröljük a régi eredményeket ugyanehhez a modellhez
        old_results = old_results[old_results["model"] != model_name]
        results_df = pd.concat([old_results, results_df])
    
    results_df.to_csv(log_file, index=False)
    
    # Eredmények mentése
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    
    # Mentjük az előrejelzéseket
    predictions_df = pd.DataFrame({
        "actual": y_true.flatten(),
        "predicted": y_pred.flatten()
    })
    predictions_df.to_csv(os.path.join(RESULTS_PATH, f"{model_name}_predictions.csv"), index=False)
    
    print(f"{model_name} kiértékelés eredményei:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return results