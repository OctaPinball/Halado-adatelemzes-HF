import torch
import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator
from config import LOG_PATH, RESULTS_PATH
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate(model, model_name, X_test, y_test, device=None, log_transform=False):
    print(f"{model_name} modell kiértékelése...")

    # Előrejelzés
    if isinstance(model, torch.nn.Module):
        assert device is not None, "Device megadása kötelező PyTorch modellhez."
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test.to(device)
            y_pred = model(X_test_tensor).cpu().numpy()
            y_true = y_test.cpu().numpy()

    elif isinstance(model, BaseEstimator):
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.numpy()
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.numpy()

        y_pred = model.predict(X_test)
        y_true = y_test

    else:
        raise ValueError("A modell típusa nem támogatott (csak PyTorch vagy sklearn lehet).")

    # Ha log-transzformált az y (pl. log1p), visszaalakítás expm1-gyel
    if log_transform:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)

    # Metrikák számítása sklearn-mel
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE kiszámítása csak, ha nincs nulla a valós adatok között
    if np.any(y_true == 0):
        mape = float('nan')
    else:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Eredmények mentése
    results = {
        "model": model_name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2
    }

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    results_df = pd.DataFrame([results])
    log_file = os.path.join(LOG_PATH, "evaluation_results.csv")

    if os.path.exists(log_file):
        old_results = pd.read_csv(log_file)
        old_results = old_results[old_results["model"] != model_name]
        results_df = pd.concat([old_results, results_df], ignore_index=True)

    results_df.to_csv(log_file, index=False)

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    predictions_df = pd.DataFrame({
        "actual": y_true.flatten(),
        "predicted": y_pred.flatten()
    })
    predictions_df.to_csv(os.path.join(RESULTS_PATH, f"{model_name}_predictions.csv"), index=False)

    # Kiírás részletesen
    try:
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R^2:  {r2:.3f}")
        print(f"MAE:  {mae:.1f} M Ft")
        print(f"RMSE: {rmse:.1f} M Ft")

        # Scatter plot log-log skálán (mint a jó kódodban)
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.3)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, '--', color='gray')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Actual Price (M Ft; log)')
        plt.ylabel('Predicted Price (M Ft; log)')
        plt.title(f'{model_name}: Actual vs Predicted')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Plot vagy bővített metrikák kiértékelése sikertelen: {e}")

    return results

