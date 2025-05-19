import torch
import torch.nn as nn
import os
import joblib  # sklearn modellek mentéséhez
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import numpy as np

from config import MODEL_PATH


def train(model, model_name, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    if isinstance(model, torch.nn.Module):
        print(f"{model_name} (PyTorch) modell tanítása...")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(MODEL_PATH):
                    os.makedirs(MODEL_PATH)
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}.pth"))
                print(f"Modell mentve: {model_name}.pth")

        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{model_name}.pth")))
        return model

    elif isinstance(model, BaseEstimator):
        print(f"{model_name} (sklearn) modell tanítása...")

        # train_loader → numpy array
        X_train, y_train = _loader_to_numpy(train_loader)
        X_val, y_val = _loader_to_numpy(val_loader)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, y_pred)
        print(f"Validation Loss: {val_loss:.4f}")

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        model_path = os.path.join(MODEL_PATH, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Sklearn modell mentve: {model_path}")

        return model

    else:
        raise ValueError("A modell típusa nem támogatott. Csak PyTorch vagy sklearn modelleket kezel.")


def _loader_to_numpy(loader):
    X_list, y_list = [], []
    for X_batch, y_batch in loader:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy().reshape(-1))
    return np.vstack(X_list), np.hstack(y_list)

