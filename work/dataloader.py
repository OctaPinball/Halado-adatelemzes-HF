import torch
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

from config import DATA_PATH, MODEL_PATH
from dataset import HousingDataset

def load_data():
    """
    Betölti és szétosztja az adatokat train, validation és test halmazokra.
    A target értékek log1p transzformáción esnek át.
    """
    try:
        data = pd.read_csv(os.path.join(DATA_PATH, "processed_data.csv"))
        
        # Target és feature-ök szétválasztása
        y = data["price_mill_ft"].values
        X = data.drop("price_mill_ft", axis=1).values
        
        # Train-validation-test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # Normalizálás
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Target log1p transzformáció
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
        y_test = np.log1p(y_test)
        
        # Scaler mentése
        joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
        
        # Torch tensorrá alakítás
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        
        input_size = X_train.shape[1]
        
        return X_train, y_train, X_val, y_val, X_test, y_test, input_size
        
    except Exception as e:
        print(f"Hiba az adatok betöltése közben: {e}")
        sys.exit(1)


def get_data_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = HousingDataset(X_train, y_train)
    val_dataset = HousingDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader