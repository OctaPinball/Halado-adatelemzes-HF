import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Konstansok
MODEL_PATH = "models"
DATA_PATH = "data"
LOG_PATH = "logs"
RESULTS_PATH = "results"

# Modellek definíciója
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.fc(x)

class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# Adatelőkészítés
def preprocess(force=False):
    """
    Adatelőkészítés és előfeldolgozás
    """
    print("Adatok előkészítése...")
    
    if os.path.exists(os.path.join(DATA_PATH, "processed_data.csv")) and not force:
        print("Az előfeldolgozott adatok már léteznek. Kihagyás...")
        return
    
    # Adatok betöltése
    try:
        raw_data = pd.read_csv(os.path.join(DATA_PATH, "raw_housing_data.csv"))
        
        # Hiányzó értékek kezelése
        raw_data = raw_data.dropna()
        
        # Kategórikus változók kezelése
        categorical_cols = ["location", "property_type"]
        raw_data = pd.get_dummies(raw_data, columns=categorical_cols)
        
        # Adatok mentése
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            
        raw_data.to_csv(os.path.join(DATA_PATH, "processed_data.csv"), index=False)
        print("Adatok előfeldolgozva és mentve.")
        
    except Exception as e:
        print(f"Hiba az adatok előfeldolgozása közben: {e}")
        sys.exit(1)

# Adat betöltő függvények
def load_data():
    """
    Betölti és szétosztja az adatokat train, validation és test halmazokra
    """
    try:
        data = pd.read_csv(os.path.join(DATA_PATH, "processed_data.csv"))
        
        # Target és feature-ök szétválasztása
        y = data["price"].values
        X = data.drop("price", axis=1).values
        
        # Train-validation-test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        # Normalizálás
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
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

# Dataset és DataLoader
class HousingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

# Model training
def train(model, model_name, train_loader, val_loader, device, num_epochs, learning_rate=0.001):
    """
    Model training
    """
    print(f"{model_name} modell tanítása...")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward és optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
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
        
        # Legjobb modell mentése
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}.pth"))
            print(f"Modell mentve: {model_name}.pth")
    
    # Betöltjük a legjobb modellt
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{model_name}.pth")))
    return model

# Evaluate
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

# Predict
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

# Fő pipeline
def main():
    # Argumentumok kezelése
    parser = argparse.ArgumentParser(description='Ingatlanár predikciós pipeline')
    
    # Előfeldolgozási paraméterek
    parser.add_argument('-p', '--preprocess', choices=['force', 'skip', 'run'], default='run',
                      help='Előfeldolgozási mód: force (erőltetett újrafeldolgozás), skip (kihagyás), run (futtatás ha szükséges)')
    
    # Modell paraméterek
    parser.add_argument('-m', '--models', nargs='+', choices=['linear', 'mlp', 'all'], default=['all'],
                      help='Használandó modellek')
    
    # Betöltési paraméter
    parser.add_argument('-l', '--load', nargs='+',
                      help='Betöltendő modellek neve')
    
    # Tanítási paraméterek
    parser.add_argument('-t', '--train', action='store_true',
                      help='Modell tanítása')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                      help='Epochok száma')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='Batch méret')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                      help='Tanulási ráta')
    
    # Kiértékelési paraméterek
    parser.add_argument('-ev', '--evaluate', action='store_true',
                      help='Modell kiértékelése')
    
    # Predikciós paraméterek
    parser.add_argument('-pr', '--predict', action='store_true',
                      help='Új adatokon való predikció')
    parser.add_argument('-i', '--input', type=str,
                      help='Input CSV fájl a predikcióhoz')

    args = parser.parse_args()
    
    # Device beállítása
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Használt eszköz: {device}")
    
    # Előfeldolgozás
    if args.preprocess == 'force':
        preprocess(force=True)
    elif args.preprocess == 'run':
        preprocess(force=False)
    else:  # skip
        print("Előfeldolgozás kihagyva.")
    
    # Modellek kiválasztása
    available_models = {
        'linear': LinearModel,
        'mlp': MLPModel
    }
    
    # Ha 'all' van megadva, minden modellt használunk
    if 'all' in args.models:
        selected_models = list(available_models.keys())
    else:
        selected_models = args.models
    
    # Adatok betöltése
    if args.train or args.evaluate:
        X_train, y_train, X_val, y_val, X_test, y_test, input_size = load_data()
        train_loader, val_loader = get_data_loaders(X_train, y_train, X_val, y_val, args.batch_size)
    
    # Modellek inicializálása és betöltése
    models = {}
    for model_name in selected_models:
        if model_name in available_models:
            models[model_name] = available_models[model_name](input_size)
    
    # Ha van betöltendő modell
    if args.load:
        for model_path in args.load:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            if model_name in models:
                try:
                    models[model_name].load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{model_path}.pth")))
                    print(f"{model_name} modell betöltve.")
                except:
                    print(f"Hiba a {model_name} modell betöltésekor.")
    
    # Training
    if args.train:
        for model_name, model in models.items():
            models[model_name] = train(
                model, model_name, train_loader, val_loader, device,
                args.epochs, args.learning_rate
            )
    
    # Evaluation
    if args.evaluate:
        for model_name, model in models.items():
            evaluate(model, model_name, X_test, y_test, device)
    
    # Prediction
    if args.predict and args.input:
        try:
            input_data = pd.read_csv(args.input)
            # Kategórikus változók átalakítása
            categorical_cols = ["location", "property_type"]  # Példa alapján
            input_data = pd.get_dummies(input_data, columns=categorical_cols)
            
            # Hiányzó oszlopok hozzáadása
            processed_data = pd.read_csv(os.path.join(DATA_PATH, "processed_data.csv"))
            feature_cols = processed_data.drop("price", axis=1).columns
            
            for col in feature_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Szükségtelen oszlopok eltávolítása
            for col in input_data.columns:
                if col not in feature_cols:
                    input_data = input_data.drop(col, axis=1)
            
            # Oszlopok sorrendjének rendezése
            input_data = input_data[feature_cols]
            
            # Predikció minden modellel
            all_predictions = {}
            for model_name, model in models.items():
                predictions = predict(model, model_name, input_data.values, device)
                all_predictions[f"{model_name}_pred"] = predictions.flatten()
            
            # Eredmények összefűzése
            results_df = input_data.copy()
            for model_name, preds in all_predictions.items():
                results_df[model_name] = preds
            
            # Eredmények mentése
            if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)
            
            results_df.to_csv(os.path.join(RESULTS_PATH, "predictions.csv"), index=False)
            print(f"Predikciók mentve: {os.path.join(RESULTS_PATH, 'predictions.csv')}")
            
        except Exception as e:
            print(f"Hiba a predikció során: {e}")


if __name__ == "__main__":
    main()