import argparse
import torch
import pandas as pd
import os

from config import DATA_PATH, MODEL_PATH, RESULTS_PATH, available_models
from preprocess import preprocess
from dataloader import load_data, get_data_loaders
from train import train
from evaluate import evaluate
from predict import predict


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


