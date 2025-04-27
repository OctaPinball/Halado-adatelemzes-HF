import pandas as pd
import os
import sys


from config import DATA_PATH


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

        # Felesleges oszlopok eltávolítása
        columns_to_drop = ["listing_id", "price", "address"]
        raw_data = raw_data.drop(columns=columns_to_drop, errors='ignore')
        
        # Kategórikus változók kezelése
        categorical_cols = []
        raw_data = pd.get_dummies(raw_data, columns=categorical_cols)
        
        # Adatok mentése
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            
        raw_data.to_csv(os.path.join(DATA_PATH, "processed_data.csv"), index=False)
        print("Adatok előfeldolgozva és mentve.")
        
    except Exception as e:
        print(f"Hiba az adatok előfeldolgozása közben: {e}")
        sys.exit(1)