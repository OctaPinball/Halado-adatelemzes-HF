import gradio as gr
import folium
import pandas as pd
import numpy as np
from folium.plugins import MarkerCluster, Draw
import os
import tempfile
from geopy.geocoders import Nominatim
import base64
from pathlib import Path
import torch
import argparse
import sys
import joblib
from sklearn.base import BaseEstimator

# Importálások a predikciós pipeline-ból
from config import DATA_PATH, MODEL_PATH, RESULTS_PATH, available_models
from preprocess import preprocess
from dataloader import load_data
from predict import predict

# Budapest középpontja (hozzávetőleges koordináták)
BUDAPEST_LAT = 47.497912
BUDAPEST_LON = 19.040235

def create_map(lat=BUDAPEST_LAT, lon=BUDAPEST_LON):
    """Budapest térkép létrehozása"""
    map_center = [lat, lon]
    m = folium.Map(location=map_center, zoom_start=12)
    
    # Hozzáadjuk a rajzoló eszközt, hogy a marker mozgatható legyen
    draw = Draw(
        draw_options={
            'polyline': False,
            'rectangle': False,
            'polygon': False,
            'circle': False,
            'circlemarker': False,
            'marker': True
        },
        edit_options={'edit': True}
    )
    draw.add_to(m)
    
    return m

def convert_boolean_features(data):
    """Boolean típusú jellemzők konvertálása 0 vagy 1 értékekké"""
    boolean_features = [
        'tegla_epitesu', 'panel', 'csuszozsalus', 'uj_epitesu',
        'ujszeru', 'felujitott', 'jo_allapotu', 'kozepes_allapotu',
        'felujitando', 'epitve_1950_elott', 'epitve_1950_1980',
        'epitve_2001_2010', 'epitve_2011_utan', 'udvari_beallo',
        'teremgarazs', 'onallo_garazs', 'utcan_parkolas'
    ]
    
    for feature in boolean_features:
        data[feature] = 1 if data[feature] else 0
    
    return data

def geocode_address(address):
    """Cím konvertálása szélességi és hosszúsági fokokba"""
    try:
        # A User-Agent megadása fontos a Nominatim használatához
        geolocator = Nominatim(user_agent="budapest_ingatlan_app")
        
        # Hozzáadjuk a "Budapest" kifejezést a pontosabb találat érdekében, ha még nincs benne
        if "budapest" not in address.lower():
            search_address = f"{address}, Budapest, Hungary"
        else:
            search_address = f"{address}, Hungary"
            
        location = geolocator.geocode(search_address)
        
        if location:
            return location.latitude, location.longitude
        else:
            # Ha nem találja a címet, Budapest központját adjuk vissza
            return BUDAPEST_LAT, BUDAPEST_LON
    except Exception as e:
        print(f"Geokódolási hiba: {e}")
        return BUDAPEST_LAT, BUDAPEST_LON

def add_property_to_map(m, property_data, lat, lon):
    """Hozzáadja az ingatlant a térképhez"""
    # Popup szöveg létrehozása az ingatlan adataival
    popup_text = f"""
    <b>Cím:</b> {property_data['address']}<br>
    <b>Méret:</b> {property_data['sqm']} m²<br>
    <b>Szobák száma:</b> {property_data['rooms']}<br>
    <b>Koordináták:</b> {lat:.6f}, {lon:.6f}<br>
    """
    
    # Popup létrehozása és hozzáadása a térképhez
    popup = folium.Popup(popup_text, max_width=300)
    folium.Marker(
        location=[lat, lon],
        popup=popup,
        icon=folium.Icon(color='blue'),
        draggable=True
    ).add_to(m)
    
    return m

def map_to_html_data(m):
    """Térkép konvertálása beágyazható HTML adattá"""
    # Ideiglenes fájl létrehozása és a térkép mentése
    temp_dir = tempfile.gettempdir()
    map_file = os.path.join(temp_dir, 'property_map.html')
    m.save(map_file)
    
    # HTML fájl tartalmának beolvasása
    with open(map_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Az iframe létrehozása a beágyazáshoz
    iframe_html = f"""
    <div style="width:100%; height:500px;">
        <iframe srcdoc="{html_content.replace('"', '&quot;')}" style="width:100%; height:100%; border:none;"></iframe>
    </div>
    """
    
    return iframe_html

def search_address(address):
    """Cím keresése és térkép frissítése"""
    lat, lon = geocode_address(address)
    m = create_map(lat, lon)
    
    # Marker elhelyezése a térképen
    folium.Marker(
        location=[lat, lon],
        popup=f"<b>{address}</b><br>Koordináták: {lat:.6f}, {lon:.6f}",
        icon=folium.Icon(color='red'),
        draggable=True
    ).add_to(m)
    
    map_html = map_to_html_data(m)
    
    return map_html, lat, lon

def prepare_data_for_prediction(property_data):
    """
    Előkészíti az adatokat a predikciós modell számára
    """
    # CSV formátumba konvertáljuk az adatokat
    data = pd.DataFrame([property_data])
    
    # Lokációt kerület alapján határozzuk meg (pl. cím alapján)
    # Itt egy egyszerűsített megoldást használunk, a valódi implementációban
    # a címből kellene kinyerni a kerületet
    district = get_district_from_address(property_data['address'])
    data['location'] = district
    
    # Ingatlan típus hozzáadása - példaként lakást feltételezünk
    data['property_type'] = 'apartment'
    
    # Ideiglenes fájlba mentjük az adatokat
    temp_dir = tempfile.gettempdir()
    input_file = os.path.join(temp_dir, 'input_property.csv')
    data.to_csv(input_file, index=False)
    
    return input_file

def get_district_from_address(address):
    """
    Meghatározza a kerületet a cím alapján
    """
    # Egyszerűsített implementáció - valós alkalmazásban komplexebb logika lenne
    address_lower = address.lower()
    
    # Kerület keresése a címben
    for i in range(1, 24):
        if f"{i}. kerület" in address_lower or f"{i}.kerület" in address_lower:
            return f"district_{i:02d}"
    
    # Ha nem találtunk kerületet, visszatérünk egy alapértelmezett értékkel
    return "district_05"  # Belváros mint alapértelmezett

def run_prediction_pipeline(input_file):
    """
    Futtatja a predikciós pipeline-t a megadott bemeneti fájlon
    """
    # Argumentumok beállítása a predikciós pipeline számára
    sys.argv = [
        'script_name',  # Ez nem számít, csak helykitöltő
        '--predict',
        '--input', input_file,
        '--models', 'all',  # Minden modellt használunk
        '--preprocess', 'skip'  # Előfeldolgozás kihagyása, mivel már feldolgozva van
    ]
    
    # Device beállítása
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modellek betöltése
    models = {}
    for model_name in available_models.keys():
        # Input size beállítása - ez valójában a betanított modellek adataiból jönne
        _, _, _, _, _, _, input_size = load_data()
        models[model_name] = available_models[model_name](input_size)
        try:
            # PyTorch vagy sklearn modell ellenőrzése és betöltése
            if isinstance(models[model_name], torch.nn.Module):
                models[model_name].load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{model_name}.pth")))
                print(f"{model_name} PyTorch modell betöltve.")
            elif isinstance(models[model_name], BaseEstimator):
                # sklearn modell betöltése joblib-bal
                models[model_name] = joblib.load(os.path.join(MODEL_PATH, f"{model_name}.pkl"))
                print(f"{model_name} sklearn modell betöltve.")
            else:
                print(f"Ismeretlen modell típus: {type(models[model_name])}")
        except Exception as e:
            print(f"Hiba a {model_name} modell betöltésekor: {e}")
    
    # Predikció futtatása
    try:
        input_data = pd.read_csv(input_file)
        # Kategórikus változók átalakítása
        categorical_cols = ["location", "property_type"]
        input_data = pd.get_dummies(input_data, columns=categorical_cols)
        
        # Hiányzó oszlopok hozzáadása
        processed_data = pd.read_csv(os.path.join(DATA_PATH, "processed_data.csv"))
        feature_cols = processed_data.drop("price_mill_ft", axis=1).columns
        
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
            all_predictions[model_name] = predictions.flatten()[0]  # Csak az első predikciót vesszük
        
        # Átlag számolása a predikciókból
        avg_prediction = np.mean(list(all_predictions.values()))
        
        # Formázott eredmény visszaadása
        formatted_result = f"Becsült érték: {avg_prediction:.2f} millió Ft\n\n"
        formatted_result += "Modellek szerinti becslések:\n"
        for model_name, pred in all_predictions.items():
            formatted_result += f"- {model_name}: {pred:.2f} millió Ft\n"
        
        return formatted_result
        
    except Exception as e:
        return f"Hiba a predikció során: {e}"

def process_inputs(
    address, lat, lon, sqm, rooms,
    tegla_epitesu, panel, csuszozsalus, uj_epitesu,
    ujszeru, felujitott, jo_allapotu, kozepes_allapotu,
    felujitando, epitve_1950_elott, epitve_1950_1980,
    epitve_2001_2010, epitve_2011_utan, udvari_beallo,
    teremgarazs, onallo_garazs, utcan_parkolas
):
    """Feldolgozza a felhasználói bemeneteket és futtatja a predikciós modellt"""
    # Összegyűjtjük az ingatlan adatait egy szótárba
    property_data = {
        'address': address,
        'latitude': lat,
        'longitude': lon,
        'sqm': sqm,
        'rooms': rooms,
        'tegla_epitesu': tegla_epitesu,
        'panel': panel,
        'csuszozsalus': csuszozsalus,
        'uj_epitesu': uj_epitesu,
        'ujszeru': ujszeru,
        'felujitott': felujitott,
        'jo_allapotu': jo_allapotu,
        'kozepes_allapotu': kozepes_allapotu,
        'felujitando': felujitando,
        'epitve_1950_elott': epitve_1950_elott,
        'epitve_1950_1980': epitve_1950_1980,
        'epitve_2001_2010': epitve_2001_2010,
        'epitve_2011_utan': epitve_2011_utan,
        'udvari_beallo': udvari_beallo,
        'teremgarazs': teremgarazs,
        'onallo_garazs': onallo_garazs,
        'utcan_parkolas': utcan_parkolas
    }
    
    # Konvertáljuk a boolean értékeket 0 vagy 1 értékekké
    property_data = convert_boolean_features(property_data)
    
    # Térkép készítése és ingatlan elhelyezése
    m = create_map(lat, lon)
    m = add_property_to_map(m, property_data, lat, lon)
    map_html = map_to_html_data(m)
    
    # Adatok előkészítése a predikcióhoz
    input_file = prepare_data_for_prediction(property_data)
    
    # Predikciós pipeline futtatása
    becsult_ar = run_prediction_pipeline(input_file)
    
    return map_html, becsult_ar

# Alaptérkép létrehozása az alkalmazás indításakor
initial_map = create_map()
initial_html = map_to_html_data(initial_map)

# Gradio felület létrehozása
with gr.Blocks(title="Budapest Ingatlan Értékelő") as app:
    gr.Markdown("# Budapest Ingatlan Értékelő Alkalmazás")
    gr.Markdown("### Adja meg a lakás adatait, és helyezze el a térképen")
    
    lat = gr.State(value=BUDAPEST_LAT)
    lon = gr.State(value=BUDAPEST_LON)
    
    with gr.Row():
        with gr.Column(scale=1):
            address = gr.Textbox(label="Cím", placeholder="Pl: Andrássy út 2, Budapest", interactive=True)
            search_btn = gr.Button("Cím keresése")
            
            sqm = gr.Number(label="Méret (m²)", value=50, interactive=True)
            rooms = gr.Number(label="Szobák száma", value=2, interactive=True)
            
            with gr.Accordion("Épület típusa", open=False):
                building_type = gr.Radio(
                    choices=["Tégla építésű", "Panel", "Csúszózsalus"],
                    label=""
                )
            
            with gr.Accordion("Állapot", open=False):
                condition = gr.Radio(
                    choices=["Új építésű", "Újszerű", "Felújított", "Jó állapotú", "Közepes állapotú", "Felújítandó"],
                    label=""
                )
            
            with gr.Accordion("Építés éve", open=False):
                built_year = gr.Radio(
                    choices=["1950 előtt", "1950-1980 között", "2001-2010 között", "2011 után"],
                    label=""
                )
            
            with gr.Accordion("Parkolás", open=False):
                parking = gr.Radio(
                    choices=["Udvari beálló", "Teremgarázs", "Önálló garázs", "Utcán parkolás"],
                    label=""
                )
            
            predict_btn = gr.Button("Érték becslése")
        
        with gr.Column(scale=2):
            map_output = gr.HTML(label="Térkép", value=initial_html)
            prediction = gr.Textbox(label="Becsült érték", interactive=False)
    
    # Cím keresés eseménykezelő
    search_btn.click(
        fn=search_address,
        inputs=[address],
        outputs=[map_output, lat, lon]
    )
    
    # Becslés eseménykezelő, előtte ellenőrzés
    def validate_and_process(
        address, lat, lon, sqm, rooms,
        building_type, condition, built_year, parking
    ):
        if not address:
            raise gr.Error("A cím megadása kötelező.")
        if sqm is None or sqm <= 0:
            raise gr.Error("A méretet meg kell adni, és pozitív számnak kell lennie.")
        if rooms is None or rooms <= 0:
            raise gr.Error("A szobák számát meg kell adni, és pozitív számnak kell lennie.")
        if not building_type:
            raise gr.Error("Válasszon épülettípust.")
        if not condition:
            raise gr.Error("Válasszon állapotot.")
        if not built_year:
            raise gr.Error("Válasszon építési időszakot.")
        if not parking:
            raise gr.Error("Válasszon parkolási lehetőséget.")

        # Átalakítás: checkboxok helyett a rádió értékek alapján állítunk be bool értékeket
        tegla_epitesu = building_type == "Tégla építésű"
        panel = building_type == "Panel"
        csuszozsalus = building_type == "Csúszózsalus"

        uj_epitesu = condition == "Új építésű"
        ujszeru = condition == "Újszerű"
        felujitott = condition == "Felújított"
        jo_allapotu = condition == "Jó állapotú"
        kozepes_allapotu = condition == "Közepes állapotú"
        felujitando = condition == "Felújítandó"

        epitve_1950_elott = built_year == "1950 előtt"
        epitve_1950_1980 = built_year == "1950-1980 között"
        epitve_2001_2010 = built_year == "2001-2010 között"
        epitve_2011_utan = built_year == "2011 után"

        udvari_beallo = parking == "Udvari beálló"
        teremgarazs = parking == "Teremgarázs"
        onallo_garazs = parking == "Önálló garázs"
        utcan_parkolas = parking == "Utcán parkolás"

        return process_inputs(
            address, lat, lon, sqm, rooms,
            tegla_epitesu, panel, csuszozsalus, uj_epitesu,
            ujszeru, felujitott, jo_allapotu, kozepes_allapotu,
            felujitando, epitve_1950_elott, epitve_1950_1980,
            epitve_2001_2010, epitve_2011_utan, udvari_beallo,
            teremgarazs, onallo_garazs, utcan_parkolas
        )

    predict_btn.click(
        fn=validate_and_process,
        inputs=[
            address, lat, lon, sqm, rooms,
            building_type, condition, built_year, parking
        ],
        outputs=[map_output, prediction]
    )

# Az alkalmazás indítása
if __name__ == "__main__":
    app.launch(share=True)