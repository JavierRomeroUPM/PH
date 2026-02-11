import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. DEFINICIÓN DE LA CLASE (OBLIGATORIO PARA QUE PICKLE LA ENCUENTRE)
# ==============================================================================
# Esta clase debe ser idéntica a la que usaste para entrenar
class InterpoladorGrid4D:
    def __init__(self, modelo_xgb, X_data, y_data, valores_discretos):
        self.xgb = modelo_xgb
        self.valores_disc = valores_discretos
        self.grids = {}

    def predecir(self, X_nuevo):
        if isinstance(X_nuevo, list):
            X_nuevo = np.array(X_nuevo)
        
        es_escalar = False
        if X_nuevo.ndim == 1:
            X_nuevo = X_nuevo.reshape(1, -1)
            es_escalar = True
        
        # Aquí va la lógica simplificada para la predicción en la App
        # El objeto cargado ya tiene los 'grids' entrenados
        predicciones = []
        # Índices fijos según tu entrenamiento
        IDX_CATEGORICAS = [4, 5, 6, 7]
        IDX_CONTINUAS = [0, 1, 2, 3]

        for i in range(len(X_nuevo)):
            x = X_nuevo[i]
            cat_combo = tuple(int(x[idx]) for idx in IDX_CATEGORICAS)
            cont_vals = x[IDX_CONTINUAS]
            
            interpolador = self.grids.get(cat_combo, None)
            
            if interpolador is None:
                # Si no hay grid, usa el modelo base (necesita importado el modelo xgb)
                pred = 0.0 # Valor por defecto o fallback
            else:
                try:
                    pred = float(interpolador(cont_vals))
                except:
                    pred = 0.0
            predicciones.append(pred)
        
        return predicciones[0] if es_escalar else np.array(predicciones)

# ==============================================================================
# 2. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Predictor Ph Grid 4D", layout="wide")

# Inicialización del historial
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# 3. CARGA DE ACTIVOS
@st.cache_resource
def load_assets():
    try:
        with open("predictor_grid_4d.pkl", "rb") as f:
            # Ahora pickle sí encontrará la clase 'InterpoladorGrid4D' arriba definida
            sistema = pickle.load(f)
        return sistema
    except FileNotFoundError:
        st.error("❌ No se encuentra 'predictor_grid_4d.pkl'")
        st.stop()

sistema = load_assets()
predictor = sistema['predictor']
metricas = sistema['metricas']
valores_discretos = sistema['valores_discretos']

# ==============================================================================
# 4. INTERFAZ STREAMLIT (Igual que la anterior)
# ==============================================================================
st.title("🚀 Predictor de Ph - Interpolación Grid 4D")

with st.form("my_form"):
    col1, col2 = st.columns(2)
    with col1:
        ucs_val = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0)
        gsi_val = st.number_input("GSI", 10, 85, 50)
        mo_val = st.number_input("mo", 5.0, 32.0, 20.0)
    with col2:
        b_val = st.number_input("B (m)", 4.5, 22.0, 11.0)
        v5 = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v6 = st.selectbox("Dilatancia", ["Nulo", "Asociada"])
        v7 = st.selectbox("Forma", ["Plana", "Axisimétrica"])
        v8 = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"])
    
    submit = st.form_submit_button("CALCULAR")

if submit:
    # Mismo mapeo de 0 y 1 que en el entrenamiento
    c5 = 1 if v5 == "Con Peso" else 0
    c6 = 1 if v6 == "Asociada" else 0
    c7 = 1 if v7 == "Axisimétrica" else 0
    c8 = 1 if v8 == "Rugoso" else 0
    
    vector = [mo_val, b_val, ucs_val, gsi_val, c5, c6, c7, c8]
    ph_pred = predictor.predecir(vector)
    
    st.success(f"### Ph Predicho: {ph_pred:.4f}")
    
    # Guardar en historial
    st.session_state["historial"].insert(0, {"Hora": datetime.now().strftime("%H:%M:%S"), "Ph": ph_pred})

# Mostrar historial
if st.session_state["historial"]:
    st.table(pd.DataFrame(st.session_state["historial"]))
