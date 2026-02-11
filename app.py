import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. DEFINICIÓN DE LA CLASE (IDÉNTICA AL ENTRENAMIENTO)
# ==============================================================================
class InterpoladorGrid4D:
    def __init__(self, modelo_xgb, X_data, y_data, valores_discretos):
        self.xgb = modelo_xgb
        self.valores_disc = valores_discretos
        self.grids = {}

    def predecir(self, X_nuevo):
        X_nuevo = np.atleast_2d(X_nuevo)
        predicciones = []
        # mo (0), B (1), UCS (2), GSI (3), Peso (4), Dilat (5), Forma (6), Rugos (7)
        IDX_CONTINUAS = [0, 1, 2, 3]
        IDX_CATEGORICAS = [4, 5, 6, 7]
        
        for x in X_nuevo:
            cat_combo = tuple(int(x[idx]) for idx in IDX_CATEGORICAS)
            cont_vals = x[IDX_CONTINUAS]
            
            interpolador = self.grids.get(cat_combo, None)
            
            if interpolador is None:
                # Si no encuentra el grid, usa XGBoost (escalones)
                log_pred = self.xgb.predict(x.reshape(1, -1))[0]
                pred = np.expm1(log_pred)
                st.warning(f"⚠️ Grid no encontrado para combo {cat_combo}. Usando XGBoost (con escalones).")
            else:
                try:
                    # Interpolación suave (le pasamos el punto como (1,4))
                    # Usamos .reshape(1,-1) para asegurar compatibilidad con scipy
                    val_interp = interpolador(cont_vals.reshape(1, -1))
                    pred = float(val_interp[0])
                    st.info(f"✅ Usando Grid 4D para combo {cat_combo}. Interpolación suave activa.")
                except Exception as e:
                    # Si falla la interpolación, vuelve a XGBoost
                    log_pred = self.xgb.predict(x.reshape(1, -1))[0]
                    pred = np.expm1(log_pred)
                    st.error(f"❌ Error en interpolación: {e}. Usando XGBoost de respaldo.")
            
            predicciones.append(pred)
        return predicciones[0] if len(predicciones) == 1 else np.array(predicciones)

# ==============================================================================
# 2. CARGA DE MODELO
# ==============================================================================
st.set_page_config(page_title="Simulador Ph - Grid 4D", layout="wide")

@st.cache_resource
def load_assets():
    try:
        with open("predictor_grid_4d.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar .pkl: {e}")
        st.stop()

sistema = load_assets()
predictor = sistema['predictor']

# ==============================================================================
# 3. INTERFAZ
# ==============================================================================
st.title("🎯 Simulador Geotécnico - Interpolación Grid 4D")

with st.form("main_form"):
    c1, c2 = st.columns(2)
    with c1:
        mo_in = st.number_input("Parámetro mo", 5.0, 32.0, 25.0, step=0.1)
        b_in = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1)
        ucs_in = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1)
        gsi_in = st.number_input("GSI", 10.0, 85.0, 50.0, step=1.0)
    with c2:
        # ¡OJO! Revisa que estos nombres coincidan con tu lógica 0/1
        pp_sel = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=0) # Index 0 = Sin Peso (0)
        dil_sel = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)     # Index 1 = Asociada (1)
        for_sel = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)    # Index 1 = Axisimétrica (1)
        rug_sel = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=0) # Index 0 = Sin Rugosidad (0)

    calculate = st.form_submit_button("CALCULAR PREDICCIÓN")

if calculate:
    # Mapeo manual para asegurar que enviamos 0 o 1
    v_pp = 1 if pp_sel == "Con Peso" else 0
    v_dil = 1 if dil_sel == "Asociada" else 0
    v_for = 1 if for_sel == "Axisimétrica" else 0
    v_rug = 1 if rug_sel == "Rugoso" else 0
    
    # Vector de entrada: mo, B, UCS, GSI, PP, Dil, Form, Rug
    input_vector = [mo_in, b_in, ucs_in, gsi_in, v_pp, v_dil, v_for, v_rug]
    
    ph = predictor.predecir(input_vector)
    
    st.success(f"## Ph Predicho: {ph:.4f}")
