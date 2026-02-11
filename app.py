import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
from collections import defaultdict

# ==============================================================================
# 1. DEFINICIÓN DE LA CLASE (DEBE SER IDÉNTICA A TU SCRIPT GENERADOR)
# ==============================================================================
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
        
        predicciones = []
        IDX_CONTINUAS = [0, 1, 2, 3]  # mo, B, UCS, GSI
        IDX_CATEGORICAS = [4, 5, 6, 7] # PP, Dil, Form, Rug

        for i in range(len(X_nuevo)):
            x = X_nuevo[i]
            cat_combo = tuple(int(x[idx]) for idx in IDX_CATEGORICAS)
            cont_vals = x[IDX_CONTINUAS]
            
            interpolador = self.grids.get(cat_combo, None)
            
            if interpolador is None:
                # Fallback a XGBoost puro (escala log -> real)
                pred = np.expm1(self.xgb.predict(x.reshape(1, -1)))[0]
            else:
                try:
                    # Interpolación n-lineal suave
                    pred = float(interpolador(cont_vals))
                    # Validación de seguridad
                    if np.isnan(pred) or np.isinf(pred) or pred < 0:
                        pred = np.expm1(self.xgb.predict(x.reshape(1, -1)))[0]
                except:
                    pred = np.expm1(self.xgb.predict(x.reshape(1, -1)))[0]
            
            predicciones.append(pred)
        
        return predicciones[0] if es_escalar else np.array(predicciones)

# ==============================================================================
# 2. CONFIGURACIÓN Y CARGA DE ACTIVOS
# ==============================================================================
st.set_page_config(page_title="Predictor Ph - Grid 4D", layout="wide")

@st.cache_resource
def load_assets():
    try:
        with open("predictor_grid_4d.pkl", "rb") as f:
            # Pickle reconstruye la clase usando la definición de arriba
            sistema = pickle.load(f)
        return sistema
    except Exception as e:
        st.error(f"❌ Error crítico: No se pudo cargar el archivo pkl. {e}")
        st.stop()

sistema = load_assets()
predictor = sistema['predictor']
metricas = sistema['metricas']
valores_discretos = sistema['valores_discretos']

# ==============================================================================
# 3. INTERFAZ DE USUARIO
# ==============================================================================
st.title("🚀 Predictor de Ph - XGBoost + Grid 4D")
st.markdown(f"**MAPE del sistema:** {metricas['grid']['mape']:.2f}% | **Interpolación:** Suave N-Lineal")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 20.0, help="Rango: 5 - 32")
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, help="Rango: 4.5 - 22")
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, help="Rango: 5 - 100")
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0, help="Rango: 10 - 85")
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=1)
    
    submit = st.form_submit_button("🎯 CALCULAR PH", use_container_width=True)

if submit:
    # Mapeo idéntico al entrenamiento
    c_pp = 1 if v_pp == "Con Peso" else 0
    c_dil = 1 if v_dil == "Asociada" else 0
    c_for = 1 if v_for == "Axisimétrica" else 0
    c_rug = 1 if v_rug == "Rugoso" else 0
    
    # Vector: mo, B, UCS, GSI, PP, Dil, Form, Rug
    vector = [mo, b, ucs, gsi, c_pp, c_dil, c_for, c_rug]
    
    ph_pred = predictor.predecir(vector)
    
    st.markdown("---")
    st.success(f"### Ph Predicho: **{ph_pred:.4f}**")
    
    # Comprobar si es punto exacto o interpolado
    esta_en_grid = (mo in valores_discretos['mo'] and b in valores_discretos['B'] and 
                    ucs in valores_discretos['UCS'] and gsi in valores_discretos['GSI'])
    
    if esta_en_grid:
        st.caption("📍 El punto coincide exactamente con los valores discretos de entrenamiento.")
    else:
        st.caption("🔄 El resultado es una interpolación lineal entre los nodos del hipercubo 4D.")

# (Opcional: puedes añadir aquí el bloque del historial que tenías antes)
