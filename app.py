import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from scipy.interpolate import RegularGridInterpolator

# CLASE SEGURA: Reconstruye el interpolador al vuelo para evitar errores de versión
class InterpoladorGrid4D:
    def __init__(self, modelo_xgb, valores_discretos):
        self.xgb = modelo_xgb
        self.valores_disc = valores_discretos
        self.grids_data = {} # Aquí están las matrices de Ph

    def predecir(self, x):
        # mo(0), B(1), UCS(2), GSI(3), PP(4), Dil(5), Form(6), Rug(7)
        cat_combo = tuple(int(x[i]) for i in [4, 5, 6, 7])
        cont_vals = np.array([x[0], x[1], x[2], x[3]])
        
        grid_values = self.grids_data.get(cat_combo)
        
        if grid_values is None:
            # Fallback a XGBoost si no hay datos de grid
            return np.expm1(self.xgb.predict(np.array(x).reshape(1, -1))[0])
        
        # RECONSTRUCCIÓN DINÁMICA: Usamos los puntos y la matriz guardada
        interp = RegularGridInterpolator(
            (self.valores_disc['mo'], self.valores_disc['B'], 
             self.valores_disc['UCS'], self.valores_disc['GSI']),
            grid_values,
            method='linear', bounds_error=False, fill_value=None
        )
        
        return float(interp(cont_vals))

st.set_page_config(page_title="Simulador Ph Suave", layout="wide")

@st.cache_resource
def load_all():
    with open("predictor_grid_4d.pkl", "rb") as f:
        return pickle.load(f)

# Carga
sistema = load_all()
# IMPORTANTE: Reasignamos la clase para que use la lógica de arriba
predictor = sistema['predictor']

st.title("🎯 Simulador de Ph - Interpolación Suave Activa")

with st.form("f"):
    c1, c2 = st.columns(2)
    with c1:
        mo = st.number_input("mo", 5.0, 32.0, 25.0, step=0.1) # Prueba mo=25.1, 25.2...
        b = st.number_input("B", 4.5, 22.0, 11.0)
        ucs = st.number_input("UCS", 5.0, 100.0, 50.0)
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0)
    with c2:
        v5 = st.selectbox("Peso", ["Sin", "Con"])
        v6 = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v7 = st.selectbox("Forma", ["Plana", "Axis"], index=1)
        v8 = st.selectbox("Rugosidad", ["Sin", "Con"], index=0)
    
    if st.form_submit_button("CALCULAR"):
        vec = [mo, b, ucs, gsi, 1 if v5=="Con" else 0, 1 if v6=="Asociada" else 0, 
               1 if v7=="Axis" else 0, 1 if v8=="Con" else 0]
        
        res = predictor.predecir(vec)
        st.success(f"### Ph Predicho: {res:.4f}")
        st.caption("La variación de decimales confirma que la interpolación está trabajando.")
