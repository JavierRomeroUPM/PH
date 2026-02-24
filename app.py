import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. CARGA DE ACTIVOS "BIT-PERFECT"
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Doctorado", layout="wide")

@st.cache_resource
def load_assets():
    with open("modelo_hibrido_bit_perfect.pkl", "rb") as f:
        return pickle.load(f)

assets = load_assets()
interp_sistema = assets['interp']
gbm_oraculo = assets['gbm_base']
nodos_malla = assets['nodos'] # Aquí están [5, 12...], [4.5, 11...], etc.
ref_vals = assets['ref_vals']

# ==============================================================================
# 2. FUNCIÓN DE PREDICCIÓN CON LÓGICA DE NODOS
# ==============================================================================
def calcular_ph_logico(mo, b, ucs, gsi, pp, dil, form, rug):
    # A. Comprobar Escenario de Referencia
    es_ref = (pp == ref_vals['Peso Propio'] and dil == ref_vals['Dilatancia'] and 
              form == ref_vals['Forma'] and rug == ref_vals['Rugosidad'])
    
    # B. Comprobar si TODOS son Nodos Exactos (con margen de error float)
    es_exacto = (
        any(np.isclose(mo, x) for x in nodos_malla['mo']) and
        any(np.isclose(b, x) for x in nodos_malla['B']) and
        any(np.isclose(ucs, x) for x in nodos_malla['UCS']) and
        any(np.isclose(gsi, x) for x in nodos_malla['GSI'])
    )

    if es_ref:
        # En el escenario de referencia, siempre tenemos la opción de interpolar
        log_ph = interp_sistema([mo, b, ucs, gsi])[0]
        ph = np.expm1(log_ph)
        
        if es_exacto:
            return ph, "🎯 GBM PURO (Nodo exacto)"
        else:
            return ph, "🔄 INTERPOLADO (Suave)"
    else:
        # Fuera de la referencia, el GBM 8D es la única opción (habrá escalones)
        vec_8d = np.array([[mo, b, ucs, gsi, pp, dil, form, rug]])
        log_ph = gbm_oraculo.predict(vec_8d)[0]
        return np.expm1(log_ph), "🤖 GBM PURO (8D - Escalonado)"

# ==============================================================================
# 3. INTERFAZ STREAMLIT
# ==============================================================================
st.title("🚀 Predictor Ph - Metamodelo de Alta Fidelidad")

with st.form("input_form"):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Variables Analíticas")
        in_mo = st.number_input("mo", 5.0, 32.0, 12.0)
        in_b = st.number_input("B (m)", 4.5, 22.0, 11.0)
        in_ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0)
        in_gsi = st.number_input("GSI", 10.0, 85.0, 50.0)
    with c2:
        st.subheader("Escenario")
        v_pp = st.selectbox("Peso Propio", [0, 1], index=1)
        v_dil = st.selectbox("Dilatancia", [0, 1], index=1)
        v_for = st.selectbox("Forma", [0, 1], index=0)
        v_rug = st.selectbox("Rugosidad", [0, 1], index=1)
    
    submit = st.form_submit_button("CALCULAR")

if submit:
    res, modo = calcular_ph_logico(in_mo, in_b, in_ucs, in_gsi, v_pp, v_dil, v_for, v_rug)
    
    st.metric("Capacidad Portante Ph", f"{res:.4f} MPa")
    st.write(f"**Modo de cálculo:** {modo}")
