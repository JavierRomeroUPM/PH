import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. CARGA DE MODELO Y CONFIGURACIÓN
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Suave - Doctorado", layout="wide")

@st.cache_resource
def load_assets():
    # Cargamos el modelo que ya tienes (el que da 119.03 en vez de 118.6)
    with open("modelo_hibrido_bit_perfect.pkl", "rb") as f:
        return pickle.load(f)

assets = load_assets()
gbm_oraculo = assets['gbm_base']
nodos_malla = assets['nodos']

# Historial en memoria
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# ==============================================================================
# 2. LÓGICA DE SUAVIZADO DINÁMICO (Adiós al escalón)
# ==============================================================================
@st.cache_data
def generar_grid_suave(pp, dil, form, rug):
    """
    Crea una rejilla 4D de predicciones del GBM para un escenario fijo.
    Esto permite que la interpolación sea SIEMPRE suave.
    """
    shape = (len(nodos_malla['mo']), len(nodos_malla['B']), 
             len(nodos_malla['UCS']), len(nodos_malla['GSI']))
    grid_data = np.zeros(shape)
    
    # Rellenamos el grid con predicciones del GBM (Modo Oráculo)
    for i, m in enumerate(nodos_malla['mo']):
        for j, b in enumerate(nodos_malla['B']):
            for k, u in enumerate(nodos_malla['UCS']):
                for l, g in enumerate(nodos_malla['GSI']):
                    v = np.array([[m, b, u, g, pp, dil, form, rug]])
                    grid_data[i, j, k, l] = gbm_oraculo.predict(v)[0]
    
    # Retornamos el interpolador lineal para este escenario
    return RegularGridInterpolator(
        (nodos_malla['mo'], nodos_malla['B'], nodos_malla['UCS'], nodos_malla['GSI']),
        grid_data, method='linear', fill_value=None
    )

def predecir_suave(mo, b, ucs, gsi, pp, dil, form, rug):
    # 1. Obtenemos (o creamos) el grid suave para este escenario específico
    interp_dinamico = generar_grid_suave(pp, dil, form, rug)
    
    # 2. Interpolamos (esto elimina el efecto escalón de mo, B, UCS y GSI)
    log_ph = interp_dinamico([mo, b, ucs, gsi])[0]
    
    # 3. Verificamos si es un nodo exacto para el historial
    es_exacto = all(np.any(np.isclose(v, nodos_malla[k])) 
                   for v, k in zip([mo, b, ucs, gsi], ['mo', 'B', 'UCS', 'GSI']))
    
    modo = "🎯 NODO (GBM)" if es_exacto else "🔄 INTERPOLADO"
    return np.expm1(log_ph), modo

# ==============================================================================
# 3. INTERFAZ DE USUARIO
# ==============================================================================
st.title("🚀 Predictor de Ph de zapatas dsobre macizo rocoso")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Variables Numéricass")
        in_ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1)
        in_gsi = st.number_input("GSI", 10.0, 85.0, 85.0, step=1.0)
        in_mo = st.number_input("Parámetro m0", 5.0, 32.0, 20.0, step=0.1)
        in_b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables Categoricas")
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=1)
        v_dil = st.selectbox("Dilatancia", ["No asociada", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=0)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=1)

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

if submit:
    # Mapeo numérico
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    
    ph_res, modo = predecir_suave(in_mo, in_b, in_ucs, in_gsi, pp_val, dil_val, for_val, rug_val)
    
    st.success(f"### Ph Predicho: **{ph_res:.4f} MPa**")
    st.info(f"Modo de cálculo: {modo}")

    # Guardar en historial
    st.session_state["historial"].insert(0, {
        "m0": in_mo, "B": in_b, "UCS": in_ucs, "GSI": in_gsi,
        "Forma": v_for, "Rugos.": v_rug, "Dilat.": v_dil, "Peso": v_pp,
        "Ph (MPa)": round(ph_res, 4), "Modo": modo
    })

# ==============================================================================
# 4. HISTORIAL
# ==============================================================================
if st.session_state["historial"]:
    st.markdown("---")
    st.dataframe(pd.DataFrame(st.session_state["historial"]), use_container_width=True)
