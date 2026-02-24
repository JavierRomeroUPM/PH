import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DE ACTIVOS (BIT-PERFECT)
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Bit-Perfect", layout="wide")

if "historial" not in st.session_state:
    st.session_state["historial"] = []

@st.cache_resource
def load_assets():
    # USAMOS TU ARCHIVO REAL
    nombre_modelo = "modelo_hibrido_bit_perfect.pkl"
    
    if not os.path.exists(nombre_modelo):
        st.error(f"❌ No se encuentra el archivo: {nombre_modelo}")
        st.stop()
        
    with open(nombre_modelo, "rb") as f:
        return pickle.load(f)

# Cargamos los datos del pkl
assets = load_assets()
interp_sistema = assets['interp']   # El interpolador lineal
gbm_oraculo = assets['gbm_base']    # El GBM puro (8D)
nodos_malla = assets['nodos']       # Valores discretos [5, 12, 50, etc]
ref_vals = assets['ref_vals']       # Peso propio, Rugosidad, etc.

# ==============================================================================
# 2. LÓGICA DE PREDICCIÓN (GBM EN NODOS / LINEAL EN RESTO)
# ==============================================================================
def calcular_ph(mo, b, ucs, gsi, pp, dil, form, rug):
    # Comprobamos si las variables secundarias coinciden con la referencia del Grid
    es_escenario_ref = (pp == ref_vals['Peso Propio'] and 
                        dil == ref_vals['Dilatancia'] and 
                        form == ref_vals['Forma'] and 
                        rug == ref_vals['Rugosidad'])
    
    if es_escenario_ref:
        # MODO SUAVE: Usamos el interpolador lineal 4D
        # Este devuelve el valor exacto del GBM en los nodos y línea en el resto
        log_ph = interp_sistema([mo, b, ucs, gsi])[0]
        return np.expm1(log_ph), "INTERPOLADO / NODO"
    else:
        # MODO ORÁCULO: Fuera del escenario base, usamos el GBM 8D directamente
        vec_8d = np.array([[mo, b, ucs, gsi, pp, dil, form, rug]])
        log_ph = gbm_oraculo.predict(vec_8d)[0]
        return np.expm1(log_ph), "GBM PURO (8D)"

# ==============================================================================
# 3. INTERFAZ DE USUARIO
# ==============================================================================
st.title("🚀 Predictor Ph - Sistema Híbrido Bit-Perfect")
st.markdown("Modelo de alta fidelidad: **GBM Puro** en nodos y **Interpolación Lineal** en transiciones.")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0)
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0)
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 12.0)
        
    with col2:
        st.subheader("⚙️ Variables de Escenario")
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0)
        v_pp = st.selectbox("Peso Propio", ["Sin Peso (0)", "Con Peso (1)"], index=1)
        v_dil = st.selectbox("Dilatancia", ["Nula (0)", "Asociada (1)"], index=1)
        v_for = st.selectbox("Forma", ["Plana (0)", "Axisimétrica (1)"], index=0)
        v_rug = st.selectbox("Rugosidad", ["Liso (0)", "Rugoso (1)"], index=1)

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

if submit:
    # Mapeo a números (0 y 1)
    pp_val = 1 if "1" in v_pp else 0
    dil_val = 1 if "1" in v_dil else 0
    for_val = 1 if "1" in v_for else 0
    rug_val = 1 if "1" in v_rug else 0
    
    ph_resultado, modo = calcular_ph(mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val)
    
    st.markdown("---")
    st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
    st.info(f"**Modo de cálculo:** {modo}")

    # Guardar en historial
    st.session_state["historial"].insert(0, {
        "Hora": datetime.now().strftime("%H:%M:%S"),
        "UCS": ucs, "GSI": gsi, "mo": mo, "B": b, "Ph (MPa)": ph_resultado, "Modo": modo
    })

# ==============================================================================
# 4. HISTORIAL
# ==============================================================================
if st.session_state["historial"]:
    st.subheader("📜 Historial de Consultas")
    st.dataframe(pd.DataFrame(st.session_state["historial"]), use_container_width=True)
