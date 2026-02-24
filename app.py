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

if "historial" not in st.session_state:
    st.session_state["historial"] = []

@st.cache_resource
def load_assets():
    nombre_pkl = "modelo_hibrido_bit_perfect.pkl"
    if not os.path.exists(nombre_pkl):
        st.error(f"❌ No se encuentra el archivo: {nombre_pkl}")
        st.stop()
    with open(nombre_pkl, "rb") as f:
        return pickle.load(f)

assets = load_assets()
interp_sistema = assets['interp']
gbm_oraculo = assets['gbm_base']
nodos_malla = assets['nodos']
ref_vals = assets['ref_vals']

# ==============================================================================
# 2. FUNCIÓN DE PREDICCIÓN CON LÓGICA DE NODOS
# ==============================================================================
def calcular_ph_logico(mo, b, ucs, gsi, pp, dil, form, rug):
    # Escenario de referencia (el que se usó para construir el Grid 4D)
    es_ref = (pp == ref_vals['Peso Propio'] and dil == ref_vals['Dilatancia'] and 
              form == ref_vals['Forma'] and rug == ref_vals['Rugosidad'])
    
    # Comprobar si los valores continuos coinciden con los nodos de la malla
    es_exacto = (
        any(np.isclose(mo, x) for x in nodos_malla['mo']) and
        any(np.isclose(b, x) for x in nodos_malla['B']) and
        any(np.isclose(ucs, x) for x in nodos_malla['UCS']) and
        any(np.isclose(gsi, x) for x in nodos_malla['GSI'])
    )

    if es_ref:
        # Modo Suave (Interpolador)
        log_ph = interp_sistema([mo, b, ucs, gsi])[0]
        ph = np.expm1(log_ph)
        modo = "🎯 PURO (Nodo)" if es_exacto else "🔄 INTERPOLADO"
        return ph, modo
    else:
        # Modo Inferencia (GBM directo - Escalonado)
        vec_8d = np.array([[mo, b, ucs, gsi, pp, dil, form, rug]])
        log_ph = gbm_oraculo.predict(vec_8d)[0]
        return np.expm1(log_ph), "🤖 GBM PURO (8D)"

# ==============================================================================
# 3. INTERFAZ DE USUARIO (CONFIGURACIÓN SOLICITADA)
# ==============================================================================
st.title("🚀 Predictor Ph - Metamodelo de Alta Fidelidad")
st.markdown("Sistema de obtención de capacidad portante mediante hibridación GBM-Linear.")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1)
        gsi = st.number_input("GSI", 10.0, 85.0, 85.0, step=1.0)
        mo = st.number_input("Parámetro m0", 5.0, 32.0, 20.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=0)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=1)
        v_dil = st.selectbox("Dilatancia", ["No asociada", "Asociada"], index=1)
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=1)

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

if submit:
    # Mapeo a formato numérico para el modelo
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    
    ph_resultado, modo = calcular_ph_logico(mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val)
    
    # Mostrar Resultado
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
    with res_col2:
        st.info(f"**Modo:** {modo}")

    # Guardar en historial con las 8 variables + resultado + modo
    nuevo_registro = {
        "m0": mo, "B": b, "UCS": ucs, "GSI": gsi,
        "Forma": v_for, "Rugos.": v_rug, "Dilat.": v_dil, "Peso": v_pp,
        "Ph (MPa)": round(ph_resultado, 4),
        "Cálculo": modo
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# ==============================================================================
# 4. HISTORIAL TÉCNICO
# ==============================================================================
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Resultados")
    df_hist = pd.DataFrame(st.session_state["historial"])
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
    
    if st.button("🗑️ Borrar Historial"):
        st.session_state["historial"] = []
        st.rerun()

st.caption("PhD Framework | XGBoost-Grid4D Hybrid | © 2024")
