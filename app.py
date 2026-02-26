import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. CARGA DE ACTIVOS
# ==============================================================================
st.set_page_config(page_title="Simulador Ph - Lógica Híbrida", layout="wide")

@st.cache_resource
def load_assets():
    # Asegúrate de que este archivo contenga el modelo CatBoost y la definición de nodos
    with open("sistema_catboost_completo.pkl", "rb") as f:
        return pickle.load(f)

try:
    assets = load_assets()
    model_cb = assets['modelo']
    nodos_malla = assets['nodos']
except Exception as e:
    st.error(f"❌ Error al cargar activos: {e}")
    st.stop()

if "historial" not in st.session_state:
    st.session_state["historial"] = []

# ==============================================================================
# 2. MOTOR DE CÁLCULO (CATBOOST + INTERPOLACIÓN FORZADA)
# ==============================================================================

@st.cache_data
def generar_grid_final(pp, dil, form, rug):
    """
    Crea la malla de predicciones. 
    Aplica la interpolación lineal forzada entre GSI 10 y 50 para escenarios no analíticos.
    """
    # Escenario de referencia (Peso=1, Dilatancia=1, Forma=0, Rugosidad=1)
    es_referencia = (pp == 1 and dil == 1 and form == 0 and rug == 1)
    
    shape = (len(nodos_malla['mo']), len(nodos_malla['B']), 
             len(nodos_malla['UCS']), len(nodos_malla['GSI']))
    grid_data = np.zeros(shape)
    
    columnas = ['mo', 'B (m)', 'UCS (MPa)', 'GSI', 'Peso Propio', 'Dilatancia', 'Forma', 'Rugosidad']
    
    for i, m in enumerate(nodos_malla['mo']):
        for j, b in enumerate(nodos_malla['B']):
            for k, u in enumerate(nodos_malla['UCS']):
                # Si no es escenario de referencia, pre-calculamos extremos para GSI 10-50
                if not es_referencia:
                    v10 = pd.DataFrame([[m, b, u, 10, pp, dil, form, rug]], columns=columnas)
                    v50 = pd.DataFrame([[m, b, u, 50, pp, dil, form, rug]], columns=columnas)
                    ph_10 = model_cb.predict(v10)[0]
                    ph_50 = model_cb.predict(v50)[0]

                for l, g in enumerate(nodos_malla['GSI']):
                    # REGLA ESPECIAL: Forzar línea recta entre GSI 10 y 50 en escenarios sin datos
                    if not es_referencia and 10 < g < 50:
                        # Interpolación lineal manual: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                        grid_data[i, j, k, l] = ph_10 + (g - 10) * (ph_50 - ph_10) / (50 - 10)
                    else:
                        # Predicción normal de CatBoost para el resto
                        v = pd.DataFrame([[m, b, u, g, pp, dil, form, rug]], columns=columnas)
                        grid_data[i, j, k, l] = model_cb.predict(v)[0]
    
    return RegularGridInterpolator(
        (nodos_malla['mo'], nodos_malla['B'], nodos_malla['UCS'], nodos_malla['GSI']),
        grid_data, method='linear', fill_value=None
    )

def calcular_prediccion(mo, b, ucs, gsi, pp, dil, form, rug):
    # 1. Comprobar si es un dato discretizado (Nodo exacto del entrenamiento)
    es_exacto = (mo in nodos_malla['mo'] and b in nodos_malla['B'] and 
                 ucs in nodos_malla['UCS'] and gsi in nodos_malla['GSI'])
    
    if es_exacto:
        columnas = ['mo', 'B (m)', 'UCS (MPa)', 'GSI', 'Peso Propio', 'Dilatancia', 'Forma', 'Rugosidad']
        v = pd.DataFrame([[mo, b, ucs, gsi, pp, dil, form, rug]], columns=columnas)
        log_ph = model_cb.predict(v)[0]
        modo = "🎯 NODO EXACTO (CatBoost)"
    else:
        # 2. Si no es exacto, usamos la malla interpolada (que ya tiene la corrección GSI 10-50)
        interp = generar_grid_final(pp, dil, form, rug)
        log_ph = interp([mo, b, ucs, gsi])[0]
        
        # Identificar si estamos en la zona de interpolación forzada
        es_referencia = (pp == 1 and dil == 1 and form == 0 and rug == 1)
        if 10 <= gsi <= 50 and not es_referencia:
            modo = "📏 INTERPOLACIÓN LINEAL (GSI 10-50)"
        else:
            modo = "🔄 INTERPOLACIÓN MULTILINEAL"
            
    return np.expm1(log_ph), modo

# ==============================================================================
# 3. INTERFAZ DE USUARIO
# ==============================================================================
st.title("Predictor Geotécnico Ph")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        in_ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0)
        in_gsi = st.number_input("GSI", 10.0, 85.0, 50.0)
        in_mo = st.number_input("m0", 5.0, 32.0, 20.0)
        in_b = st.number_input("B (m)", 4.5, 22.0, 11.0)
    with col2:
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=1)
        v_dil = st.selectbox("Dilatancia", ["No asociada", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=0)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=1)
    
    submit = st.form_submit_button("CALCULAR", use_container_width=True)

if submit:
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0

    ph_res, modo = calcular_prediccion(in_mo, in_b, in_ucs, in_gsi, pp_val, dil_val, for_val, rug_val)
    
    st.metric("Presión de Hundimiento (Ph)", f"{ph_res:.4f} MPa")
    st.info(f"Método: {modo}")

    # Guardar en historial
    st.session_state["historial"].insert(0, {
        "m0": in_mo, "B": in_b, "UCS": in_ucs, "GSI": in_gsi,
        "Forma": v_for, "Rugos.": v_rug, "Dilat.": v_dil, "Peso": v_pp,
        "Ph (MPa)": round(ph_res, 4), "Modo": modo
    })

# ==============================================================================
# 4. HISTORIAL (TABLA DINÁMICA)
# ==============================================================================
if st.session_state["historial"]:
    st.write("### Histórico de Cálculos")
    st.dataframe(pd.DataFrame(st.session_state["historial"]), use_container_width=True)
