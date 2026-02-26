import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. CARGA DE ACTIVOS (CATBOOST OPTIMIZADO)
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Pro - CatBoost Híbrido", layout="wide")

@st.cache_resource
def load_assets():
    # Cargamos el sistema CatBoost generado por el pipeline de Optuna
    # Asegúrate de que el archivo se llame exactamente así
    with open("sistema_catboost_completo.pkl", "rb") as f:
        return pickle.load(f)

try:
    assets = load_assets()
    model_cb = assets['modelo']
    nodos_malla = assets['nodos']
except FileNotFoundError:
    st.error("❌ No se encontró 'sistema_catboost_completo.pkl'. Por favor, cárgalo en el directorio.")
    st.stop()

if "historial" not in st.session_state:
    st.session_state["historial"] = []

# ==============================================================================
# 2. LÓGICA DE INTERPOLACIÓN PROTEGIDA (GSI 30)
# ==============================================================================
@st.cache_data
def generar_grid_protegido(pp, dil, form, rug):
    """
    Crea una rejilla 4D usando CatBoost. 
    Aplica una corrección lineal en GSI 30 para escenarios sin datos (Sparsity).
    """
    # Escenario de referencia (el único que tiene datos reales para GSI 30)
    es_referencia = (pp == 1 and rug == 1 and form == 0 and dil == 1)
    
    shape = (len(nodos_malla['mo']), len(nodos_malla['B']), 
             len(nodos_malla['UCS']), len(nodos_malla['GSI']))
    grid_data = np.zeros(shape)
    
    columnas = ['mo', 'B (m)', 'UCS (MPa)', 'GSI', 'Peso Propio', 'Dilatancia', 'Forma', 'Rugosidad']
    
    for i, m in enumerate(nodos_malla['mo']):
        for j, b in enumerate(nodos_malla['B']):
            for k, u in enumerate(nodos_malla['UCS']):
                for l, g in enumerate(nodos_malla['GSI']):
                    
                    # --- LÓGICA DE PROTECCIÓN GSI 30 ---
                    if g == 30 and not es_referencia:
                        # Para evitar el sesgo de CatBoost, calculamos GSI 50 y GSI 10
                        # Nodo GSI 50 (Fiable)
                        v50 = pd.DataFrame([[m, b, u, 50, pp, dil, form, rug]], columns=columnas)
                        ph_g50 = model_cb.predict(v50)[0]
                        
                        # Nodo GSI 10 (Mínimo teórico: asumimos un 15% de la capacidad de GSI 50)
                        ph_g10 = ph_g50 * 0.15 
                        
                        # Interpolación lineal manual para el nodo 30: 
                        # Está a mitad de camino entre 10 y 50
                        grid_data[i, j, k, l] = (ph_g10 + ph_g50) / 2
                    
                    else:
                        # Predicción normal para el resto de la malla
                        v = pd.DataFrame([[m, b, u, g, pp, dil, form, rug]], columns=columnas)
                        grid_data[i, j, k, l] = model_cb.predict(v)[0]
    
    return RegularGridInterpolator(
        (nodos_malla['mo'], nodos_malla['B'], nodos_malla['UCS'], nodos_malla['GSI']),
        grid_data, method='linear', fill_value=None
    )

def predecir_hibrido(mo, b, ucs, gsi, pp, dil, form, rug):
    interp = generar_grid_protegido(pp, dil, form, rug)
    log_ph = interp([mo, b, ucs, gsi])[0]
    
    # Verificación de zona de incertidumbre para el feedback al usuario
    es_referencia = (pp == 1 and rug == 1 and form == 0 and dil == 1)
    if gsi <= 35 and not es_referencia:
        modo = "⚠️ INTERP. SEGURIDAD (GSI 10-50)"
    else:
        modo = "✅ CATBOOST + GRID"
        
    return np.expm1(log_ph), modo

# ==============================================================================
# 3. INTERFAZ DE USUARIO
# ==============================================================================
st.title("Predictor Ph - Macizo Rocoso (Motor CatBoost)")
st.markdown("""
Esta aplicación utiliza un modelo **CatBoost optimizado** integrado con una malla de interpolación 4D. 
Incluye una corrección de seguridad para rocas de baja calidad (GSI ≤ 30).
""")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Parámetros Geomecánicos")
        in_ucs = st.number_input("UCS - Resistencia Compresión Simple (MPa)", 5.0, 100.0, 50.0, step=0.1)
        in_gsi = st.number_input("GSI - Geological Strength Index", 10.0, 85.0, 50.0, step=1.0)
        in_mo = st.number_input("Parámetro m0 (Hoek-Brown)", 5.0, 32.0, 20.0, step=0.1)
        in_b = st.number_input("Ancho de la zapata B (m)", 4.5, 22.0, 11.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Escenario de Cálculo")
        v_pp = st.selectbox("Peso Propio de la roca", ["Sin Peso", "Con Peso"], index=1)
        v_dil = st.selectbox("Dilatancia", ["No asociada", "Asociada"], index=1)
        v_for = st.selectbox("Forma de zapata", ["Plana", "Axisimétrica"], index=0)
        v_rug = st.selectbox("Rugosidad de contacto", ["Sin Rugosidad", "Rugoso"], index=1)

    submit = st.form_submit_button("🚀 CALCULAR PRESIÓN DE HUNDIMIENTO", use_container_width=True)

if submit:
    # Mapeo numérico para el modelo
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    
    ph_res, modo = predecir_hibrido(in_mo, in_b, in_ucs, in_gsi, pp_val, dil_val, for_val, rug_val)
    
    # Mostrar resultados destacados
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Presión de Hundimiento (Ph)", f"{ph_res:.4f} MPa")
    with c2:
        if "⚠️" in modo:
            st.warning(f"Estado: {modo}")
        else:
            st.success(f"Estado: {modo}")

    # Notificación de seguridad para GSI bajo
    if in_gsi <= 30:
        st.error("📢 **Nota Técnica:** Para GSI=30 en escenarios distintos al de referencia, el sistema aplica una interpolación lineal de seguridad debido a la ausencia de datos numéricos en la base de datos original.")

    # Guardar en historial
    st.session_state["historial"].insert(0, {
        "m0": in_mo, "B": in_b, "UCS": in_ucs, "GSI": in_gsi,
        "Forma": v_for, "Rugos.": v_rug, "Dilat.": v_dil, "Peso": v_pp,
        "Ph (MPa)": round(ph_res, 4), "Modo": modo
    })

# ==============================================================================
# 4. TABLA DE RESULTADOS
# ==============================================================================
if st.session_state["historial"]:
    st.markdown("### 📋 Historial de cálculos")
    st.dataframe(pd.DataFrame(st.session_state["historial"]), use_container_width=True)
