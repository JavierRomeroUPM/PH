import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DE ACTIVOS
# ==============================================================================
st.set_page_config(page_title="Predictor Híbrido - Control de Sesgos", layout="wide")

@st.cache_resource
def load_assets():
    # Cargamos el sistema CatBoost (el motor de precisión)
    with open("sistema_catboost_completo.pkl", "rb") as f:
        return pickle.load(f)

try:
    assets = load_assets()
    model_cb = assets['modelo']
    nodos_malla = assets['nodos']
    # Nombres de columnas que espera el modelo
    X_cols = ['mo', 'B (m)', 'UCS (MPa)', 'GSI', 'Peso Propio', 'Dilatancia', 'Forma', 'Rugosidad']
except Exception as e:
    st.error(f"❌ Error al cargar activos: {e}")
    st.stop()

# ==============================================================================
# 2. GENERADOR DE MALLA CON ANULACIÓN MANUAL DE SESGOS
# ==============================================================================
@st.cache_data
def generar_grid_hibrido(pp, dil, form, rug):
    """
    Crea la malla 4D. 
    Si el escenario es distinto al entrenado para GSI 30, 
    anula al ML y aplica interpolación lineal de seguridad.
    """
    # Escenario controlado (el único donde el GSI 30 del modelo es fiable)
    es_escenario_fiel = (pp == 1 and rug == 1 and form == 0 and dil == 1)
    
    shape = (len(nodos_malla['mo']), len(nodos_malla['B']), 
             len(nodos_malla['UCS']), len(nodos_malla['GSI']))
    grid_data = np.zeros(shape)
    
    for i, m in enumerate(nodos_malla['mo']):
        for j, b in enumerate(nodos_malla['B']):
            for k, u in enumerate(nodos_malla['UCS']):
                for l, g in enumerate(nodos_malla['GSI']):
                    
                    # --- REGLA DE SEGURIDAD PARA GSI 30 ---
                    if g == 30 and not es_escenario_fiel:
                        # 1. Obtenemos el valor fiable en GSI 50 para este escenario
                        v50 = pd.DataFrame([[m, b, u, 50, pp, dil, form, rug]], columns=X_cols)
                        ph_g50 = np.expm1(model_cb.predict(v50)[0])
                        
                        # 2. Definimos Ph en GSI 10 como un valor residual (ej: 10% de GSI 50)
                        ph_g10 = ph_g50 * 0.10
                        
                        # 3. Calculamos el valor en 30 por interpolación lineal (punto medio entre 10 y 50)
                        # Nota: Guardamos el log1p porque la malla se interpola en log
                        grid_data[i, j, k, l] = np.log1p((ph_g10 + ph_g50) / 2)
                        
                    else:
                        # Predicción normal de CatBoost (donde hay datos o hay coherencia)
                        v = pd.DataFrame([[m, b, u, g, pp, dil, form, rug]], columns=X_cols)
                        grid_data[i, j, k, l] = model_cb.predict(v)[0]
    
    return RegularGridInterpolator(
        (nodos_malla['mo'], nodos_malla['B'], nodos_malla['UCS'], nodos_malla['GSI']),
        grid_data, method='linear'
    )

# ==============================================================================
# 3. INTERFAZ Y CÁLCULO
# ==============================================================================
st.title("Predictor Ph - Híbrido (ML + Reglas de Ingeniería)")
st.info("Este modelo detecta zonas con sesgo de datos (GSI ≤ 30) y aplica correcciones físicas automáticas.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        in_ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0)
        in_gsi = st.number_input("GSI", 10.0, 85.0, 50.0)
        in_mo = st.number_input("m0", 5.0, 32.0, 20.0)
        in_b = st.number_input("B (m)", 4.5, 22.0, 11.0)
    with col2:
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=0)
        v_dil = st.selectbox("Dilatancia", ["No asociada", "Asociada"], index=1)
    
    submit = st.form_submit_button("CALCULAR PREDICCIÓN", use_container_width=True)

if submit:
    # Mapeo
    pp_val = 1 if v_pp == "Con Peso" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    
    # Generar Malla Protegida
    interp = generar_grid_hibrido(pp_val, dil_val, for_val, rug_val)
    
    # Predecir
    res_log = interp([in_mo, in_b, in_ucs, in_gsi])[0]
    ph_final = np.expm1(res_log)
    
    # Mostrar Resultado
    st.metric("Presión de Hundimiento Ph", f"{ph_final:.4f} MPa")
    
    # Feedback de seguridad
    es_referencia = (pp_val == 1 and rug_val == 1 and for_val == 0 and dil_val == 1)
    if in_gsi <= 35 and not es_referencia:
        st.warning("⚠️ **Corrección Activa:** Se ha aplicado una interpolación lineal de seguridad debido a falta de datos históricos para GSI ≤ 30 en este escenario.")
    else:
        st.success("✅ **Predicción basada en ML:** El modelo se encuentra en una zona de alta densidad de datos.")
