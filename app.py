import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. Configuración de la página
st.set_page_config(page_title="Predictor Ph GPR - Kriging Metamodel", layout="wide")

# 2. Inicialización del historial
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# 3. Carga de Activos (Modelo + Escalador)
@st.cache_resource
def load_assets():
    # Cargamos el modelo Kriging y el escalador entrenado
    gpr_model = joblib.load("modelo_gpr.pkl")
    scaler = joblib.load("escalador_gpr.pkl")
    return gpr_model, scaler

try:
    model, scaler = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error al cargar archivos del modelo: {e}")
    assets_loaded = False

# 4. Interfaz de Usuario
st.title("🚀 Predictor de Ph - Kriging Metamodel (GPR)")
st.subheader("Simulador de Alta Fidelidad para Tesis Doctoral")
st.markdown("""
Esta versión utiliza **Gaussian Process Regression**, permitiendo una **interpolación continua** y precisa 
entre los escenarios simulados en FLAC.
""")

# Formulario de entrada
with st.form("my_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        ucs_val = st.number_input("UCS - Resistencia Compresión Simple (MPa)", 
                                   min_value=5.0, max_value=100.0, value=35.0, step=0.5, format="%.1f")
        gsi_val = st.number_input("GSI - Geological Strength Index", 
                                   min_value=10, max_value=85, value=40, step=1)
        mo_val = st.number_input("Parámetro mo", 
                                  min_value=5.0, max_value=32.0, value=7.0, step=0.1, format="%.1f")
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        b_val = st.number_input("Ancho de cimentación - B (m)", 
                                 min_value=4.5, max_value=22.0, value=5.0, step=0.1, format="%.2f")
        v5_sel = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=1)
        v6_sel = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=0)
        v7_sel = st.selectbox("Forma del modelo", ["Plana", "Axisimétrica"], index=0)
        v8_sel = st.selectbox("Rugosidad de la base", ["Sin Rugosidad", "Rugoso"], index=0)
        
    st.markdown("---")
    submit = st.form_submit_button("CALCULAR PREDICCIÓN", use_container_width=True)

# 5. Lógica de Predicción
if submit and assets_loaded:
    try:
        # Conversión de categorías a 0/1
        v5 = 1 if v5_sel == "Con Peso" else 0
        v6 = 1 if v6_sel == "Asociada" else 0
        v7 = 1 if v7_sel == "Axisimétrica" else 0
        v8 = 1 if v8_sel == "Rugoso" else 0
        
        # El orden DEBE ser el mismo que en el entrenamiento:
        # ['mo', 'B (m)', 'UCS (MPa)', 'GSI', 'Peso Propio', 'Dilatancia', 'Forma', 'Rugosidad']
        input_data = pd.DataFrame([[
            mo_val, b_val, ucs_val, gsi_val, v5, v6, v7, v8
        ]], columns=['mo', 'B (m)', 'UCS (MPa)', 'GSI', 'Peso Propio', 'Dilatancia', 'Forma', 'Rugosidad'])
        
        # 1. Escalado (Fundamental para GPR)
        X_scaled = scaler.transform(input_data)
        
        # 2. Predicción con incertidumbre (Sigma)
        pred_log, sigma_log = model.predict(X_scaled, return_std=True)
        
        # 3. Invertir logaritmo
        ph_pred = np.expm1(pred_log)[0]
        # Incertidumbre aproximada en escala real
        uncertainty = np.expm1(sigma_log)[0]
        
        # Mostrar resultado
        st.success(f"### Ph Predicho: {ph_pred:.4f}")
        st.info(f"**Incertidumbre del modelo (±):** {uncertainty:.6f}")
        st.caption("Nota: Una incertidumbre baja indica que los datos están cerca de puntos simulados en FLAC.")
        
        # Guardar en historial
        st.session_state["historial"].insert(0, {
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "UCS": ucs_val, "GSI": gsi_val, "mo": mo_val, "B": b_val,
            "Ph": round(float(ph_pred), 4),
            "Confianza (±)": round(float(uncertainty), 6)
        })
        
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")

# 6. Historial
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Análisis")
    df_h = pd.DataFrame(st.session_state["historial"])
    st.dataframe(df_h, use_container_width=True)