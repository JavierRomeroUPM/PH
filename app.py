import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. Configuración de la página
st.set_page_config(page_title="Predictor Ph GPR - Doctorado", layout="wide")

# 2. Inicialización del historial
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# 3. Carga del Modelo y Escalador (Cacheado)
@st.cache_resource
def load_assets():
    # Cargamos los archivos generados por el entrenamiento de Kriging
    model = joblib.load("modelo_gpr.pkl")
    scaler = joblib.load("escalador_gpr.pkl")
    return model, scaler

try:
    gpr_model, gpr_scaler = load_assets()
    assets_ready = True
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo (.pkl): {e}")
    assets_ready = False

# 4. Interfaz de Usuario
st.title("🚀 Predictor de Ph - Kriging Metamodel (GPR)")
st.markdown("""
Esta versión utiliza **Gaussian Process Regression (Kriging)**. 
A diferencia de los modelos anteriores, este permite una **interpolación suave y continua**, 
siendo sensible a pequeñas variaciones en variables como el UCS.
""")

# Formulario de entrada
with st.form("my_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        ucs_val = st.number_input("UCS - Resistencia Compresión Simple (MPa)", 
                                   min_value=0.0, max_value=200.0, value=35.0, step=0.1, format="%.1f")
        gsi_val = st.number_input("GSI - Geological Strength Index", 
                                   min_value=0, max_value=100, value=40, step=1)
        mo_val = st.number_input("Parámetro mo", 
                                  min_value=0.0, max_value=50.0, value=7.0, step=0.1, format="%.1f")
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        b_val = st.number_input("Ancho de cimentación - B (m)", 
                                 min_value=0.0, max_value=50.0, value=5.0, step=0.1, format="%.2f")
        v5_sel = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=1)
        v6_sel = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=0)
        v7_sel = st.selectbox("Forma del modelo", ["Plana", "Axisimétrica"], index=0)
        v8_sel = st.selectbox("Rugosidad de la base", ["Sin Rugosidad", "Rugoso"], index=0)
        
    st.markdown("---")
    submit = st.form_submit_button("CALCULAR PH", use_container_width=True)

# 5. Lógica de Predicción
if submit and assets_ready:
    try:
        # Conversión de categorías a 0/1
        v5 = 1 if v5_sel == "Con Peso" else 0
        v6 = 1 if v6_sel == "Asociada" else 0
        v7 = 1 if v7_sel == "Axisimétrica" else 0
        v8 = 1 if v8_sel == "Rugoso" else 0
        
        # El orden debe ser EXACTAMENTE el del entrenamiento:
        # ['mo', 'B (m)', 'UCS (MPa)', 'GSI', 'Peso Propio', 'Dilatancia', 'Forma', 'Rugosidad']
        input_data = pd.DataFrame([[
            mo_val, b_val, ucs_val, gsi_val, v5, v6, v7, v8
        ]], columns=['mo', 'B (m)', 'UCS (MPa)', 'GSI', 'Peso Propio', 'Dilatancia', 'Forma', 'Rugosidad'])
        
        # 1. Escalado
        X_scaled = gpr_scaler.transform(input_data)
        
        # 2. Predicción con incertidumbre
        pred_log, sigma_log = gpr_model.predict(X_scaled, return_std=True)
        
        # 3. Transformación inversa
        ph_pred = np.expm1(pred_log)[0]
        incertidumbre = np.expm1(sigma_log)[0]
        
        # Mostrar resultado principal
        st.success(f"### Ph Predicho: {ph_pred:.4f}")
        st.info(f"**Incertidumbre del modelo (±):** {incertidumbre:.6f}")
        
        # Guardar en historial
        st.session_state["historial"].insert(0, {
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "UCS": ucs_val, 
            "GSI": gsi_val, 
            "mo": mo_val, 
            "B": b_val,
            "Peso": v5_sel,
            "Ph": round(float(ph_pred), 4),
            "Confianza (±)": round(float(incertidumbre), 6)
        })
        
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")

# 6. Historial y Descarga
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Análisis")
    df_h = pd.DataFrame(st.session_state["historial"])
    st.dataframe(df_h, use_container_width=True)
    
    c1, c2 = st.columns([2, 8])
    csv = df_h.to_csv(index=False).encode('utf-8')
    c1.download_button("📥 Descargar CSV", data=csv, file_name="historial_ph_gpr.csv", mime="text/csv")
    if c2.button("🗑️ Borrar Historial"):
        st.session_state["historial"] = []
        st.rerun()
