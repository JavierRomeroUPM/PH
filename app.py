import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime

# 1. Configuración de la página
st.set_page_config(page_title="Predictor Ph Grid 4D - XGBoost Interpolado", layout="wide")

# 2. Inicialización del historial
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# 3. Carga del Modelo Grid 4D
@st.cache_resource
def load_assets():
    """Carga el modelo Grid 4D desde el archivo pickle"""
    try:
        with open("predictor_grid_4d.pkl", "rb") as f:
            sistema = pickle.load(f)
        return sistema
    except FileNotFoundError:
        st.error("❌ No se encuentra el archivo 'predictor_grid_4d.pkl' en el repositorio.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

try:
    sistema = load_assets()
    predictor = sistema['predictor']
    metricas = sistema['metricas']
    valores_discretos = sistema['valores_discretos']
    assets_loaded = True
except Exception as e:
    st.error(f"Error al inicializar los activos del modelo: {e}")
    assets_loaded = False

# 4. Interfaz de Usuario
st.title("🚀 Predictor de Ph - XGBoost + Interpolación Grid 4D")
st.subheader("Sistema de Alta Precisión con Interpolación Suave")

if assets_loaded:
    st.markdown(f"""
    Esta versión utiliza **XGBoost + Grid 4D**, permitiendo **interpolación continua** entre valores discretos sin efecto escalón.

    **📊 Rendimiento del modelo:** Error máximo {metricas['grid']['error_max']:.2f}% | MAPE {metricas['grid']['mape']:.2f}%
    """)

# Información sobre valores discretos
with st.expander("ℹ️ Información sobre las variables continuas"):
    st.markdown(f"""
    Las siguientes variables fueron entrenadas con valores discretos, pero el modelo 
    **interpola suavemente** entre ellos:
    
    - **mo**: {list(valores_discretos['mo'])}
    - **B (m)**: {list(valores_discretos['B'])}
    - **UCS (MPa)**: {list(valores_discretos['UCS'])}
    - **GSI**: {list(valores_discretos['GSI'])}
    
    ✅ **Puedes introducir cualquier valor intermedio** (ej: B=7.3m, UCS=35 MPa)
    y el modelo interpolará correctamente sin efecto escalón.
    """)

# Formulario de entrada
with st.form("my_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        ucs_val = st.number_input("UCS - Resistencia Compresión Simple (MPa)", 5.0, 100.0, 50.0, 0.1, format="%.1f")
        gsi_val = st.number_input("GSI - Geological Strength Index", 10, 85, 50, 1)
        mo_val = st.number_input("Parámetro mo", 5.0, 32.0, 20.0, 0.1, format="%.1f")
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        b_val = st.number_input("Ancho de cimentación - B (m)", 4.5, 22.0, 11.0, 0.1, format="%.2f")
        v5_sel = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=0)
        v6_sel = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v7_sel = st.selectbox("Forma del modelo", ["Plana", "Axisimétrica"], index=1)
        v8_sel = st.selectbox("Rugosidad de la base", ["Sin Rugosidad", "Rugoso"], index=1)
        
    st.markdown("---")
    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

# 5. Lógica de Predicción
if assets_loaded and submit:
    try:
        # Conversión de categorías
        v5 = 1 if v5_sel == "Con Peso" else 0
        v6 = 1 if v6_sel == "Asociada" else 0
        v7 = 1 if v7_sel == "Axisimétrica" else 0
        v8 = 1 if v8_sel == "Rugoso" else 0
        
        input_vector = [mo_val, b_val, ucs_val, gsi_val, v5, v6, v7, v8]
        ph_pred = predictor.predecir(input_vector)
        
        if np.isnan(ph_pred) or ph_pred < 0:
            st.error("⚠️ Predicción fuera de rango válido.")
        else:
            st.success(f"### 🎯 Ph Predicho: **{ph_pred:.4f}**")
            
            # Guardar en historial
            st.session_state["historial"].insert(0, {
                "Hora": datetime.now().strftime("%H:%M:%S"),
                "mo": mo_val, "B": b_val, "UCS": ucs_val, "GSI": gsi_val,
                "Ph": round(float(ph_pred), 4)
            })
    except Exception as e:
        st.error(f"❌ Error en el cálculo: {e}")

# 6. Historial
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Predicciones")
    df_h = pd.DataFrame(st.session_state["historial"])
    st.dataframe(df_h, use_container_width=True)
    if st.button("🗑️ Limpiar Historial"):
        st.session_state["historial"] = []
        st.rerun()
