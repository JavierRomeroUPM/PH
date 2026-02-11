import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. DEFINICIÓN DE LA CLASE PARA RECONSTRUCCIÓN DINÁMICA
# ==============================================================================
class InterpoladorGrid4D:
    def __init__(self, modelo_xgb, valores_discretos):
        self.xgb = modelo_xgb
        self.valores_disc = valores_discretos
        self.grids_data = {} 

    def predecir(self, x):
        cat_combo = tuple(int(x[i]) for i in [4, 5, 6, 7])
        cont_vals = np.array([x[0], x[1], x[2], x[3]])
        grid_values = self.grids_data.get(cat_combo)
        
        if grid_values is None:
            log_pred = self.xgb.predict(np.array(x).reshape(1, -1))[0]
            return np.expm1(log_pred)
        
        try:
            interp = RegularGridInterpolator(
                (self.valores_disc['mo'], self.valores_disc['B'], 
                 self.valores_disc['UCS'], self.valores_disc['GSI']),
                grid_values, method='linear', bounds_error=False, fill_value=None
            )
            punto_a_interpolar = cont_vals.reshape(1, -1)
            resultado = interp(punto_a_interpolar)
            return float(resultado[0])
        except:
            log_pred = self.xgb.predict(np.array(x).reshape(1, -1))[0]
            return np.expm1(log_pred)

# ==============================================================================
# 2. CONFIGURACIÓN Y CARGA
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Suave - Doctorado", layout="wide")

if "historial" not in st.session_state:
    st.session_state["historial"] = []

@st.cache_resource
def load_all_assets():
    try:
        with open("predictor_grid_4d.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

assets = load_all_assets()
predictor = assets['predictor']
valores_discretos = assets['valores_discretos']

# ==============================================================================
# 3. INTERFAZ DE USUARIO
# ==============================================================================
st.title("🚀 Predictor Ph - Metamodelo de Alta Fidelidad")
st.markdown("Sistema híbrido **XGBoost + Grid 4D** para la eliminación del efecto escalón.")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1)
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0, step=0.1)
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 20.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1)
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=0)

    submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)

if submit:
    # Mapeo a formato numérico
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    
    # Vector: mo, B, UCS, GSI, PP, Dil, Form, Rug
    vec = [mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val]
    ph_resultado = predictor.predecir(vec)
    
    # DETECCIÓN DE MODO (Interpolado vs Exacto)
    # Comprobamos si las 4 variables continuas están en los arrays originales del grid
    es_exacto = (mo in valores_discretos['mo'] and 
                 b in valores_discretos['B'] and 
                 ucs in valores_discretos['UCS'] and 
                 gsi in valores_discretos['GSI'])
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
    
    with res_col2:
        if es_exacto:
            st.info("🎯 **MODO: PURO**\n\nCoincide con un punto de simulación.")
        else:
            st.warning("🔄 **MODO: INTERPOLADO**\n\nCálculo suave entre nodos.")

    # Guardar en historial
    nuevo_registro = {
        "Fecha/Hora": datetime.now().strftime("%H:%M:%S"),
        "mo": mo, "B": b, "UCS": ucs, "GSI": gsi,
        "Modo": "Puro" if es_exacto else "Interpolado",
        "Ph (MPa)": round(ph_resultado, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# ==============================================================================
# 4. HISTORIAL
# ==============================================================================
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Predicciones")
    df_hist = pd.DataFrame(st.session_state["historial"])
    st.dataframe(df_hist, use_container_width=True)
    
    if st.button("🗑️ Borrar Historial"):
        st.session_state["historial"] = []
        st.rerun()

st.markdown("---")
st.caption(f"Modelo: XGBoost + Interpolador Grid 4D | Python 3.11 | SciPy RegularGridInterpolator")
