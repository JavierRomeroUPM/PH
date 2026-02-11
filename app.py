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
        # mo(0), B(1), UCS(2), GSI(3), Peso(4), Dil(5), Form(6), Rug(7)
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
                grid_values, 
                method='linear', 
                bounds_error=False, 
                fill_value=None
            )
            punto_a_interpolar = cont_vals.reshape(1, -1)
            resultado = interp(punto_a_interpolar)
            return float(resultado[0])
        except:
            log_pred = self.xgb.predict(np.array(x).reshape(1, -1))[0]
            return np.expm1(log_pred)

# ==============================================================================
# 2. CONFIGURACIÓN DE PÁGINA Y CARGA DE ACTIVOS
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Suave - Doctorado", layout="wide")

# Inicialización del historial en la sesión
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
st.title("🚀 Predictor Ph - Metamodelo con Historial")
st.markdown("Este simulador permite interpolación suave y guarda un registro de tus consultas.")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 25.0, step=0.1)
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1)
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1)
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables de Simulación")
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad", ["Sin Rugosidad", "Rugoso"], index=0)

    submit = st.form_submit_button("CALCULAR PH", use_container_width=True)

if submit:
    # Mapeo a 0/1
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    
    vec = [mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val]
    ph_resultado = predictor.predecir(vec)
    
    st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")

    # GUARDAR EN EL HISTORIAL
    nuevo_registro = {
        "Fecha/Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mo": mo, "B (m)": b, "UCS": ucs, "GSI": gsi,
        "Peso": v_pp, "Dilat.": v_dil, "Forma": v_for, "Rugos.": v_rug,
        "Ph (MPa)": round(ph_resultado, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# ==============================================================================
# 4. SECCIÓN DE HISTORIAL
# ==============================================================================
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Predicciones")
    
    df_hist = pd.DataFrame(st.session_state["historial"])
    st.dataframe(df_hist, use_container_width=True)
    
    col_h1, col_h2 = st.columns([1, 4])
    
    with col_h1:
        # Botón para descargar CSV
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"historial_ph_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    
    with col_h2:
        # Botón para limpiar historial
        if st.button("🗑️ Borrar Historial"):
            st.session_state["historial"] = []
            st.rerun()

st.markdown("---")
st.caption("Modelo: Interpolador Grid 4D | Python 3.11")
