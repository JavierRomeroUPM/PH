import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. DEFINICIÓN DE LA CLASE PARA RECONSTRUCCIÓN DINÁMICA
# ==============================================================================
class InterpoladorGrid4D:
    def __init__(self, modelo_base, nodos, ref_vals, interp_original):
        self.model = modelo_base
        self.nodos = nodos
        self.ref_vals = ref_vals
        self.interp = interp_original

    def predecir(self, vec_8d):
        """
        Si el punto coincide con las variables de referencia del grid, 
        usa la interpolación suave. Si no, usa el modelo base (GBM/XGB) 
        directamente para mantener la fidelidad en 8D.
        """
        # Índices: mo(0), B(1), UCS(2), GSI(3), Peso(4), Dilat(5), Forma(6), Rugos(7)
        # Comprobamos si las variables no analíticas coinciden con el Grid guardado
        coincide_con_grid = (
            vec_8d[4] == self.ref_vals['Peso Propio'] and
            vec_8d[5] == self.ref_vals['Dilatancia'] and
            vec_8d[6] == self.ref_vals['Forma'] and
            vec_8d[7] == self.ref_vals['Rugosidad']
        )

        if coincide_con_grid:
            # Usamos el interpolador lineal (Suavidad Bit-Perfect)
            p_log = self.interp([vec_8d[0], vec_8d[1], vec_8d[2], vec_8d[3]])[0]
            return np.expm1(p_log)
        else:
            # Usamos el Oráculo (GBM) directamente para capturar el efecto 8D
            p_log = self.model.predict(np.array(vec_8d).reshape(1, -1))[0]
            return np.expm1(p_log)

# ==============================================================================
# 2. CONFIGURACIÓN Y CARGA DE ACTIVOS
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Suave - Doctorado", layout="wide")

if "historial" not in st.session_state:
    st.session_state["historial"] = []

@st.cache_resource
def load_all_assets():
    try:
        # Cargamos el pkl que generamos anteriormente
        with open("modelo_hibrido_bit_perfect.pkl", "rb") as f:
            data = pickle.load(f)
        
        # Envolvemos en nuestra clase de lógica doctoral
        return InterpoladorGrid4D(
            modelo_base=data['gbm_base'],
            nodos=data['nodos'],
            ref_vals=data['ref_vals'],
            interp_original=data['interp']
        )
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}. Asegúrate de que 'modelo_hibrido_bit_perfect.pkl' esté en la misma carpeta.")
        st.stop()

predictor = load_all_assets()

# ==============================================================================
# 3. INTERFAZ DE USUARIO
# ==============================================================================
st.title("🚀 Predictor Ph - Metamodelo de Alta Fidelidad")
st.markdown("""
Sistema híbrido de **Interpolación n-lineal** y **Gradient Boosting** para la obtención de superficies de respuesta continuas en macizos rocosos.
""")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas (Mecánica de Rocas)")
        ucs = st.number_input("UCS (MPa) - Resistencia Intacta", 5.0, 100.0, 50.0, step=0.1)
        gsi = st.number_input("GSI - Calidad del Macizo", 10.0, 85.0, 50.0, step=1.0)
        mo = st.number_input("Parámetro geomecánico mo", 5.0, 32.0, 12.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables de Configuración (Escenario)")
        b = st.number_input("Ancho de cimentación B (m)", 4.5, 22.0, 11.0, step=0.1)
        v_pp = st.selectbox("Efecto Peso Propio", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Ley de Dilatancia", ["Nula (No asociada)", "Asociada"], index=1)
        v_for = st.selectbox("Geometría de Zapata", ["Plana (2D)", "Axisimétrica (Circular)"], index=0)
        v_rug = st.selectbox("Condición de Interfaz", ["Liso", "Rugoso"], index=1)

    submit = st.form_submit_button("🎯 CALCULAR CAPACIDAD PORTANTE", use_container_width=True)

if submit:
    # Mapeo a formato numérico (ajustado a cómo se entrenó el modelo)
    # Importante: Estos mapeos deben coincidir exactamente con los de tu Excel
    pp_val = 1 if v_pp == "Con Peso" else 0
    dil_val = 1 if v_dil == "Asociada" else 0
    for_val = 1 if v_for == "Axisimétrica (Circular)" else 0
    rug_val = 1 if v_rug == "Rugoso" else 0
    
    # Vector de entrada: mo, B, UCS, GSI, PP, Dil, Form, Rug
    vec = [mo, b, ucs, gsi, pp_val, dil_val, for_val, rug_val]
    ph_resultado = predictor.predecir(vec)
    
    # Detección de Modo
    # Es exacto si está en los nodos del grid y las variables secundarias coinciden con la referencia
    nodos = predictor.nodos
    es_nodo = (mo in nodos['mo'] and b in nodos['B'] and ucs in nodos['UCS'] and gsi in nodos['GSI'])
    coincide_ref = (pp_val == predictor.ref_vals['Peso Propio'] and 
                    rug_val == predictor.ref_vals['Rugosidad'])
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
        st.caption("Valor obtenido mediante metamodelo de regresión secuencial.")
    
    with res_col2:
        if es_nodo and coincide_ref:
            st.info("🎯 **MODO: PURO (NODO)**\n\nCoincidencia exacta con simulación numérica.")
        elif coincide_ref:
            st.warning("🔄 **MODO: INTERPOLADO**\n\nTransición suave sobre superficie de respuesta.")
        else:
            st.write("🤖 **MODO: INFERENCIA DIRECTA**\n\nCálculo fuera del Grid de referencia.")

    # Guardar en historial
    nuevo_registro = {
        "Fecha": datetime.now().strftime("%H:%M:%S"),
        "UCS": ucs, "GSI": gsi, "mo": mo, "B": b,
        "Peso": v_pp, "Ph (MPa)": round(ph_resultado, 4)
    }
    st.session_state["historial"].insert(0, nuevo_registro)

# ==============================================================================
# 4. HISTORIAL Y CIERRE
# ==============================================================================
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Análisis")
    st.dataframe(pd.DataFrame(st.session_state["historial"]), use_container_width=True, hide_index=True)
    
    if st.button("🗑️ Limpiar Sesión"):
        st.session_state["historial"] = []
        st.rerun()

st.markdown("---")
st.caption("PhD Geotechnical Framework | XGBoost-Grid4D Hybrid Model | © 2024")
