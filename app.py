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
        self.grids_data = {} # Aquí se almacenan las matrices de Ph generadas en el entrenamiento

    def predecir(self, x):
        """
        Realiza la predicción interpolando suavemente si existe el grid,
        o usando XGBoost como respaldo si la combinación no existe.
        """
        # Mapeo de índices: mo(0), B(1), UCS(2), GSI(3), Peso(4), Dilat(5), Forma(6), Rugos(7)
        cat_combo = tuple(int(x[i]) for i in [4, 5, 6, 7])
        cont_vals = np.array([x[0], x[1], x[2], x[3]])
        
        grid_values = self.grids_data.get(cat_combo)
        
        if grid_values is None:
            # Fallback a XGBoost puro (convertir de escala logarítmica a real)
            log_pred = self.xgb.predict(np.array(x).reshape(1, -1))[0]
            return np.expm1(log_pred)
        
        # RECONSTRUCCIÓN DEL INTERPOLADOR
        # Se recrea aquí para asegurar compatibilidad con la versión de SciPy del servidor
        try:
            interp = RegularGridInterpolator(
                (self.valores_disc['mo'], self.valores_disc['B'], 
                 self.valores_disc['UCS'], self.valores_disc['GSI']),
                grid_values, 
                method='linear', 
                bounds_error=False, 
                fill_value=None
            )
            
            # Ajuste crítico de dimensiones para evitar TypeError
            punto_a_interpolar = cont_vals.reshape(1, -1)
            resultado = interp(punto_a_interpolar)
            
            return float(resultado[0])
            
        except Exception as e:
            # Si falla la interpolación por versión o límites, usamos XGBoost base
            log_pred = self.xgb.predict(np.array(x).reshape(1, -1))[0]
            return np.expm1(log_pred)

# ==============================================================================
# 2. CONFIGURACIÓN DE PÁGINA Y CARGA DE ACTIVOS
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Suave - Doctorado", layout="wide")

@st.cache_resource
def load_all_assets():
    try:
        with open("predictor_grid_4d.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Archivo 'predictor_grid_4d.pkl' no encontrado en el repositorio.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

# Cargar el sistema
assets = load_all_assets()
predictor = assets['predictor']
valores_discretos = assets['valores_discretos']

# ==============================================================================
# 3. INTERFAZ DE USUARIO (STREAMLIT)
# ==============================================================================
st.title("🚀 Predictor Ph - Metamodelo de Alta Fidelidad")
st.markdown("""
Este simulador utiliza una arquitectura de **Interpolación en Hipercubo 4D** para eliminar el efecto escalón del XGBoost. 
Esto permite obtener variaciones de presión realistas al modificar mínimamente variables como el UCS o el parámetro mo.
""")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas (Continuas)")
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 25.0, step=0.1)
        b = st.number_input("Ancho de cimentación B (m)", 4.5, 22.0, 11.0, step=0.1)
        ucs = st.number_input("UCS - Resistencia Compresión (MPa)", 5.0, 100.0, 50.0, step=0.1)
        gsi = st.number_input("GSI - Geological Strength Index", 10.0, 85.0, 50.0, step=0.1)
        
    with col2:
        st.subheader("⚙️ Variables de Simulación (Discretas)")
        v_pp = st.selectbox("Peso Propio del Terreno", ["Sin Peso", "Con Peso"])
        v_dil = st.selectbox("Comportamiento de Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Geometría del Modelo", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad de la Base", ["Sin Rugosidad", "Rugoso"], index=0)

    # Botón de cálculo
    submit = st.form_submit_button("CALCULAR PRESIÓN DE HUNDIMIENTO (Ph)", use_container_width=True)

if submit:
    # Mapeo a formato 0/1 para el modelo
    vec = [
        mo, 
        b, 
        ucs, 
        gsi, 
        1 if v_pp == "Con Peso" else 0, 
        1 if v_dil == "Asociada" else 0, 
        1 if v_for == "Axisimétrica" else 0, 
        1 if v_rug == "Rugoso" else 0
    ]
    
    # Ejecución de la predicción con interpolación suave
    ph_resultado = predictor.predecir(vec)
    
    # Mostrar resultados
    st.markdown("---")
    st.success(f"### Resultado Ph Predicho: **{ph_resultado:.4f} MPa**")
    
    # Diagnóstico visual de la predicción
    es_exacto = (mo in valores_discretos['mo'] and b in valores_discretos['B'] and 
                 ucs in valores_discretos['UCS'] and gsi in valores_discretos['GSI'])
    
    if es_exacto:
        st.info("📍 **Punto de Control:** El valor coincide con un nodo de la malla de simulación original.")
    else:
        st.warning("🔄 **Valor Interpolado:** El sistema ha calculado una transición suave entre los nodos más cercanos.")

# Pie de página técnico
st.markdown("---")
st.caption("Modelo: Interpolador Grid 4D (Multilinear) + XGBoost Regressor | Python 3.11 | SciPy Library")
