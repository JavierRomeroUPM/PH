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
        self.grids_data = {} # Aquí se almacenan las matrices de Ph

    def predecir(self, x):
        """
        Realiza la predicción interpolando suavemente si existe el grid,
        o usando XGBoost como respaldo.
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
        # Se crea en cada llamada para asegurar compatibilidad con la versión de SciPy del servidor
        try:
            interp = RegularGridInterpolator(
                (self.valores_disc['mo'], self.valores_disc['B'], 
                 self.valores_disc['UCS'], self.valores_disc['GSI']),
                grid_values, 
                method='linear', 
                bounds_error=False, 
                fill_value=None
            )
            
            # Ajuste de dimensiones para evitar el TypeError (reshape a 1 fila, 4 columnas)
            punto_a_interpolar = cont_vals.reshape(1, -1)
            resultado = interp(punto_a_interpolar)
            
            return float(resultado[0])
            
        except Exception as e:
            # Si algo falla en la interpolación, usamos el modelo XGBoost base
            log_pred = self.xgb.predict(np.array(x).reshape(1, -1))[0]
            return np.expm1(log_pred)

# ==============================================================================
# 2. CONFIGURACIÓN DE PÁGINA Y CARGA DE DATOS
# ==============================================================================
st.set_page_config(page_title="Simulador Ph Suave - Doctorado", layout="wide")

@st.cache_resource
def load_all_assets():
    try:
        with open("predictor_grid_4d.pkl", "rb") as f:
            # Pickle cargará el diccionario que contiene el objeto 'predictor'
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Archivo 'predictor_grid_4d.pkl' no encontrado. Verifica tu repositorio en GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

# Cargar el sistema completo
sistema = load_all_assets()
predictor = sistema['predictor']
valores_discretos = sistema['valores_discretos']

# ==============================================================================
# 3. INTERFAZ DE USUARIO (STREAMLIT)
# ==============================================================================
st.title("🎯 Predictor Ph - Metamodelo de Alta Fidelidad")
st.markdown("""
Este simulador utiliza una combinación de **XGBoost y Grid 4D** para eliminar el efecto escalón. 
La interpolación n-lineal garantiza transiciones suaves entre las variables analíticas.
""")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Variables Analíticas")
        mo = st.number_input("Parámetro mo", 5.0, 32.0, 25.0, step=0.1, help="Rango entrenado: 5 a 32")
        b = st.number_input("Ancho B (m)", 4.5, 22.0, 11.0, step=0.1, help="Rango entrenado: 4.5 a 22")
        ucs = st.number_input("UCS (MPa)", 5.0, 100.0, 50.0, step=0.1, help="Rango entrenado: 5 a 100")
        gsi = st.number_input("GSI", 10.0, 85.0, 50.0, step=0.1, help="Rango entrenado: 10 a 85")
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        v_pp = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=0)
        v_dil = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v_for = st.selectbox("Forma del modelo", ["Plana", "Axisimétrica"], index=1)
        v_rug = st.selectbox("Rugosidad de la base", ["Sin Rugosidad", "Rugoso"], index=0)

    # Botón de cálculo
    submit = st.form_submit_button("CALCULAR PRESIÓN DE HUNDIMIENTO (Ph)", use_container_width=True)

if submit:
    # Mapeo de inputs a formato numérico 0/1
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
    
    # Ejecución de la predicción suavizada
    ph_resultado = predictor.predecir(vec)
    
    # Mostrar resultados
    st.markdown("---")
    st.success(f"### Ph Predicho: **{ph_resultado:.4f} MPa**")
    
    # Comprobación de si es interpolado o exacto
    es_exacto = (mo in valores_discretos['mo'] and b in valores_discretos['B'] and 
                 ucs in valores_discretos['UCS'] and gsi in valores_discretos['GSI'])
    
    if es_exacto:
        st.info("📍 Punto de control exacto (coincide con la malla de simulación).")
    else:
        st.warning("🔄 Valor interpolado (calculado suavemente entre nodos del hipercubo).")

# Pie de página con información técnica
st.markdown("---")
st.caption(f"Modelo: Interpolador Grid 4D + XGBoost | Python 3.1
