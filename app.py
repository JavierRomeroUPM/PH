import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime # IMPORTANTE: Verificar que esté aquí
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. LA CLASE DEBE IR PRIMERO (Antes de cargar el pickle)
# ==============================================================================
class InterpoladorGrid4D:
    def __init__(self, modelo_base, nodos, ref_vals, interp_original):
        self.model = modelo_base
        self.nodos = nodos
        self.ref_vals = ref_vals
        self.interp = interp_original

    def predecir(self, vec_8d):
        coincide_con_grid = (
            vec_8d[4] == self.ref_vals['Peso Propio'] and
            vec_8d[5] == self.ref_vals['Dilatancia'] and
            vec_8d[6] == self.ref_vals['Forma'] and
            vec_8d[7] == self.ref_vals['Rugosidad']
        )
        if coincide_con_grid:
            p_log = self.interp([vec_8d[0], vec_8d[1], vec_8d[2], vec_8d[3]])[0]
            return np.expm1(p_log)
        else:
            p_log = self.model.predict(np.array(vec_8d).reshape(1, -1))[0]
            return np.expm1(p_log)

# ==============================================================================
# 2. CARGA DE DATOS (Usando nombres exactos del archivo generado)
# ==============================================================================
@st.cache_resource
def load_all_assets():
    # CAMBIA ESTO al nombre real de tu archivo .pkl
    nombre_pkl = "modelo_hibrido_bit_perfect.pkl" 
    
    if not os.path.exists(nombre_pkl):
        st.error(f"Archivo {nombre_pkl} no encontrado.")
        st.stop()
        
    with open(nombre_pkl, "rb") as f:
        data = pickle.load(f)
    
    # Reconstrucción del objeto usando la clase definida arriba
    return InterpoladorGrid4D(
        modelo_base=data['gbm_base'],
        nodos=data['nodos'],
        ref_vals=data['ref_vals'],
        interp_original=data['interp']
    )

predictor = load_all_assets()
# ... resto del código (Interfaz, historial, etc.)
