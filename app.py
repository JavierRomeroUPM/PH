iimport streamlit as st
import pandas as pd
import numpy as np
import joblib
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
        import pickle
        with open("predictor_grid_4d.pkl", "rb") as f:
            sistema = pickle.load(f)
        return sistema
    except FileNotFoundError:
        st.error("❌ No se encuentra 'predictor_grid_4d.pkl'")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {e}")
        st.stop()

try:
    sistema = load_assets()
    predictor = sistema['predictor']
    metricas = sistema['metricas']
    valores_discretos = sistema['valores_discretos']
    assets_loaded = True
except Exception as e:
    st.error(f"Error al cargar archivos del modelo: {e}")
    assets_loaded = False

# 4. Interfaz de Usuario
st.title("🚀 Predictor de Ph - XGBoost + Interpolación Grid 4D")
st.subheader("Sistema de Alta Precisión con Interpolación Suave")
st.markdown(f"""
Esta versión utiliza **XGBoost + Grid 4D**, permitiendo **interpolación continua** 
entre valores discretos sin efecto escalón.

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
        
        # UCS con ayuda visual
        ucs_val = st.number_input(
            "UCS - Resistencia Compresión Simple (MPa)", 
            min_value=5.0, 
            max_value=100.0, 
            value=50.0, 
            step=0.1, 
            format="%.1f",
            help="Valores de entrenamiento: 5, 10, 50, 100 MPa. Puedes usar valores intermedios."
        )
        
        # GSI con ayuda visual
        gsi_val = st.number_input(
            "GSI - Geological Strength Index", 
            min_value=10, 
            max_value=85, 
            value=50, 
            step=1,
            help="Valores de entrenamiento: 10, 30, 50, 70, 85. Puedes usar valores intermedios."
        )
        
        # mo con ayuda visual
        mo_val = st.number_input(
            "Parámetro mo", 
            min_value=5.0, 
            max_value=32.0, 
            value=20.0, 
            step=0.1, 
            format="%.1f",
            help="Valores de entrenamiento: 5, 12, 20, 32. Puedes usar valores intermedios."
        )
        
    with col2:
        st.subheader("⚙️ Variables No Analíticas")
        
        # B con ayuda visual
        b_val = st.number_input(
            "Ancho de cimentación - B (m)", 
            min_value=4.5, 
            max_value=22.0, 
            value=11.0, 
            step=0.1, 
            format="%.2f",
            help="Valores de entrenamiento: 4.5, 11, 16.5, 22 m. Puedes usar valores intermedios."
        )
        
        # Variables categóricas (binarias)
        st.caption("Variables categóricas (0 o 1):")
        v5_sel = st.selectbox("Peso Propio", ["Sin Peso", "Con Peso"], index=0)
        v6_sel = st.selectbox("Dilatancia", ["Nulo", "Asociada"], index=1)
        v7_sel = st.selectbox("Forma del modelo", ["Plana", "Axisimétrica"], index=1)
        v8_sel = st.selectbox("Rugosidad de la base", ["Sin Rugosidad", "Rugoso"], index=1)
        
    st.markdown("---")
    
    # Botones de acción
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        submit = st.form_submit_button("🎯 CALCULAR PREDICCIÓN", use_container_width=True)
    with col_btn2:
        ejemplo_alto = st.form_submit_button("📈 Ejemplo Ph Alto", use_container_width=True)
    with col_btn3:
        ejemplo_bajo = st.form_submit_button("📉 Ejemplo Ph Bajo", use_container_width=True)

# 5. Lógica de Predicción
if assets_loaded:
    # Cargar ejemplos predefinidos
    if ejemplo_alto:
        mo_val, b_val, ucs_val, gsi_val = 32, 16.5, 100, 85
        v5_sel, v6_sel, v7_sel, v8_sel = "Sin Peso", "Asociada", "Axisimétrica", "Axisimétrica"
        st.info("📈 Cargado ejemplo con Ph alto (~1700)")
        submit = True  # Activar cálculo
    
    if ejemplo_bajo:
        mo_val, b_val, ucs_val, gsi_val = 5, 4.5, 5, 10
        v5_sel, v6_sel, v7_sel, v8_sel = "Con Peso", "Nulo", "Plana", "Sin Rugosidad"
        st.info("📉 Cargado ejemplo con Ph bajo (~0.5)")
        submit = True  # Activar cálculo
    
    if submit:
        try:
            # Conversión de categorías a 0/1
            v5 = 1 if v5_sel == "Con Peso" else 0
            v6 = 1 if v6_sel == "Asociada" else 0
            v7 = 1 if v7_sel == "Axisimétrica" else 0
            v8 = 1 if v8_sel == "Rugoso" else 0
            
            # Crear vector de entrada (orden correcto)
            # [mo, B, UCS, GSI, Peso Propio, Dilatancia, Forma, Rugosidad]
            input_vector = [mo_val, b_val, ucs_val, gsi_val, v5, v6, v7, v8]
            
            # Predicción con Grid 4D
            ph_pred = predictor.predecir(input_vector)
            
            # Validar resultado
            if np.isnan(ph_pred) or np.isinf(ph_pred) or ph_pred < 0:
                st.error("⚠️ Predicción fuera de rango válido")
            else:
                # Mostrar resultado principal
                st.markdown("---")
                col_res1, col_res2, col_res3 = st.columns([2, 1, 1])
                
                with col_res1:
                    st.success(f"### 🎯 Ph Predicho: **{ph_pred:.4f}**")
                
                with col_res2:
                    # Indicador de calidad basado en si está en valores del grid
                    esta_en_grid = (
                        mo_val in valores_discretos['mo'] and
                        b_val in valores_discretos['B'] and
                        ucs_val in valores_discretos['UCS'] and
                        gsi_val in valores_discretos['GSI']
                    )
                    
                    if esta_en_grid:
                        st.info("✅ **Punto del grid**\n\n(Valor exacto)")
                    else:
                        st.warning("🔄 **Interpolado**\n\n(Entre puntos)")
                
                with col_res3:
                    # Clasificación del Ph
                    if ph_pred < 5:
                        categoria = "Muy Bajo"
                        color = "🔵"
                    elif ph_pred < 50:
                        categoria = "Bajo"
                        color = "🟢"
                    elif ph_pred < 500:
                        categoria = "Medio"
                        color = "🟡"
                    elif ph_pred < 1000:
                        categoria = "Alto"
                        color = "🟠"
                    else:
                        categoria = "Muy Alto"
                        color = "🔴"
                    
                    st.metric("Categoría", f"{color} {categoria}")
                
                # Detalles de la predicción
                with st.expander("📊 Detalles de la predicción"):
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.markdown("**Valores de entrada:**")
                        st.code(f"""
mo:           {mo_val}
B (m):        {b_val}
UCS (MPa):    {ucs_val}
GSI:          {gsi_val}
Peso Propio:  {v5}
Dilatancia:   {v6}
Forma:        {v7}
Rugosidad:    {v8}
                        """)
                    
                    with col_det2:
                        st.markdown("**Rendimiento del modelo:**")
                        st.code(f"""
Error máximo: {metricas['grid']['error_max']:.2f}%
MAPE:         {metricas['grid']['mape']:.2f}%
Método:       Grid 4D Interpolado
Escalones:    0% (suave)
                        """)
                
                # Guardar en historial
                st.session_state["historial"].insert(0, {
                    "Hora": datetime.now().strftime("%H:%M:%S"),
                    "mo": mo_val,
                    "B": b_val,
                    "UCS": ucs_val, 
                    "GSI": gsi_val,
                    "PP": v5,
                    "Dil": v6,
                    "Form": v7,
                    "Rug": v8,
                    "Ph": round(float(ph_pred), 4),
                    "Tipo": "Grid" if esta_en_grid else "Interp"
                })
                
                # Limitar historial a 50 entradas
                if len(st.session_state["historial"]) > 50:
                    st.session_state["historial"] = st.session_state["historial"][:50]
        
        except Exception as e:
            st.error(f"❌ Error en el cálculo: {e}")
            st.exception(e)  # Mostrar traceback completo para debug

# 6. Historial
if st.session_state["historial"]:
    st.markdown("---")
    st.subheader("📜 Historial de Predicciones")
    
    df_h = pd.DataFrame(st.session_state["historial"])
    
    # Mostrar tabla con formato
    st.dataframe(
        df_h,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ph": st.column_config.NumberColumn(
                "Ph Predicho",
                format="%.4f"
            ),
            "Tipo": st.column_config.TextColumn(
                "Tipo",
                help="Grid=Punto exacto, Interp=Interpolado"
            )
        }
    )
    
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar Historial"):
        st.session_state["historial"] = []
        st.rerun()
    
    # Botón para descargar historial
    csv = df_h.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Historial (CSV)",
        data=csv,
        file_name=f"historial_predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption(f"""
**Modelo:** XGBoost + Interpolación Grid 4D | 
**Precisión:** Error máx {metricas['grid']['error_max']:.2f}% | 
**Interpolación:** Suave (sin escalones) | 
**Desarrollado para:** Tesis Doctoral
""")