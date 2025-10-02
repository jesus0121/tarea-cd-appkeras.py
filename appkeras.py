import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import pandas as pd

# Título
st.title("JESUS DAVID RODRIGUEZ MULFORD 2025")

# Lista de modelos disponibles
modelos_disponibles = ["numerosD1.keras", "numerosC2.keras", "numerosC3.keras"]

# Función para cargar modelos con caché (más eficiente)
@st.cache_resource
def load_model_from_file(modelo_path):
    if not os.path.exists(modelo_path):
        st.error(f"❌ Archivo no encontrado: {modelo_path}")
        return None
    try:
        modelobien = load_model(modelo_path)
        return modelobien  # No es necesario recompilar si solo es para predicción
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo '{modelo_path}': {e}")
        return None

# Cargar los modelos
modelo_d1 = load_model_from_file("numerosD1.keras")
modelo_c2 = load_model_from_file("numerosC2.keras")
modelo_c3 = load_model_from_file("numerosC3.keras")

# Verificar si todos los modelos se cargaron correctamente
if not all([modelo_d1, modelo_c2, modelo_c3]):
    st.stop()  # Detiene la ejecución si faltan modelos

# Lienzo para dibujar
st.subheader("🖌 Dibuja un número")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Mostrar mensaje personalizado según la probabilidad
def mostrar_mensaje(probabilidad, modelo_nombre):
    if probabilidad < 0.8:
        return f" ({modelo_nombre}): ❗ No identificado adecuadamente, intenta nuevamente"
    else:
        return f" ({modelo_nombre}): ✅ Alta confianza ({probabilidad:.2f})"

# Botón de predicción
if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Procesar la imagen
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
        img = img.convert("L")  # Escala de grises
        img = img.resize((28, 28))
        
        # Convertir a array e invertir colores
        img_array = np.array(img)
        #img_array = 255 - img_array  # Invertir: fondo negro, número blanco
        
        # Mostrar la imagen invertida
        st.image(img_array, caption="Imagen procesada (28x28)", use_container_width=False)
        
        # Preparar para el modelo
        img_array = img_array.reshape((1, 28, 28, 1)) / 255.0
                

        with st.spinner("🔍 Realizando predicciones..."):
            # Modelo D1
            prediction_d1 = modelo_d1.predict(img_array)
            predicted_class_d1 = np.argmax(prediction_d1)
            predicted_probability_d1 = np.max(prediction_d1)

            # Modelo C2
            prediction_c2 = modelo_c2.predict(img_array)
            predicted_class_c2 = np.argmax(prediction_c2)
            predicted_probability_c2 = np.max(prediction_c2)

            # Modelo C3
            prediction_c3 = modelo_c3.predict(img_array)
            predicted_class_c3 = np.argmax(prediction_c3)
            predicted_probability_c3 = np.max(prediction_c3)

        # Mostrar resultados en columnas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🧠 Modelo D1")
            st.write(f"Predicción: {predicted_class_d1}" + mostrar_mensaje(predicted_probability_d1, "D1"))

        with col2:
            st.subheader("🧠 Modelo C2")
            st.write(f"Predicción: {predicted_class_c2}" + mostrar_mensaje(predicted_probability_c2, "C2"))

        with col3:
            st.subheader("🧠 Modelo C3")
            st.write(f"Predicción: {predicted_class_c3}" + mostrar_mensaje(predicted_probability_c3, "C3"))

        # Mostrar resumen en tabla
        resultados = pd.DataFrame({
            "Modelo": ["D1", "C2", "C3"],
            "Predicción": [predicted_class_d1, predicted_class_c2, predicted_class_c3],
            "Confianza": [predicted_probability_d1, predicted_probability_c2, predicted_probability_c3]
        })

        st.subheader("📊 Resumen de predicciones")
        st.table(resultados)

    else:
        st.warning("⚠ Por favor, dibuja un número antes de predecir.")
