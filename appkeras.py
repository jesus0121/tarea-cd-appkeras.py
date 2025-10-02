import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ---- UI ----
st.title("JESUS DAVID RODRIGUEZ MULFORD 2025")

# ---- Modelos disponibles ----
modelos_disponibles = ['numerosD1.keras','numerosC2.keras','numerosC3.keras']
modelo_seleccionado = st.selectbox("Selecciona un modelo", modelos_disponibles)

# ---- Carga de modelo con caché ----
@st.cache_resource(show_spinner=False)
def load_model_from_file(modelo_path):
    modelobien = load_model(modelo_path)
    # Solo compila si vas a entrenar/ evaluar con .evaluate; para .predict no es necesario,
    # pero lo dejamos por si luego quieres evaluar.
    modelobien.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return modelobien

modelo = load_model_from_file(modelo_seleccionado)

# ---- Lienzo ----
st.title("Dibuja un número")
canvas_result = st_canvas(
    fill_color="#FFFFFF",           # blanco
    stroke_width=10,
    stroke_color="#000000",         # negro
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def hay_trazo(img_rgba: np.ndarray, umbral=5):
    """
    Devuelve True si hay píxeles “oscuros” (trazo) por encima de un umbral mínimo.
    Evita predecir cuando el lienzo está en blanco.
    """
    # img_rgba shape: (H, W, 4). Pasamos a gris rápido
    if img_rgba is None:
        return False
    # Considera el canal alpha: muchos strokes llegan con alpha < 255
    rgb = img_rgba[..., :3].astype(np.uint8)
    gray = np.mean(rgb, axis=-1)  # 0..255
    # Cuenta píxeles “no blancos”
    no_blancos = np.sum(gray < 250)
    return no_blancos >= umbral

if st.button("Predecir"):
    if canvas_result.image_data is not None and hay_trazo(canvas_result.image_data):
        # Procesar la imagen dibujada
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img = img.convert('L')           # a escala de grises
        img = img.resize((28, 28))       # 28x28

        img_array = np.array(img)

        # Invertir colores (fondo blanco -> 255, trazo negro -> 0) para modelos tipo MNIST
        img_array = 255 - img_array

        # Normalizar y dar forma (1, 28, 28, 1)
        img_array = (img_array / 255.0).astype("float32")
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        # Mostrar la imagen preprocesada (opcional)
        st.image(img, caption="Imagen dibujada", width=140)  # <-- evita use_container_width

        # Predicción
        prediction = modelo.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        predicted_probability = float(np.max(prediction))

        st.success(f"La predicción es: {predicted_class} (prob.: {predicted_probability:.2f})")
    else:
        st.warning("Por favor, dibuja un número antes de predecir.")