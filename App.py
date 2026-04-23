import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# 🎨 ESTILOS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #020617, #0c4a6e);
    color: #e2e8f0;
}

/* TITULOS */
.title {
    text-align: center;
    font-size: 42px;
    color: #e0f2fe;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* BOTÓN */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    color: white;
    border-radius: 14px;
    font-size: 18px;
    padding: 10px 20px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.07);
    background: linear-gradient(135deg, #2563eb, #1e40af);
}

/* CANVAS CENTRADO */
canvas {
    display: block;
    margin: auto;
    border-radius: 12px;
}

.element-container:has(canvas) {
    display: flex;
    justify-content: center;
    background: transparent !important;
}

iframe {
    background: transparent !important;
}

/* RESULTADO */
.result {
    font-size: 30px;
    color: #38bdf8;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# 🧠 FUNCIÓN (SIN CAMBIOS)
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# 🐋 HEADER
st.markdown("<div class='title'>🔢 Reconocimiento de Dígitos</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Dibuja un número y deja que la IA lo adivine 💙</div>", unsafe_allow_html=True)

# 🎚️ SLIDER
stroke_width = st.slider('🖌️ Ancho del trazo', 1, 30, 15)

# 🧊 CANVAS CENTRADO
col1, col2, col3 = st.columns([1,2,1])

with col2:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.2)",
        stroke_width=stroke_width,
        stroke_color='#FFFFFF',
        background_color='#000000',
        height=220,
        width=220,
        key="canvas",
    )

# 🔘 BOTÓN
if st.button('✨ Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")

        res = predictDigit(img)

        st.markdown(f"<div class='result'>🔢 Resultado: {res}</div>", unsafe_allow_html=True)
    else:
        st.warning('⚠️ Dibuja un dígito primero')

# 📌 SIDEBAR
st.sidebar.markdown("### 💡 Acerca de")
st.sidebar.write("Esta aplicación utiliza una red neuronal para reconocer dígitos escritos a mano.")
st.sidebar.write("Dibuja un número en el panel y presiona *Predecir*.")
