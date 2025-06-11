import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar modelo entrenado
model = tf.keras.models.load_model('flower_classifier_model')

# Clases según el dataset
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

st.title("Identificador de flores")

uploaded_file = st.file_uploader("Sube una foto de una flor", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(180, 180))
    st.image(img, caption='Imagen subida', use_column_width=True)
    
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    
    predictions = model.predict(x)
    score = tf.nn.softmax(predictions[0])
    
    st.write(f"Predicción: **{class_names[np.argmax(score)]}**")
    st.write(f"Confianza: {100 * np.max(score):.2f}%")
