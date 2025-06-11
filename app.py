import streamlit as st
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image

model = load_model('cat_dog_classifier_final.h5')

st.title("🐶🐱 Cat and Dog Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img = image.resize((128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    prob_dog = pred[0][0]
    prob_cat = 1 - prob_dog
    label = 'Dog 🐶' if prob_dog > 0.5 else 'Cat 🐱'
    confidence = prob_dog if label == 'Dog 🐶' else prob_cat
    st.subheader(f"Prediction: **{label}** ({confidence*100:.2f}% confidence)")
