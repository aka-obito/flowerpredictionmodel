import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

class_labels = [
    "astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation",
    "common_daisy", "coreopsis", "daffodil", "dandelion", "iris", "magnolia", "rose", "sunflower",
    "tulip", "water_lily"
]

model = load_model('flower_classification_model.keras')

def preprocess_image(image_buffer, target_size=(128, 128)):
    image = load_img(image_buffer, target_size=target_size)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


def prediction_of_class(output):
    predicted_class = class_labels[np.argmax(output)]
    print(predicted_class)
    return predicted_class

import streamlit as st

st.title('Flower Classification')
st.write('Upload a flower photo')

inp_image_buffer = st.file_uploader(label="Upload flowe image PNG/ JPG", type=['png', 'jpg'])

if st.button("Classify Flower"):
    if inp_image_buffer is not None:
        preprocessed_image = preprocess_image(image_buffer=inp_image_buffer)
        print("Input shape:", preprocessed_image.shape)
        try:
            model_output = model.predict(preprocessed_image)
        except ValueError:
            st.write('Shape undefined')
        predicted_class = prediction_of_class(model_output)
        st.write(f'The predicted flower is: {predicted_class}')
        st.write(f'Prediction score: {model_output[0][0]}')
    else:
        st.write("Please Upload a flower image!")
else:
    st.write('Upload a flower image')
