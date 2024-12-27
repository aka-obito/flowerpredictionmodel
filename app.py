import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import streamlit as st

# Define class labels
class_labels = [
    "astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation",
    "common_daisy", "coreopsis", "daffodil", "dandelion", "iris", "magnolia", "rose", "sunflower",
    "tulip", "water_lily"
]

# Load the model
model = load_model('flower_classification_model.keras')

# Preprocess image
def preprocess_image(image_buffer, target_size=(128, 128)):
    image = Image.open(image_buffer).convert('RGB')
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Get the class prediction
def prediction_of_class(output):
    predicted_class = class_labels[np.argmax(output)]
    return predicted_class

# Streamlit app
st.title('Flower Classification')
st.write('Upload a flower photo')

# File uploader
inp_image_buffer = st.file_uploader(label="Upload a flower image (PNG/ JPG)", type=['png', 'jpg'])

if st.button("Classify Flower"):
    if inp_image_buffer is not None:
        # Display uploaded image
        st.image(inp_image_buffer, caption="Uploaded Image", use_column_width=True)

        # Preprocess and classify
        preprocessed_image = preprocess_image(image_buffer=inp_image_buffer)
        try:
            model_output = model.predict(preprocessed_image)
            predicted_class = prediction_of_class(model_output)
            st.write(f'The predicted flower is: **{predicted_class}**')
        except ValueError:
            st.write('Error: Unable to process the image. Please check the input image.')
    else:
        st.write("Please upload a flower image!")
else:
    st.write('Upload a flower image to classify.')
