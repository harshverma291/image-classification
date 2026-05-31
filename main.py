import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# Web App Title & Header
st.title("Image Classification Model")
st.write("Loading model and preparing dataset, please wait...")

@st.cache_resource
def train_my_model():
    # Dataset Preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        'dataset/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        'dataset/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Build CNN Model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(train_generator.num_classes, activation='softmax'))

    # Compile Model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train Model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10
    )
    
    # Save Model (Must be done before the return statement)
    model.save('image_classifier_model.h5')
    
    # Return both the model and class indices to map predictions later
    return model, train_generator.class_indices

# --- OUTSIDE THE FUNCTION (NO INDENTATION) ---
# This part runs automatically on page load

# Call the function to train/load the model and get class mappings
model, class_indices = train_my_model()

# Reverse the dictionary to map class indices back to labels/names
class_names = {v: k for k, v in class_indices.items()}

st.success("Model loaded successfully!")

# User Interface Section
uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = load_img(uploaded_file, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale to match training data configuration

    # Perform Prediction
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100

    # Display Results on UI
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"### Prediction Result: {predicted_class_name}")
    st.write(f"Confidence Level: {confidence:.2f}%")