import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = pickle.load(open("digit_classifier.pkl", "rb"))

# Load the digits dataset (needed for displaying images)
from sklearn.datasets import load_digits
digits = load_digits()

def classify_digit():
    st.title("Digit Classification")

    # User input: area to be entered for classification
    digit_index = st.number_input("Enter an index for the digit image (0-1796):", min_value=0, max_value=1796, step=1)

    # Display button to trigger prediction
    if st.button("Classify"):
        # Fetch the image and features for the chosen digit
        image_data = digits.images[digit_index]
        flattened_data = digits.data[digit_index].reshape(1, -1)

        # Predict the digit using the loaded model
        prediction = model.predict(flattened_data)

        # Display the result
        st.write(f"The predicted digit is: **{prediction[0]}**")

        # Plot and show the image
        plt.imshow(image_data)
        st.pyplot(plt)

# Run the function to display the Streamlit app
classify_digit()
