# Simple Image Classification with Logistic Regression

This repository contains the code and resources for the **Simple Image Classification** project using Logistic Regression. The project was developed as part of an internship program at EI Systems in the domain of Machine Learning.

## Project Overview

In this project, we classify handwritten digits (0-9) using the **Digits dataset** from Scikit-learn. The dataset consists of 8x8 pixel grayscale images of digits, and a Logistic Regression model is used to predict the digits based on their pixel values.

### Files in the Repository

- **Project.ipynb**: Jupyter Notebook file that contains the full code for training the model, evaluating its performance, and making predictions.
- **proj.py**: Python script to deploy the trained model using **Streamlit**. This app allows users to input a digit index and get the predicted digit along with the image of the digit.
- **digit-classifier.pickle**: The trained Logistic Regression model saved in a pickle file, which is used for deployment.
- **deployment-video.webm**: Video demonstrating the deployment of the model via the Streamlit interface.

## How to Run the Project

### Prerequisites
Make sure you have the following libraries installed:
- `numpy`
- `matplotlib`
- `scikit-learn`
- `streamlit`
- `pickle`

You can install the required packages using pip:
```bash
pip install numpy matplotlib scikit-learn streamlit
```
### Running the Jupyter Notebook
To explore the project, open the Project.ipynb file in Jupyter Notebook and run the code cells. This will walk you through:

`Loading the dataset
Training the Logistic Regression model
Evaluating its performance
Making predictions
Running the Streamlit App`

### To deploy the model and interact with the Streamlit app, use the following command in your terminal:

`streamlit run proj.py`

This will launch a local web application where you can input a digit index and view the prediction along with the digit image.

### YouTube Video Link
Watch the deployment of the model via Streamlit on YouTube: [Click here to view the video](https://www.youtube.com/watch?v=NWxn6zqZd0I)

### License
This project is open-source and available for anyone to use or modify.
Note: Be sure to provide appropriate attributions if you use any external datasets or libraries in your project.
