# Import required libraries
import streamlit as st
import numpy as np
import pickle

# Set custom CSS for padding
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 4rem !important;
            padding-bottom: 0rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Configure Streamlit page
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Load the trained iris classification model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Main title
st.title("Iris Flower Classification Model ")

# Sidebar: About the project
st.sidebar.markdown("## About The Project")

# Sidebar: Project info and styling
st.sidebar.markdown(
    """
    <style>
    .custom-info {
        background-color: #ff4b4b;
        color: white;
        padding: 16px;
        border-radius: 5px;
        font-size: 16px;
        margin-bottom: 16px;
    }
    </style>
    <div class="custom-info">
        This app predicts the species of an Iris flower based on its features.<br>
        Adjust the sliders and click <b>Predict</b> to see the result.
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar: Display image and model accuracy
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Flower")
st.sidebar.markdown("**Model accuracy:** 97%")
st.sidebar.markdown("---")
# Sidebar: Author link
st.sidebar.markdown(
    """
    <style>
    .custom-link {
        color: #ff4b4b !important;
        text-decoration: none !important;
    }
    </style>
    Made by <a href="https://github.com/abhisheku007" class="custom-link" target="_blank">Abhishek Upadhyay</a>
    """,
    unsafe_allow_html=True
)

# Main content: Feature selection sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

# Layout for prediction button and result
col1, col2, col3 = st.columns([1.5, 1, 1])

with col2:
    if st.button("Predict"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        probab = model.predict_proba(features)
        
        species = ['Setosa', 'Versicolor', 'Virginica']
        st.markdown(
        f"""
        <div style='text-align:center; margin-left:-100px; margin-bottom:50px; white-space:nowrap;'>
            The predicted species is: <b>{species[prediction[0]]}</b>

            Confidence: {100*max(probab[0]):.2f}%
        </div>
        """,
        unsafe_allow_html=True
        )
