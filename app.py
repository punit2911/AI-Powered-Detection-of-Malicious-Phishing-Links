import streamlit as st
import pickle
from features_extraction import Extractor
import os


# Load the model
ml_model = pickle.load(open('model.pkl', 'rb'))
extractor = Extractor()

# Function to extract features
def extract_features(url):
    features = extractor.extract(url)
    return features

# Function to make predictions
def predict_phishing(features):
    prediction = ml_model.predict([features])
    return prediction

# Placeholder for EDA function
def generate_eda_graphs(features):
    # Implement your EDA graph generation here
    pass

# Streamlit app
st.title("Phishing Classifier")

# Input URL
input_url = st.text_input("Enter URL")


# Predict button
if st.button("Analyse"):

    # Extract features
    features = extract_features(input_url)

    # Make prediction
    prediction = predict_phishing(features)

    # Display prediction
    if prediction == 1:
        st.subheader("Likely a Phishing Website")
        st.image("image_red_cross.png", width=25)
    else:
        st.subheader("Legitimate Website")
        st.image("image_green_tick.png", width=25)


    st.markdown("---")

    feature_names = [
            "Number of Dots",
            "Subdomain Level",
            "Path Level",
            "URL Length",
            "Number of Dashes",
            "Number of Dashes in Hostname",
            "At Symbol",
            "Tilde Symbol",
            "Number of Underscores",
            "Number of Percent Signs",
            "Number of Query Components",
            "Number of Ampersands",
            "Number of Hashtags",
            "Number of Numeric Characters",
            "No HTTPS",
            "Random String",
            "HTTPS in Hostname",
            "Hostname Length",
            "Path Length",
            "Query Length",
            "Double Slash in Path",
            "TinyURL",
            "iFrame",
            "MouseOver",
            "Forwarding",
        ]
    
    extracted_features_list = list(zip(feature_names, features))

    st.subheader("Extracted Features:")

    for name, value in extracted_features_list:
        st.text(f"{name}: {value}")

    # Generate EDA graphs
    generate_eda_graphs(features)

    st.markdown("---")

   
    if prediction == 1:  # Check prediction
        content = ml_model.generate_content(
            "Generate content about phishing websites, detailing their characteristics, common tactics, and how to protect oneself from them. And also provide some online sources."
        )
    else:
        content = ml_model.generate_content(
            "Generate content about legitimate websites, explaining their key features and how they ensure user safety and security. And also provide some online sources."
        )
    st.markdown(content.text)
    

    # learn_more_btn = st.button("Learn More", on_click=learn_more)
