import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title = "Shiba Vs Akita Classifier",
    page_icon = "🦊",
    layout="centered"
)

st.title("Shiba Vs Akita Classifier")
st.write("Upload an image of to check if the dog is a shiba🦊 or akita🐶. This app uses a vision transformer model to classify the result.")
uploaded_file = st.file_uploader("Upload images", type=["jpg", "png"])
if uploaded_file:
    st.image(uploaded_file)
if st.button("Classify"):
    if not uploaded_file:
        st.error("Please upload an image.")
    else:
        with st.spinner("Classifying... ⏳"):
            
            files = {"file": uploaded_file}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
                result = response.json()
                dog_breed = result.get("result")
                if dog_breed == 'akita':
                    st.header(f"It's an {dog_breed}🐶")

                elif dog_breed == 'shiba':
                    st.header(f"It's a {dog_breed}🦊")
            