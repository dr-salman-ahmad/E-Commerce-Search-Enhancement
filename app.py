import streamlit as st
from PIL import Image
from main import ProductSearch

st.title("Ecommerce Products")
st.session_state['text_input'] = st.text_input("Enter some text:")
st.session_state['uploaded_file'] = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"],
                                                     label_visibility="collapsed")

if 'app' not in st.session_state:
    st.session_state['app'] = ProductSearch()

if st.button("Generate Embeddings"):
    st.session_state['app'].prepare_data(crop_limit=100)
    st.session_state['app'].add_data_to_vector_db()
    st.write("Embeddings Completed Successfully")


if len(st.session_state['text_input']) > 0:
    st.session_state['app'].run(st.session_state['text_input'])
    st.session_state['text_input'] = ""

if st.session_state['uploaded_file'] is not None:
    image = Image.open(st.session_state['uploaded_file'])
    st.session_state['app'].run(image)
    st.session_state['uploaded_file'] = None
