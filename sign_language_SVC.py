import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load model and PCA
model = joblib.load('sign_svmmodel.pkl')
pca = joblib.load('pca.pkl')

# App Title and Styling
st.set_page_config(page_title="Sign Language Classifier", page_icon="✋", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Sign Language Classifier ✋</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a 28x28 grayscale image of a hand sign to recognize the letter.</p>", unsafe_allow_html=True)

# Initialize session state for word building
if 'letters' not in st.session_state:
    st.session_state.letters = []

# File uploader
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Grayscale
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized) / 255.0
    img_flatten = img_array.reshape(1, -1)
    img_pca = pca.transform(img_flatten)

    prediction = model.predict(img_pca)
    predicted_label = chr(prediction[0] + 65)

    # Layout
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.success(f"🧠 **Predicted Sign:** {predicted_label}")

        # Word-building buttons
        if st.button("➕ Add to Word"):
            st.session_state.letters.append(predicted_label)

        if st.button("🧹 Clear Word"):
            st.session_state.letters = []

        if st.session_state.letters:
            current_word = ''.join(st.session_state.letters)
            st.info(f"🔡 Current Word: **{current_word}**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

