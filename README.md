# Sign_language_classifier

# 🧠 Sign Language Recognition using SVM and PCA

This project is a Sign Language recognition system built using **Support Vector Machine (SVM)** combined with **Principal Component Analysis (PCA)** for dimensionality reduction. The interface is built using **Streamlit**, allowing users to interact with the model in a simple web app.

## 📁 Project Files

- `sign_language_SVC.py` – The **Streamlit web app** that loads the model and allows users to make predictions.
- `PCA.pkl` – The pretrained PCA object used to reduce the dimensionality of the input features.
- `sign_svmmodel.pkl` – The trained SVM model used for classifying the sign language gestures.


## 📊 Dataset
The model was trained on the [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) from Kaggle, which contains:
- 28×28 grayscale images
- Hand gestures representing letters A-Z (excluding J and Z which require motion)

  
## 🧰 Technologies Used

- Python
- Streamlit
- scikit-learn
- SVM (Support Vector Machine)
- PCA (Principal Component Analysis)
- NumPy, Pandas  
- joblib (for saving/loading models)

## 🚀 How to Run the Project

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kinzy4/sign_language_classifier.git
   cd sign_language_classifier
2. **Install the dependencies**:

    ```bash
    pip install streamlit scikit-learn numpy pandas joblib

3. **Run the Streamlit app**:

    ```bash
    streamlit run sign_language_SVC.py
 
4. **Using the App**:

   - Upload a **28×28 grayscale image** representing a hand gesture.
   - The model will analyze the image and display the **predicted sign language letter**.
   - You can:
     - Predict a **single letter**, or
     - Upload multiple images one by one to build and display a **complete word**.

   
## 🖼️ Sample Images

You can try out the app using the images in the [`sample_images`](sample_images) folder.

Example:
- [A.png](sample_images/A.png)
- [G.png](sample_images/G.png)


Make sure each image is:
- 28×28 pixels
