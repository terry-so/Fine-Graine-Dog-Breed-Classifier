# 🐕 Shiba Inu vs. Akita Image Classifier

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg)

The **Shiba Inu** and **Akita Inu** are two popular dog breeds that are often confused due to their similar appearances, as highlighted in various discussions online:
![reddit](/asset/reddit.png)
To address this common mix-up, this project provides an end-to-end solution for image classification. It covers the entire machine learning pipeline: training a model, tuning hyperparameters, and deploying the final model as an interactive web application.
<table align="center">
  <tr>
    <td align="center">
      <p><b>Shiba Inu</b></p>
      <img src="/asset/shiba.jpg" alt="Shiba Inu" width="300">
    </td>
    <td align="center">
      <p><b>Akita</b></p>
      <img src="/asset/akita.webp" alt="Akita" width="300">
    </td>
  </tr>
</table>

## 🚀 Live Demo
 **Note**:Due to free hosting, the app may take a couple of minutes to load on the first visit.
 **https://shiba-akita-classifier-0305.streamlit.app/**



---
![demo](/asset/demo.gif)

---
## ✨ Project Features

* **Model:** Uses a Vision Transformer (ViT) fine-tuned for this task, achieving about 94% accuracy.
* **Web App Architecture:** The project is a web app with two parts: a FastAPI backend for the model and a Streamlit frontend for the user interface.
* **Hyperparameter Tuning:** Used Weights & Biases (W&B) to automatically test different learning rates and find the best one.
* **Deployment:** The backend is packaged with Docker and hosted on Hugging Face Spaces. The frontend is hosted on Streamlit Community Cloud.
* **User Interface:** The Streamlit app lets you upload an image to get a prediction and a confidence score.

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, HuggingFace, PyTorch, Torchvision, Pillow
* **Frontend:** Streamlit, Requests
* **ML Tools:** Weights & Biases (W&B), Hugging Face Spaces, Docker

---

## 🏛️ Project Architecture

The frontend and backend are separate applications. The Streamlit app (frontend) communicates with the FastAPI server (backend) using API requests. This setup makes the project easier to manage and update.

---

## 🧠 Model Training Steps

1.  **Preparing the Data:** The dataset was compiled by collecting images from Google and Instagram. All images were preprocessed by resizing them to 224x224 pixels and normalizing their pixel values.
2.  **Training the Model:** A pre-trained Vision Transformer (ViT) was fine-tuned on the custom dog dataset. To accelerate training, all layers of the base model were frozen, and only the final classifier head was trained.
3.  **Finding the Best Learning Rate:** A hyperparameter sweep was conducted using Weights & Biases to find the optimal learning rate. The rate that achieved the lowest validation loss was selected.
5.  **Saving the Model:** The final trained model weights were saved to a .pth file for deployment in the web application.

---

