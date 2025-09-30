# 🐕 Shiba Inu vs. Akita Image Classifier

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg)

This project classifies images of Shiba Inu and Akita dogs. It covers the full process: training a model, finding the best hyperparameters, and deploying it as a live web application that anyone can use.

## 🚀 Live Demos

[place holder][TBD]

---
<!-- 
ACTION REQUIRED: Add a GIF or a high-quality screenshot of your Streamlit app in action right below this comment. 
Example: ![App Demo](./demo.gif)
-->

---

## ✨ Project Features

* **Model:** Uses a Vision Transformer (ViT) fine-tuned for this task, achieving about 94% accuracy.
* **Web App Architecture:** The project is a web app with two parts: a FastAPI backend for the model and a Streamlit frontend for the user interface.
* **Hyperparameter Tuning:** Used Weights & Biases (W&B) to automatically test different learning rates and find the best one.
* **Deployment:** The backend is packaged with Docker and hosted on Hugging Face Spaces. The frontend is hosted on Streamlit Community Cloud. [TBD]
* **User Interface:** The Streamlit app lets you upload an image to get a prediction and a confidence score. [TBD]

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, HuggingFace, PyTorch, Torchvision, Pillow
* **Frontend:** Streamlit, Requests
* **ML Tools:** Weights & Biases (W&B), Hugging Face Spaces, Docker

---

## 🏛️ Project Architecture

The frontend and backend are separate applications. The Streamlit app (frontend) communicates with the FastAPI server (backend) using API requests. This setup makes the project easier to manage and update.

<!-- 
ACTION REQUIRED: Create a simple diagram (e.g., with diagrams.net or Excalidraw), save it in your repo, and link it here.
Example: ![Architecture Diagram](./architecture.png)
-->

---

## 🧠 Model Training Steps

1.  **Preparing the Data:** Half of the data were prepared by scraping from google and instagram. All images were resized to 224x224 pixels and normalized to prepare them for the model.
2.  **Training the Model:** I used a pre-trained Vision Transformer (ViT) and fine-tuned it on the custom dog dataset. This means I only had to train the final layer, which is much faster.
3.  **Finding the Best Learning Rate:** I used Weights & Biases to automatically test several learning rates and find the one that gave the lowest validation loss.
4.  **Final Training:** The model was trained using the best learning rate and an "early stopping" mechanism to prevent it from overfitting.
5.  **Saving the Model:** The final trained model was saved to a `.pth` file for use in the application.

---

