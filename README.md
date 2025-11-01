# EcoVision

> "See waste. Know its impact."

EcoVision is a web application that uses AI and OpenCV to identify whether an image of waste is biodegradable or non-biodegradable. It also provides an estimated eco-impact score for the item.

---

## ✨ Features

* **Image Input:** Users can upload or capture an image of a waste item (e.g., plastic bottle, paper, banana peel).
* **AI Classification:** A model, trained using OpenCV + CNN or a pretrained MobileNet, classifies the item into biodegradable or non-biodegradable.
* **Results Display:** The application shows the waste type, an eco-impact score (from 1-10), and a practical tip, such as "This item takes 400 years to decompose.".

---

## 🛠️ System Design

The system is designed with the following components:

* **Frontend:** HTML, CSS, and JavaScript are used for the user-facing side, allowing image uploads and displaying results.
* **Backend (Flask):**
    * A `/predict` route manages the uploaded image.
    * The image is processed using OpenCV and then passed to the machine learning model.
    * The backend serves the final result: "Type + Eco Impact Score + Tip".
* **Storage:**
    * Uploaded images are stored in a local folder.
    * (Optional) A SQLite database can be used for logging or maintaing user history.

---

## 🧠 Model & Dataset

* **Dataset:** The model can be trained using the **"Waste Classification Data"** available on Kaggle. This dataset contains two primary classes: Organic (biodegradable) and Recyclable (non-biodegradable).
* **Preprocessing:** Images are resized to 224x224, and pixel values are normalized.
* **Model:** A **CNN (Convolutional Neural Network)** or a pretrained **MobileNetV2** is used for classification.
* **Training:** The model is trained using TensorFlow/Keras with an 80-20 train-test split to identify the 2 output classes (Biodegradable, Non-Biodegradable).

---

## 🌊 Prototype Flow

1.  **Home Page:** The user is greeted with an "Upload a Waste Image" prompt.
2.  **Backend Processing:** The Flask backend sends the uploaded image to the model for prediction.
3.  **Output Page:** The results page displays:
    * The classification result.
    * An eco-impact score (e.g., 2/10 for biodegradable).
    * A green tip, such as "Compost organic waste to enrich soil".

---

## 🗓️ 3-Week Timeline

| Week | Milestone |
| :--- | :--- |
| **Week 1** | Research + Collect Dataset + Preprocess images |
| **Week 2** | Train CNN or MobileNetV2 + Test model accuracy |
| **Week 3** | Build Flask web app + Integrate model + UI polish |

---

## 🚀 Optional Future Add-ons

* Add a leaderboard for users who recycle the most.
* Integrate the Google Maps API to show nearby recycling centers.
* Gamify the experience to let users earn "eco-points".