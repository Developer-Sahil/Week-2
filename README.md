# [cite_start]EcoVision [cite: 1]

> [cite_start]"See waste. Know its impact." [cite: 3]

[cite_start]EcoVision is a web application that uses AI and OpenCV to identify whether an image of waste is biodegradable or non-biodegradable[cite: 5]. [cite_start]It also provides an estimated eco-impact score for the item[cite: 5].

---

## ✨ Features

* [cite_start]**Image Input:** Users can upload or capture an image of a waste item (e.g., plastic bottle, paper, banana peel)[cite: 10].
* [cite_start]**AI Classification:** A model, trained using OpenCV + CNN or a pretrained MobileNet [cite: 12][cite_start], classifies the item into biodegradable or non-biodegradable[cite: 12].
* [cite_start]**Results Display:** The application shows the waste type, an eco-impact score (from 1-10), and a practical tip, such as "This item takes 400 years to decompose."[cite: 13].

---

## 🛠️ System Design

The system is designed with the following components:

* [cite_start]**Frontend:** HTML, CSS, and JavaScript are used for the user-facing side, allowing image uploads and displaying results[cite: 24].
* [cite_start]**Backend (Flask):** [cite: 25]
    * [cite_start]A `/predict` route manages the uploaded image[cite: 27].
    * [cite_start]The image is processed using OpenCV [cite: 28] [cite_start]and then passed to the machine learning model[cite: 29].
    * [cite_start]The backend serves the final result: "Type + Eco Impact Score + Tip"[cite: 30].
* **Storage:**
    * [cite_start]Uploaded images are stored in a local folder[cite: 32].
    * (Optional) [cite_start]A SQLite database can be used for logging or maintaing user history[cite: 33].

---

## 🧠 Model & Dataset

* [cite_start]**Dataset:** The model can be trained using the **"Waste Classification Data"** available on Kaggle[cite: 16, 17]. [cite_start]This dataset contains two primary classes: Organic (biodegradable) and Recyclable (non-biodegradable)[cite: 18].
* [cite_start]**Preprocessing:** Images are resized to 224x224, and pixel values are normalized[cite: 21].
* [cite_start]**Model:** A **CNN (Convolutional Neural Network)** or a pretrained **MobileNetV2** is used for classification[cite: 21].
* [cite_start]**Training:** The model is trained using TensorFlow/Keras with an 80-20 train-test split [cite: 21] [cite_start]to identify the 2 output classes (Biodegradable, Non-Biodegradable)[cite: 21].

---

## 🌊 Prototype Flow

1.  [cite_start]**Home Page:** The user is greeted with an "Upload a Waste Image" prompt[cite: 35].
2.  [cite_start]**Backend Processing:** The Flask backend sends the uploaded image to the model for prediction[cite: 36].
3.  **Output Page:** The results page displays:
    * [cite_start]The classification result (e.g., Biodegradable)[cite: 38].
    * [cite_start]An eco-impact score (e.g., 2/10)[cite: 40].
    * [cite_start]A green tip, such as "Compost organic waste to enrich soil"[cite: 41].

---

## 🗓️ 3-Week Timeline

| Week | Milestone |
| :--- | :--- |
| **Week 1** | [cite_start]Research + Collect Dataset + Preprocess images [cite: 45] |
| **Week 2** | [cite_start]Train CNN or MobileNetV2 + Test model accuracy [cite: 46] |
| **Week 3** | [cite_start]Build Flask web app + Integrate model + UI polish [cite: 47] |

---

## 🚀 Optional Future Add-ons

* [cite_start]Add a leaderboard for users who recycle the most[cite: 49].
* [cite_start]Integrate the Google Maps API to show nearby recycling centers[cite: 50].
* [cite_start]Gamify the experience to let users earn "eco-points"[cite: 51].