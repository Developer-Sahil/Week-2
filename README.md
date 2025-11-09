# EcoVision

> "See waste. Know its impact."

EcoVision is an AI-powered web application that uses deep learning and computer vision to classify waste items as biodegradable or non-biodegradable. The application provides users with eco-impact scores, decomposition time estimates, and practical recycling tips to promote environmental awareness.

---

## ✨ Features

* **Image Upload & Classification:** Users can upload images of waste items (plastic bottles, paper, food waste, etc.) for instant AI-powered classification
* **MobileNetV2 Model:** Leverages transfer learning with MobileNetV2 for accurate predictions (~89% validation accuracy)
* **Eco-Impact Assessment:** 
  - Eco-score rating (1-10 scale)
  - Estimated decomposition time
  - Personalized recycling tips
* **Fast Predictions:** Optimized model provides results in under a second
* **RESTful API:** Clean Flask backend with `/predict` and `/health` endpoints
* **Responsive Frontend:** Basic web interface for image uploads and result visualization

---

## 🛠️ Tech Stack

### Backend
- **Flask:** Lightweight web framework for API endpoints
- **TensorFlow/Keras:** Deep learning framework for model training and inference
- **OpenCV:** Image preprocessing and computer vision operations
- **NumPy:** Numerical computing for array operations

### Frontend
- **HTML/CSS/JavaScript:** Basic interface for user interaction
- **AJAX:** Asynchronous communication with Flask backend

### Model
- **MobileNetV2:** Pre-trained on ImageNet, fine-tuned for waste classification
- **Binary Classification:** Sigmoid activation for biodegradable vs. non-biodegradable

---

## 📊 Model Architecture

The model uses transfer learning with MobileNetV2 as the base:

```
MobileNetV2 (frozen, pre-trained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (1 unit, sigmoid activation)
```

**Training Configuration:**
- Image Size: 96×96 pixels
- Batch Size: 128
- Epochs: 5
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: Binary Cross-Entropy
- Training Time: ~2-3 minutes

**Performance:**
- Training Accuracy: ~92%
- Validation Accuracy: ~89%

---

## 🗂️ Dataset

The model is trained on the **Waste Classification Data** from Kaggle:
- **Classes:** 
  - Organic (O) - Biodegradable waste
  - Recyclable (R) - Non-biodegradable waste
- **Source:** [Kaggle - Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- **Split:** 80% training, 20% testing

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Developer-Sahil/Week-2.git
cd Week-2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (if not already trained):
```bash
python train_model.py
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

---

## 📁 Project Structure

```
EcoVision/
│
├── app.py                          # Flask backend API
├── train_model.py                  # Model training script
├── waste_classifier_model.h5       # Trained model (excluded from Git)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── templates/
│   └── index.html                  # Frontend interface
│
├── uploads/                        # Temporary storage for uploaded images
│
├── DATASET/                        # Training dataset (excluded from Git)
│   ├── TRAIN/
│   │   ├── O/                      # Organic/Biodegradable
│   │   └── R/                      # Recyclable/Non-biodegradable
│   └── TEST/
│       ├── O/
│       └── R/
│
└── training_history.png            # Training metrics visualization
```

---

## 🌊 API Endpoints

### POST /predict
Upload an image for classification.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "classification": "Biodegradable (Organic)",
  "confidence": "94.32%",
  "eco_score": "2/10",
  "tip": "Compost organic waste to enrich soil and reduce landfill waste!",
  "decompose_time": "2-6 months"
}
```

### GET /health
Check API status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 📈 Week-by-Week Progress

### Week 1: Planning & Data Preparation
- Researched waste classification approaches
- Downloaded and explored the Kaggle dataset
- Created project plan and architecture design
- Preprocessed images (resize, normalization)

### Week 2: Model Training & Backend Development
- Implemented MobileNetV2 with transfer learning
- Optimized training pipeline (96×96 images, batch size 128, 5 epochs)
- Achieved 89% validation accuracy in ~2-3 minutes
- Built Flask API with preprocessing pipeline
- Implemented eco-impact assessment system
- Created basic frontend interface
- Added comprehensive error handling and file validation

### Week 3: UI/UX Polish & Deployment (Planned)
- Enhanced frontend design with modern UI/UX
- Responsive mobile design
- Optional features: leaderboard, nearby recycling centers, gamification
- Deployment to cloud platform

---

<!-- ## 🌱 Future Enhancements

- **UI/UX Improvements:** Modern, responsive design with better visual feedback
- **Multi-class Classification:** Expand to specific waste categories (paper, plastic, metal, glass)
- **Gamification:** User accounts, eco-points, and recycling leaderboards
- **Location Services:** Integration with Google Maps API to show nearby recycling centers
- **Mobile App:** Native iOS/Android applications
- **Batch Processing:** Support for multiple image uploads
- **Historical Tracking:** User dashboard showing recycling impact over time

--- -->

<!-- ## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open source and available under the MIT License.

--- -->

## 👨‍💻 Author

**Sahil Sharma**
- GitHub: [@Developer-Sahil](https://github.com/Developer-Sahil)

---

## 🙏 Acknowledgments

- Dataset: [Waste Classification Data on Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- MobileNetV2: [Google's TensorFlow Models](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
- Inspiration: Building sustainable solutions for environmental awareness