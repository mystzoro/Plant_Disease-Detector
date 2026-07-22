<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=Plant%20Disease%20Detector&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Leaf%20Disease%20Classification&descAlignY=58&descAlign=62" />

  <br />

  [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
  [![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
</div>

---

## 🌿 About

A deep learning model trained to **detect plant diseases from leaf images** using Convolutional Neural Networks (CNN) and transfer learning. The model was trained in Google Colab on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and can classify 38 different plant disease categories.

Simply provide an image of a plant leaf and the model predicts whether it is healthy or diseased — and which disease.

---

## ✨ Features

- 🧠 **CNN-based Architecture** — Deep convolutional network trained end-to-end on leaf images
- 🔄 **Transfer Learning** — Uses pretrained weights (MobileNetV2/VGG16) for faster convergence
- 📊 **High Accuracy** — Achieves 95%+ validation accuracy on the PlantVillage dataset
- 🌾 **38 Disease Classes** — Covers 14 plant species and 26 disease types + healthy variants
- 📓 **Google Colab** — Fully runnable notebook with GPU acceleration
- 🖼️ **Image Prediction** — Feed any leaf image and get disease prediction with confidence score

---

## 🦠 Disease Classes (Sample)

| Plant | Conditions Detected |
|---|---|
| Tomato | Early Blight, Late Blight, Leaf Mold, Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Apple | Apple Scab, Black Rot, Cedar Rust, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| ... | 38 total classes across 14 plant species |

---

## 🏗️ Model Architecture

```
Input (256x256 RGB Image)
    ↓
MobileNetV2 Backbone (pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense(256, ReLU) + Dropout(0.5)
    ↓
Dense(38, Softmax)  ← 38 disease classes
    ↓
Prediction + Confidence Score
```

---

## 📈 Training Details

| Parameter | Value |
|---|---|
| Dataset | PlantVillage (54,306 images) |
| Train/Val Split | 80/20 |
| Image Size | 256×256 |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.0001) |
| Epochs | 25 |
| Validation Accuracy | ~95% |

---

## 🚀 Getting Started

### Run in Google Colab (Recommended)

1. Open the notebook in this repository
2. Click **"Open in Colab"**
3. Run all cells (GPU runtime recommended)

### Run Locally

```bash
# Clone the repo
git clone https://github.com/mystzoro/Plant_Disease-Detector.git
cd Plant_Disease-Detector

# Install dependencies
pip install tensorflow numpy matplotlib pillow jupyter

# Launch Jupyter
jupyter notebook
```

### Predict on Your Own Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('plant_disease_model.h5')
img = image.load_img('your_leaf.jpg', target_size=(256, 256))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
class_idx = np.argmax(pred)
confidence = pred[0][class_idx] * 100
print(f"Prediction: {CLASS_NAMES[class_idx]} ({confidence:.2f}% confidence)")
```

---

## 📁 Repository Structure

```
Plant_Disease-Detector/
├── plant_disease_detection.ipynb   # Main training notebook
├── requirements.txt                 # Dependencies (if running locally)
└── README.md
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ by <a href="https://priyanshuchand.netlify.app/">Priyanshu Chand</a>
</div>
