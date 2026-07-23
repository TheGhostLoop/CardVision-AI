# 🂡 CardVision AI

A deep learning-based Playing Card Classification application built from scratch using **PyTorch** and deployed with **Streamlit**.

The model classifies **53 different playing card classes** (including Joker) from an uploaded image and displays the predicted card along with confidence scores.

---

## 🚀 Live Demo

🔗 **Streamlit App:** *(Add your Streamlit URL here)*

---

## 📸 Preview

> Add screenshots or a GIF here.

Example:

```
images/demo.gif
```

---

## ✨ Features

- 🧠 Custom CNN built completely from scratch
- 🃏 Supports **53 Playing Card Classes**
- 📷 Upload any playing card image
- 🎯 Predicts the card with confidence score
- 📊 Displays Top-5 Predictions
- ⚡ Dynamic Quantization (196 MB → 49 MB)
- 🌐 Interactive Streamlit Web App

---

## 🏗 Model Architecture

```
Input Image (224 × 224 × 3)

↓

Conv2D (3 → 32)
ReLU
MaxPool

↓

Conv2D (32 → 64)
ReLU
MaxPool

↓

Conv2D (64 → 128)
ReLU
MaxPool

↓

Flatten

↓

Linear (100352 → 512)

↓

ReLU

↓

Dropout (0.5)

↓

Linear (512 → 53)

↓

Prediction
```

---

## 📊 Training Details

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Classes | 53 |
| Image Size | 224 × 224 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | ✅ |
| Best Model Checkpoint | ✅ |
| Dynamic Quantization | ✅ |

---

## 📈 Model Performance

- **Training Accuracy:** ~95%
- **Validation Accuracy:** ~84%

The model was additionally tested on unseen internet images and demonstrated good generalization under different lighting conditions.

---

## 📂 Project Structure

```
CardVision-AI/
│
├── app.py
├── best_model_quantized.pth
├── requirements.txt
├── README.md
└── images/
    ├── demo.png
    └── architecture.png
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/theghostloop/CardVision-AI.git
```

Move into the project directory

```bash
cd CardVision-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🛠 Tech Stack

- Python
- PyTorch
- TorchVision
- Streamlit
- Pillow

---

## 🎯 Future Improvements

- Transfer Learning using ResNet/EfficientNet
- Grad-CAM Visualizations
- Webcam Live Prediction
- Batch Image Prediction
- Model Explainability

---

## 👨‍💻 Author

**Prince**

If you found this project useful, consider giving it a ⭐ on GitHub!
