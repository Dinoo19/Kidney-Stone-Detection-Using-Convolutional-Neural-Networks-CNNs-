# Kidney-Stone-Detection-Using-Convolutional-Neural-Networks-CNNs(Custom Vs Pretrained)
This project applies deep learning for detecting kidney stones from CT scans using CNNs. It compares a custom CNN with a transfer learning model(VGG16) to evaluate performance. The custom CNN achieved 92% accuracy, outperforming transfer learning, showing model design can surpass pretrained approaches.


# 🧠 Kidney Stone Detection Using CNNs

## 📖 Overview
Kidney stones are a common medical issue that require early detection for effective treatment. This project uses **deep learning** and **image processing** to identify kidney stones from **CT scan images**. It compares a **custom-built CNN** and a **transfer learning** model to analyze performance for medical diagnostics.

---

## 🎯 Objectives
- Build and train CNN models for kidney stone detection.
- Compare CNNs with and without transfer learning.
- Apply preprocessing to improve CT image clarity and contrast.
- Evaluate model performance using accuracy, confusion matrix, and classification metrics.

---

## 🧩 Dataset
Dataset used: [GitHub - Kidney Stone Detection Dataset](https://github.com/yildirimozal/Kidney_stone_detection/tree/main/Dataset)

- ~1700 CT scan images
- Two classes: `Kidney_Stone` and `Normal`
- All images resized to **128x128 pixels**
- **Train/Test split:** 90% / 10%

---

## ⚙️ Data Preprocessing
- **Median Blurring:** Reduces noise in CT images.
- **Histogram Equalization:** Enhances image contrast and feature visibility.
- **Grayscale Conversion:** Simplifies analysis for feature extraction.
- **Normalization:** Standardizes input for model consistency.

These steps improve image quality, highlight stone regions, and enhance feature learning.

---

## 🧠 Model Architectures

### 🧩 **1. Custom CNN Model**
A sequential CNN built from scratch:
- Two convolutional layers with ReLU activation
- Max pooling for dimensionality reduction
- Dense layer with 80 neurons and 50% dropout
- Sigmoid output layer for binary classification  
**Optimizer:** RMSprop  
**Loss:** Binary Crossentropy  
**Accuracy:** **92.18%**

---

### 🧩 **2. Transfer Learning Model**
Based on pretrained architectures (ResNet50, VGG16, Xception):
- Feature extraction from pretrained base
- Added Flatten, Dense (ReLU), Dropout, and Sigmoid output layers  
**Optimizer:** Adam  
**Loss:** Binary Crossentropy  
**Accuracy:** **90.5%**

---

## 📈 Model Evaluation

| Model Type          | Accuracy | Optimizer | Loss Function        |
|----------------------|-----------|------------|----------------------|
| Custom CNN           | 92.18%    | RMSprop    | Binary Crossentropy  |
| Transfer Learning    | 90.5%     | Adam       | Binary Crossentropy  |

- The **custom CNN** achieved higher accuracy than transfer learning, showing that task-specific architectures can outperform pretrained ones for specialized datasets.
- Confusion matrices and loss graphs confirmed stable training and minimal overfitting.

---

## 🧰 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **OpenCV**
- **Matplotlib / Seaborn**
- **Scikit-learn**
