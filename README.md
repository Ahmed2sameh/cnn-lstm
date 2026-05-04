# CNN-LSTM Image Classification with ResNet50V2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ahmed2sameh/cnn-lstm/blob/main/cnn_lstm_image_classifier.ipynb)

> A hybrid deep learning architecture combining ResNet50V2 (CNN) as a feature extractor with LSTM layers for binary image classification.

## 📌 Overview
This project explores a CNN-LSTM hybrid model for image classification. Instead of treating images as independent frames, the architecture feeds CNN-extracted feature maps into an LSTM to capture sequential/spatial dependencies — a design often applied to video understanding and medical imaging tasks.

## 🧰 Technologies Used
- **Python 3**
- **TensorFlow / Keras** — Model building, training, evaluation
- **ResNet50V2** — Pretrained ImageNet backbone (feature extraction)
- **LSTM** — Sequential layer for feature refinement
- **ImageDataGenerator** — Data loading, normalization, augmentation
- **split-folders** — Automated train/val/test split
- **Matplotlib** — Training curves visualization

## 🏗️ Model Architecture
```
Input (224×224×3)
    ↓
ResNet50V2 (pretrained, frozen) — Feature extraction
    ↓
TimeDistributed(Flatten)
    ↓
LSTM (128 units)
    ↓
Dense (1024, ReLU) → Dense (512, ReLU)
    ↓
Flatten → Dense (2, Softmax) — Binary output
```

## 📊 Training Details
- Optimizer: Adam (lr=0.0001)
- Loss: Sparse Categorical Crossentropy
- Epochs: 10
- Batch size: 16
- Dataset split: 80% train / 15% test / 5% val

## 🚀 How to Run
1. Click the **Open in Colab** badge above
2. Mount your Google Drive and place dataset in `My Drive/cnn-lstm dataset/`
3. Run all cells in order

## 📂 Project Structure
```
├── cnn_lstm_image_classifier.ipynb   # Full pipeline: data loading → model → training → evaluation
└── README.md
```

## 👤 Author
Ahmed Sameh — [GitHub](https://github.com/Ahmed2sameh)
