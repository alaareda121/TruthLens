<div align="center">

# 🔍 TruthLens
### Forensic Deepfake Face Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-97.99%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI-powered deepfake detection using EfficientNetB3 and Transfer Learning**

[Demo](#demo) • [Features](#features) • [Installation](#installation) • [Results](#results)

</div>

---

## 📌 Overview

TruthLens is a forensic AI system designed to detect AI-generated face images
(deepfakes). Built with EfficientNetB3 and Transfer Learning, trained on
102,041 images from the CelebA + StyleGAN dataset.

> **Academic Project** — Neural Networks & Optimization Courses  
> Faculty of Computers and Information

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 Single Image Analysis | Upload any face image and get instant verdict |
| 📦 Batch Analysis | Analyze multiple images simultaneously |
| ⚖️ Image Comparison | Compare two images side-by-side |
| 🔥 Grad-CAM Heatmap | Visualize which facial regions triggered detection |
| 📊 Error Level Analysis | JPEG compression artifact detection |
| 📄 PDF Report | Download forensic report with legal references |
| ⚖️ Legal Framework | Egyptian law references (Art. 57, Law 175/2018) |

---

## 🧠 Model Architecture
**Training Strategy — Two Phases:**
- **Phase 1:** Frozen base, train head only — `lr=1e-3`, 5 epochs
- **Phase 2:** Fine-tune last 50 layers — `lr=1e-5`, 15 epochs

---

## 📊 Results

### Training Progress

| Phase | Epoch | Train Acc | Val Acc |
|-------|-------|-----------|---------|
| 1 | 5/5 | 82.32% | 86.56% |
| 2 | 5/15 | 92.20% | 96.20% |
| 2 | 10/15 | 95.07% | 97.64% |
| 2 | **15/15** | **96.04%** | **98.06%** |

### Final Test Results (20,000 images)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Fake | 0.97 | 0.99 | 0.98 |
| Real | 0.99 | 0.97 | 0.98 |
| **Overall** | **0.98** | **0.98** | **0.98** |

> **Test Accuracy: 97.99%**

---

## 🗂️ Dataset

- **Source:** [aryansingh16/deepfake-dataset](https://www.kaggle.com/datasets/aryansingh16/deepfake-dataset)
- **Real images:** CelebA dataset
- **Fake images:** StyleGAN-generated faces
- **Split:** 102,041 train / 20,000 valid / 20,000 test
- **Classes:** `{'fake': 0, 'real': 1}`

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/TruthLens.git
cd TruthLens
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the model file
Download `truthlens_best.h5` from the
[Kaggle notebook](YOUR_KAGGLE_LINK)
and place it in the project root:

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
---

## ⚖️ Legal Framework (Egypt)

| Reference | Description |
|-----------|-------------|
| Constitution Art. 57 | Privacy and personal image protection |
| Law 175/2018 | Cybercrime Act — up to 5 years imprisonment |
| Penal Code Art. 327 | Digital image forgery |
| Penal Code Art. 179 | Publishing fabricated images online |

**Report To:**
🏛 NTRA: 155 | 👮 Cybercrime Unit: 08008880 |
⚖️ Prosecution: 16000 | 🛡 ncsc.gov.eg

---

## 🔮 Future Work

- [ ] Video deepfake detection
- [ ] Support for Midjourney/DALL-E/Stable Diffusion
- [ ] Mobile application
- [ ] Real-time camera detection
- [ ] Multi-language support (Arabic)

---

## 📚 References

- Tan, M., & Le, Q. V. (2019). EfficientNet. ICML 2019
- Karras, T., et al. (2019). StyleGAN. CVPR 2019
- Egyptian Cybercrime Law 175/2018

---

<div align="center">

**TruthLens © 2025 — Neural Networks Course Project**

Made with ❤️ using TensorFlow & Streamlit

</div>
