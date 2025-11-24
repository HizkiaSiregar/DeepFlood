# DeepFlood

A Flask-based image segmentation web application developed for the 2025 Computer Science undergraduate thesis at Klabat University. DeepFlood uses a DeepLabV3+ model with an Xception backbone to detect and segment flood areas from aerial and satellite images. It also integrates Grad-CAM for explainability and model transparency.

Thesis title = Development of Web-based Flood Segmentation From Aerial Imagery Using Deeplabv3+ and LLM

---

## 🚀 Features

- Upload an image and receive a flood segmentation mask
- DeepLabV3+ (Xception backbone) for accurate semantic segmentation
- Grad-CAM explainability for model decision transparency
- Clean web interface using HTML, CSS, and JavaScript
- Flask-based backend for lightweight deployment
- Suitable for academic research and practical demonstrations

---

## 📁 Project Structure

```
project/
│
├── models/
│   └── Xception/
│       └── final_model.keras (NOT included in repo)
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── about.html
│
├── uploads/
│   (empty folder for runtime user uploads)
│
├── app.py
├── config.py
├── database.py
├── model_utils.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Important Note About Model Files

This repository does **not** include the trained machine learning model:

```
final_model.keras
```

This file is not uploaded because:
- It is large and exceeds GitHub's file recommendations
- It is not required for supervisors/lecturers to review the code
- It should be stored locally for security and size reasons

To run the system, manually place the model file here:

```
models/Xception/final_model.keras
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained model

Place your `.keras` model file inside:

```
models/Xception/
```

---

## ▶️ Running the Application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 🎓 Thesis Information

**Project Title:** *DeepFlood: AI-Powered Flood Detection and Segmentation Using DeepLabV3+ with Grad-CAM Explainability*

**Authors:**
* Hizkia Siregar
* Audrey Bambulu
* Gloria Dumaha

**Academic Advisors:**
* Green Arther Sandag, S.Kom., M.S.
* Joe Yuan Mambu, BSIT., MCIS

**University:** Klabat University  
**Department:** Computer Science  
**Year:** 2025

---

## 📚 Method Summary

### **1. Dataset**
Kaggle — Flood Area Segmentation Dataset.

### **2. Model Architecture**
* DeepLabV3+
* Xception backbone
* ASPP (Atrous Spatial Pyramid Pooling)
* Image normalization & resizing

### **3. Training Details**
* 100 epochs
* Adam optimizer
* Binary Cross-Entropy
* Augmentation: rotation, flip, brightness, contrast
* Backbones tested: Xception, ResNet50, MobileNetV2, EfficientNetB0, DenseNet121

### **4. Explainability**
Grad-CAM visualizations highlight which image areas strongly influenced the model's prediction.

---

## 📊 Results Overview

* **IoU:** > 0.87
* **Accuracy:** > 94%
* Strong segmentation stability across multiple conditions
* Efficient for early flood assessment

---

## 📧 Contact

* [s22210002@student.unklab.ac.id](mailto:s22210002@student.unklab.ac.id) — Hizkia Siregar
* [s22210181@student.unklab.ac.id](mailto:s22210181@student.unklab.ac.id) — Audrey Bambulu
* [s22210117@student.unklab.ac.id](mailto:s22210117@student.unklab.ac.id) — Gloria Dumaha

---

## 📄 License

This project is intended for academic and educational use only.
