[![Stars][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

---

<div align="center">

# 🏰✨ SchoolProject — Cityscapes Image Segmentation Benchmark ✨🏰

🧙‍♂️ *An end-to-end semantic segmentation benchmark crafted like a grimoire,  
from raw urban scenes to deployed inference spells.*

⚔️ **School Project — Machine Learning & Computer Vision Engineering**

🗺️ **Dataset**  
👉 https://www.cityscapes-dataset.com/dataset-overview/

</div>

---

## 📜 About the Project

🧠 Autonomous driving systems depend on **semantic segmentation** to perceive and understand complex urban environments: roads, vehicles, pedestrians, buildings, and more.

This project is a **complete benchmark pipeline** built around the **Cityscapes dataset**, designed to compare:

- 🏹 **CNN-based architectures** (DeepLabV3+)
- 🧙‍♂️ **Transformer-based models** (Mask2Former / SegFormer-style)

The objective is not only performance, but also **reproducibility, interpretability, and deployment-readiness**, following professional ML & MLOps standards.

Covered end-to-end:

- 🧹 Data preprocessing & class remapping  
- 🏗️ Model training & evaluation  
- 📊 Quantitative benchmarking (mIoU, loss, class-wise metrics)  
- 🧪 Experiment tracking with MLflow  
- 🏰 FastAPI inference backend  
- 🔮 Streamlit web interface for visualization & comparison  

---

## 🛠️✨ Built With

<div align="center">

![Python][python-shield]
![PyTorch][pytorch-shield]
![TensorFlow][tensorflow-shield]
![FastAPI][fastapi-shield]
![Streamlit][streamlit-shield]
![Docker][docker-shield]

</div>

🧙‍♀️ *Each tool is a rune in the spellbook of this end-to-end system.*

---

## 🗺️ Dataset

📚 **Cityscapes** is a large-scale dataset dedicated to semantic understanding of urban street scenes, widely used in autonomous driving research.

**Key characteristics:**

- 🌆 5,000 finely annotated images  
- 🏙️ 20,000 coarsely annotated images  
- 🎯 Pixel-level semantic labels  
- 🚗 Real-world driving scenarios  

**Semantic classes used in this benchmark:**

- Road  
- Sidewalk  
- Building  
- Vehicle  
- Pedestrian  
- Vegetation  
- Sky  
- Background / Ignore  

🔗 Official dataset page:  
👉 https://www.cityscapes-dataset.com/dataset-overview/

---

## 🏗️ Project Structure

        SchoolProject---Cityscapes-image-segmentation-benchmark/
        │
        ├── back/ # 🏰 FastAPI inference backend
        │ ├── main.py # API entrypoint
        │ ├── requirements.txt # Backend dependencies
        │ ├── install-app.sh # App setup script
        │ └── install-conda.sh # Conda environment setup
        │
        ├── front/ # 🔮 Streamlit visualization app
        │ ├── app.py # UI entrypoint
        │ ├── metric_info.py # Metrics display helpers
        │ └── requirements.txt # Frontend dependencies
        │
        ├── modelisation/ # 🧙 Model training & experiments
        │ ├── notebooks (.ipynb) # Training & evaluation notebooks
        │ ├── mlruns/ # MLflow experiment tracking
        │ └── README.md # Modeling-specific documentation
        │
        ├── .gitignore
        ├── .gitattributes
        └── README.md # Project documentation
---

## 🧪 Models Benchmarked

⚔️ **CNN-based Baseline**

- DeepLabV3+ (ResNet backbone)
- Strong spatial inductive bias
- Efficient and stable baseline

🧙‍♂️ **Transformer-based Model**

- Mask2Former / SegFormer-style architecture
- Global context modeling
- Better handling of complex urban scenes

---

## 📊 Evaluation & Metrics

🔍 Models are evaluated using:

- 📐 **Mean Intersection over Union (mIoU)**  
- 📉 Training & validation loss  
- 🧮 Class-wise IoU  
- ⏱️ Inference latency  
- 🖼️ Qualitative visual comparisons  

🧪 All experiments are logged with **MLflow** to ensure full traceability and reproducibility.

---

## 🏰 Backend — FastAPI Inference API

⚙️ A production-ready API providing semantic segmentation inference.

**Features:**

- Automatic image preprocessing  
- Model loading & inference  
- Post-processing (argmax + color mapping)  
- PNG segmentation mask output  

📜 Interactive documentation available via `/docs`.

---

## 🔮 Frontend — Streamlit Application

🧭 The Streamlit app allows users to:

- Upload street-scene images  
- Visualize predicted segmentation masks  
- Compare outputs from different models  
- Inspect metrics interactively  

Designed for **demonstration, comparison, and explainability**.

---

## 🧙‍♀️ Reproducibility & MLOps

- 📦 Environment isolation via `requirements.txt`  
- 🧪 Experiment tracking with MLflow  
- 🐳 Docker-ready architecture  
- 📁 Clear separation between training, API, and UI  

---

## 📜 License

🛡️ This project is intended for **educational and research purposes**.  
You are free to reuse, adapt, and extend it for learning or demonstration.

---

✨ *May your gradients vanish not, and your mIoU rise ever higher.* ✨

---

[stars-shield]: https://img.shields.io/github/stars/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark?style=flat-square
[stars-url]: https://github.com/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark/stargazers
[issues-shield]: https://img.shields.io/github/issues/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark?style=flat-square
[issues-url]: https://github.com/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark/issues
[license-shield]: https://img.shields.io/github/license/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark?style=flat-square
[license-url]: https://github.com/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat-square
[linkedin-url]: https://www.linkedin.com/

[python-shield]: https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[pytorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[tensorflow-shield]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[fastapi-shield]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[streamlit-shield]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[docker-shield]: https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white
[gha-shield]: https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white






