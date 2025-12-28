<p align="center">
  <img src="https://img.shields.io/github/license/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark?style=for-the-badge" />
  <img src="https://img.shields.io/badge/School%20Project-ML%20%26%20Data-blueviolet?style=for-the-badge" />
</p>

<h1 align="center">✨ Cityscapes Image Segmentation Benchmark ✨</h1>

<div align="center">
  <em>
     Benchmarking machine perception for urban environments
  </em>
</br>

 <b>
   End-to-end semantic image segmentation benchmark for autonomous driving
 </b>
</br>
</br>
🗃️ <b>Dataset</b>  

      https://www.cityscapes-dataset.com/dataset-overview/
  
</div>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>🧭 Table of Contents</summary>
  <ol>
    <li>About The Project</li>
    <li>Dataset</li>
    <li>System Architecture</li>
    <li>Models presentation</li>
    <li>Model Evaluation</li>
    <li>Repository Structure</li>
    <li>Getting Started</li>
    <li>License</li>
    <li>Contact</li>
  </ol>
</details>

---

### ✨ Built With

[![Python][Python-shield]][Python-url]
[![Jupyter][Jupyter-shield]][Jupyter-url]
[![Pandas][Pandas-shield]][Pandas-url]
[![NumPy][NumPy-shield]][NumPy-url]
[![Matplotlib][Matplotlib-shield]][Matplotlib-url]
[![Seaborn][Seaborn-shield]][Seaborn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🎯 About The Project

This project is an end-to-end semantic image segmentation benchmark based on the Cityscapes dataset.

It is designed to compare different deep learning approaches for urban scene understanding, from classical CNN-based architectures to more recent transformer-based models.

The project covers the full machine learning pipeline:
- Data preparation and preprocessing
- Model training and evaluation
- Quantitative benchmarking
- Inference via a FastAPI backend
- Visualization through a Streamlit frontend

---

## 🗃️ Dataset

The project uses the **Cityscapes** dataset, a large-scale benchmark dataset dedicated to semantic understanding of urban street scenes.

Dataset overview:

      https://www.cityscapes-dataset.com/dataset-overview/

The dataset provides:
- High-resolution street images
- Pixel-level semantic annotations
- Real-world driving scenarios

---

## 🏰 System Architecture

The project follows a modular end-to-end architecture:

- **Modelisation**: training, experimentation, and benchmarking notebooks
- **Backend**: FastAPI inference service exposing segmentation predictions
- **Frontend**: Streamlit web application for visualization and comparison
- **Tracking**: MLflow for experiment reproducibility and metrics logging

Each component is isolated to ensure clarity, maintainability, and reproducibility.

---

## 🪄 Models presentation

The benchmark includes multiple segmentation models, such as:

- CNN-based architectures (baseline models)
- Encoder-decoder segmentation networks
- Transformer-based segmentation models

Each model is trained and evaluated under the same conditions to ensure fair comparison.

---

## 👑 Model Evaluation

Models are evaluated using standard semantic segmentation metrics:

- Mean Intersection over Union (mIoU)
- Training and validation loss
- Class-wise performance
- Qualitative visual comparison of predicted masks

All experiments and metrics are tracked using MLflow.

---

## 🗺️ Repository Structure

    SchoolProject---Cityscapes-image-segmentation-benchmark/
    ├── back/
    │   ├── main.py                 # FastAPI inference backend
    │   ├── model/                  # Trained / exported models
    │   ├── requirements.txt
    │   └── README.md
    │
    ├── front/
    │   ├── app.py                  # Streamlit visualization app
    │   ├── requirements.txt
    │   └── README.md
    │
    ├── modelisation/
    │   ├── notebooks/              # Training & evaluation notebooks
    │   ├── mlruns/                 # MLflow experiment tracking
    │   └── README.md
    │
    ├── .gitignore
    ├── .gitattributes
    └── README.md

---

## ⚔️ Getting Started

### 1. Clone the repository

    git clone https://github.com/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark.git

### 2. Set up the backend

    cd back
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    uvicorn main:app --reload

### 3. Run the frontend

    cd ../front
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

---

## ✒️ License

This project is intended for educational and research purposes.

---

## 🕊️ Contact

Joëlle JEAN BAPTISTE  
LinkedIn:

      https://fr.linkedin.com/in/joëllejnbaptiste  

Project Link:

      https://github.com/joelle-jnbaptiste/SchoolProject---Cityscapes-image-segmentation-benchmark

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

[Jupyter-shield]: https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white
[Jupyter-url]: https://jupyter.org/

[Pandas-shield]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/

[NumPy-shield]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/

[Matplotlib-shield]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge
[Matplotlib-url]: https://matplotlib.org/

[Seaborn-shield]: https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge
[Seaborn-url]: https://seaborn.pydata.org/
