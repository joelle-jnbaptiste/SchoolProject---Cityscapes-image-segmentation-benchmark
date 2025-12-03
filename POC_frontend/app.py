from metric_info import metric_info
import plotly.express as px
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow
import io
import base64
import streamlit as st
import requests
from PIL import Image

# -------------------------------------------------------------------
# CONFIG — ACCESSIBILITÉ
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Comparaison Segmentation – DeepLab vs Mask2Former",
    layout="wide"
)


PRIMARY_COLOR = "#003566"   
SECONDARY_COLOR = "#f1f1f1"  

# -------------------------------------------------------------------
# TITRE + INTRO
# -------------------------------------------------------------------

st.header("Preuve de Concept : Segmentation d’images avec DeepLabV3+ et Mask2Former")

st.markdown("""
### Dataset : Cityscapes (version réduite à 8 classes)

Ce projet utilise une version simplifiée du dataset **Cityscapes**, largement utilisé 
pour la recherche en **segmentation d’images dans les systèmes de conduite autonome**.

Les images proviennent de scènes urbaines (Allemagne) capturées depuis un véhicule,  
et chaque image possède un **masque sémantique** où chaque pixel correspond à une classe.

Nous utilisons ici une version regroupée du dataset, limitée à **8 grandes catégories** :
""")

# --- Classes ---
st.markdown("""
#### Les 8 classes retenues
- **flat** — route, trottoir  
- **human** — piétons  
- **vehicle** — voitures, bus, camions  
- **construction** — bâtiments, structures  
- **object** — panneaux, barrières, poteaux  
- **nature** — végétation, arbres  
- **sky** — ciel  
- **void** — pixels ignorés ou non pertinents  
""")

# --- Répartition ---
st.markdown("""
## Répartition du dataset

Deux configurations ont été utilisées pour analyser le comportement des modèles :

- **Train : 300** images &nbsp; | &nbsp; **Val : 50** images  
- **Train : 2975** images &nbsp; | &nbsp; **Val : 500** images  

La première (300/50) sert à tester l’apprentissage avec peu de données.  
La seconde (2975/500) permet d’évaluer les modèles à plus grande échelle.
""")

# --- Exemple image + masque ---
st.markdown("""
## Exemple d’image et masque de vérité terrain (GT)

Voici un exemple permettant de visualiser ce à quoi ressemblent les données utilisées
pour l’entraînement du modèle.
""")

img = Image.open("img/masque.png")
st.image(img, use_container_width=True)
st.markdown(
    "<p aria-label='Image segmentée DeepLab, chaque couleur représente une classe du masque sémantique.'></p>",
    unsafe_allow_html=True
)


# --- Entraînement ---
st.markdown("""
### Brève description de l’entraînement

Chaque modèle apprend à prédire, pour **chaque pixel**, la classe correcte parmi 8 catégories.

L’entraînement suit les étapes suivantes :

1. **Prétraitement** : redimensionnement, normalisation, encodage des masques.  
2. **Architecture du modèle** :  
   - **DeepLabV3+ (ResNet50)** — CNN avec décodeur dilaté, rapide et stable  
   - **Mask2Former** — architecture transformer moderne, très performante  
3. **Suivi des métriques** :  
    - mIoU
    - pixel accuracy
    - pertes
    - vitesse  
4. **Validation** :  
   - le modèle est testé sur un jeu **jamais vu**  
   - comparaison systématique DeepLabV3+ vs Mask2Former sur 10 epochs  

L’objectif final du projet est de comparer la performance et l’efficacité des deux modèles.
""")


st.markdown("---")

# -------------------------------------------------------------------
# SECTION : GRAPHIQUES DES RÉSULTATS
# -------------------------------------------------------------------


client = MlflowClient()

DEEPLAB_RUN_ID = "9d1d6201075647d088840506a93f7a3f"
MASK2F_RUN_ID = "e07ee9aa361c469a929e2db5cfeeb029" 


def get_metric_df(run_id, metric_name):
    history = client.get_metric_history(run_id, metric_name)
    df = pd.DataFrame({
        "step": [m.step for m in history],
        metric_name: [m.value for m in history]
    })
    return df


def plot_metric(run_id, metric, title):
    df = get_metric_df(run_id, metric)
    fig = px.line(df, x="step", y=metric, title=title)
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        xaxis_title="Step",
        yaxis_title=metric,
    )
    return fig


def get_combined_metric(run_id_A, run_id_B, metric_name, name_A="DeepLabV3+", name_B="Mask2Former"):
    # DeepLab
    hist_A = client.get_metric_history(run_id_A, metric_name)
    df_A = pd.DataFrame({
        "step": [m.step for m in hist_A],
        "value": [m.value for m in hist_A],
        "model": name_A
    })

    # Mask2Former
    hist_B = client.get_metric_history(run_id_B, metric_name)
    df_B = pd.DataFrame({
        "step": [m.step for m in hist_B],
        "value": [m.value for m in hist_B],
        "model": name_B
    })

    return pd.concat([df_A, df_B], axis=0)


def plot_comparison(df, metric_name):
    fig = px.line(
        df,
        x="step",
        y="value",
        color="model",
        title=f"Comparaison {metric_name} — DeepLabV3+ vs Mask2Former",
        markers=False
    )

    fig.update_layout(
        template="plotly_white",
        title_font_size=20,
        xaxis_title="Step",
        yaxis_title=metric_name,
        legend_title="Modèle",
        height=500
    )

    fig.update_traces(
        selector=dict(name="DeepLabV3+"),
        line=dict(color="#003566")  
    )

    fig.update_traces(
        selector=dict(name="Mask2Former"),
        line=dict(color="#D62828") 
    )

    return fig

st.markdown("""## Comparaison des modèles sur les métriques clés""")

metric = st.selectbox(
    "Sélectionnez une métrique à comparer :",
    ["train_loss", "val_loss", "miou", "pixel_acc", "imgs_per_sec", "train_time_sec"]
)

df_metric = get_combined_metric(DEEPLAB_RUN_ID, MASK2F_RUN_ID, metric)
fig = plot_comparison(df_metric, metric)


col1, col2 = st.columns([3, 2])  

with col1:
    st.markdown(f"## {metric_info[metric]['title']}")
    st.plotly_chart(fig, use_container_width=True)
        # Description alternative pour lecteurs d’écran
    st.markdown(
        f"""
        <div role="doc-subtitle" aria-label="Résumé textuel du graphique {metric}">
        <p><strong>Description alternative (accessibilité) :</strong></p>
        <p>{metric_info[metric]["alt_text"]}</p>
        </div>
        """, unsafe_allow_html=True
    )

with col2:
    st.markdown(metric_info[metric]["description"])
    st.markdown(metric_info[metric]["comparison"])


st.markdown("---")

# -------------------------------------------------------------------
# SECTION : APPEL API + PRÉDICTIONS
# -------------------------------------------------------------------
st.markdown("""## Tester les modèles sur une nouvelle image""")


import streamlit as st
from PIL import Image

CITYSCAPES_COLORS_BACKEND = {
    "flat":        (128, 64, 128),
    "human":       (244, 35, 232),
    "vehicle":     (70, 70, 70),
    "construction":(102, 102, 156),
    "object":      (190, 153, 153),
    "nature":      (153, 153, 153),
    "sky":         (250, 170, 30),
    "void":        (220, 220, 0),
}

CITYSCAPES_LABELS_FR = {
    "flat": "Surfaces planes (routes, trottoirs)",
    "human": "Personnes (piétons, silhouettes)",
    "vehicle": "Véhicules (voitures, bus, motos…)",
    "construction": "Éléments de construction (bâtiments, murs)",
    "object": "Objets urbains (poteaux, panneaux…)",
    "nature": "Nature (arbres, herbes, végétation)",
    "sky": "Ciel",
    "void": "Régions non pertinentes / inconnues",
}

st.markdown("#### Légende des classes")

cols = st.columns(4)
i = 0
for key, rgb in CITYSCAPES_COLORS_BACKEND.items():
    img = Image.new("RGB", (40, 40), rgb)
    with cols[i % 4]:
        st.image(img, width=40)
        st.markdown(f"**{key.capitalize()}**  \n*{CITYSCAPES_LABELS_FR[key]}*")
    i += 1

st.markdown("---")



uploaded_file = st.file_uploader(
    "Sélectionnez une image (JPEG/PNG)",
    type=["jpg", "png"],
    help="Téléverser une image entre 100×100 et 2000×2000 px. Compatible JPG/PNG. Taille maximale 200 Mo."
)

API_URL = "http://18.234.222.127:8000/predict"
#API_URL = "http://127.0.0.1:8000//predict"


def decode_base64_image(data_url: str) -> Image.Image:
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(data))


# -------------------------------------------------------------------
# ENVOI À L'API
# -------------------------------------------------------------------
if uploaded_file is not None:
    st.markdown("### 📡 Envoi de l'image à l'API…")

    files = {
        "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
    }

    try:
        response = requests.post(API_URL, files=files, timeout=20)

        if response.status_code != 200:
            st.error(
                f"Erreur API : impossible d'obtenir une prédiction. (Code {response.status_code})")

        else:
            results = response.json()
            col1, col2 = st.columns(2)

            # -----------------------------------
            # DeepLab
            # -----------------------------------
            with col1:
                st.markdown("### DeepLabV3+ – Résultat")

                deeplab_img = decode_base64_image(results["deeplab_png"])
                st.image(
                    deeplab_img,
                    use_container_width=True,
                    caption="Prédiction DeepLab – Masque segmenté"
                )

            # -----------------------------------
            # Mask2Former
            # -----------------------------------
            with col2:
                st.markdown("### Mask2Former – Résultat")

                mask2former_img = decode_base64_image(
                    results["mask2former_png"])
                st.image(
                    mask2former_img,
                    use_container_width=True,
                    caption="Prédiction Mask2Former – Masque segmenté"
                )

    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {e}")
