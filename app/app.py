import streamlit as st
import requests
from PIL import Image
import io

# URL de l'API
API_URL = "http://api:8000/predict"

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Application de Segmentation - Véhicules Autonomes",
    layout="wide"
)

st.title("🚗 Application de Segmentation pour Véhicules Autonomes")
st.write("Cette application utilise des modèles d'intelligence artificielle pour segmenter des images en **8 catégories principales** :")

st.markdown("""
- **void** : Éléments d'arrière-plan (0, 1, 2, 3, 4, 5, 6)
- **flat** : Routes, trottoirs, etc. (7, 8, 9, 10)
- **construction** : Bâtiments, clôtures, murs (11, 12, 13, 14, 15, 16)
- **object** : Poteaux, panneaux de signalisation, feux de circulation (17, 18, 19, 20)
- **nature** : Végétation et terrain (21, 22)
- **sky** : Ciel (23)
- **human** : Piétons, cyclistes (24, 25)
- **vehicle** : Voitures, camions, bus, motos (26 à 33, -1)
""")

# Interface de chargement de l'image
uploaded_file = st.file_uploader("Déposez votre image ici ou cliquez pour télécharger", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image d'entrée", use_column_width=True)

    # Soumission de l'image
    submit_button = st.button("Soumettre")

    if submit_button:
        # Envoi de l'image à l'API
        st.write("⏳ Envoi de l'image à l'API...")
        with st.spinner("Traitement de l'image en cours..."):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            response = requests.post(API_URL, files={"file": img_bytes})

        # Gestion de la réponse
        if response.status_code == 200:
            st.success("✅ Prédiction terminée avec succès !")

            # Sauvegarder le masque téléchargé
            with open("downloaded_mask.png", "wb") as f:
                f.write(response.content)

            # Charger le masque sauvegardé en tant qu'image
            predicted_mask = Image.open("downloaded_mask.png")

            # Affichage des résultats
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Image d'entrée", use_column_width=True)

            with col2:
                st.image(predicted_mask, caption="Masque Prédit (Téléchargé)", use_column_width=True)

            # Lien pour télécharger le masque
            st.download_button(
                label="Télécharger le masque",
                data=response.content,
                file_name="predicted_mask.png",
                mime="image/png"
            )

        else:
            st.error(f"❌ Une erreur s'est produite : {response.text}")

# # Section des exemples
# st.sidebar.header("📂 Exemples")
# examples = ["example1.jpg", "example2.jpg", "example3.jpg"]  # Liste des exemples
# for example in examples:
#     if st.sidebar.button(f"Charger {example}"):
#         st.image(f"./examples/{example}", caption="Image Exemple")


