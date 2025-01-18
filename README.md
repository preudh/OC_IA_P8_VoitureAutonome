# **Déploiement d'une API FastAPI et d'une Application Streamlit pour la Segmentation d'Images**

## **Mission**

Future Vision Transport est une entreprise spécialisée dans les systèmes embarqués de vision par ordinateur pour les
véhicules autonomes. En tant qu’ingénieur IA au sein de l’équipe R&D, votre mission consiste à :

1. **Concevoir un modèle de segmentation d’images** :  
Entraîner un modèle performant sur les **8 catégories principales** du dataset **Cityscapes**, tout en respectant les 
contraintes d’intégration dans un système embarqué.
   
2. **Développer une API de prédiction** :  
Utiliser **FastAPI** pour créer une API qui prend une image en entrée et renvoie un masque de segmentation en sortie.

3. **Créer une application Web** :  
Développer une interface avec **Streamlit** pour tester l’API, visualiser les masques prédits, et les télécharger.

4. **Déployer l’ensemble sur Azure** :  
Déployer l’API et l’application sur **Azure Container Instances**, tout en garantissant la communication entre les deux services.

---

## **Approche Technique**

### **1. Modélisation**
- **Jeu de données** :  
  [Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) :  
  Images segmentées et annotées provenant de caméras embarquées.  
  Utilisation restreinte aux **8 catégories principales** (au lieu des 32 sous-catégories).

- Framework utilisé : **PyTorch** (au lieu de Keras, pour une meilleure flexibilité et compatibilité avec le projet).

- Modèles comparés :
  - **U-Net Mini**
  - **VGG16 U-Net** (modèle sélectionné)
  - **DilatedNet**
- **Critères de comparaison** : validation loss, IoU, Dice coefficient, et temps d’entraînement.
  
#### **Étude comparative des modèles :**
| **Modèle**       | **Loss Type**                 | **Validation Loss** | **IoU**  | **Dice** | **Training Time (min)** |
|-------------------|-------------------------------|----------------------|----------|----------|--------------------------|
| **U-Net Mini**    | categorical_crossentropy      | 0.462157            | 0.588130 | 0.696657 | 79.14                   |
| **VGG16 U-Net**   | combine_loss                 | 0.411980            | 0.607362 | 0.713207 | 153.59                  |
| **DilatedNet**    | categorical_crossentropy      | 0.799034            | 0.390745 | 0.502706 | 138.12                  |

- **Optimisation du modèle** :
  - **Data augmentation** : rotation, translation, flip horizontal.
  - **Early stopping** pour éviter le sur-apprentissage.

---

### **2. Développement de l’API (FastAPI)**
- Permet d’envoyer une image et de recevoir un masque de segmentation.
- Fonctionnalités clés :
  - Conversion des prédictions en masque PNG.
  - Streaming du fichier prédictif directement en mémoire.
  - Endpoint interactif disponible sur `/docs`.

---

### **3. Développement de l’application Web (Streamlit)**
- Permet de :
  - Charger une image.
  - Envoyer l’image à l’API.
  - Afficher l’image originale et le masque de segmentation.
  - Télécharger le masque au format PNG.
- Variable d’environnement utilisée : `API_URL` pour pointer vers l’URL de l’API FastAPI déployée.

---

### **4. Déploiement sur Azure**
#### **API (FastAPI)**
Commandes utilisées :
```bash
az container create --name fastapi-container --resource-group P9ResourceGroup --image p8containerregistry.azurecr.io/api:latest --cpu 2 --memory 4 --registry-login-server p8containerregistry.azurecr.io --registry-username <USERNAME> --registry-password <PASSWORD> --ports 8000 --dns-name-label fastapi-p8 --environment-variables PYTHONUNBUFFERED=1
```

#### **Application Web (Streamlit)**
Commandes utilisées :
```bash
az container create --name streamlit-container --resource-group P9ResourceGroup --image p8containerregistry.azurecr.io/streamlit:latest --cpu 2 --memory 4 --registry-login-server p8containerregistry.azurecr.io --registry-username <USERNAME> --registry-password <PASSWORD> --ports 8501 --dns-name-label streamlit-p8 --environment-variables API_URL=http://fastapi-p8.westeurope.azurecontainer.io:8000/predict
```

---

## **Accès et Utilisation**

### **Services déployés :**
- **API FastAPI** : [http://fastapi-p8.westeurope.azurecontainer.io:8000/docs](http://fastapi-p8.westeurope.azurecontainer.io:8000/docs)
- **Application Streamlit** : [http://streamlit-p8.westeurope.azurecontainer.io:8501](http://streamlit-p8.westeurope.azurecontainer.io:8501)

### **En local :**
- Lancer les conteneurs :
  ```bash
  docker-compose up
  ```
- Arrêter les conteneurs :
  ```bash
  docker-compose down
  ```
Conclusions et Perspectives :
- **Résultats** :  
  Le modèle VGG16 U-Net a été retenu pour sa meilleure performance en termes de métriques et de temps d’entraînement.
- D'autres modèles peuvent être testés pour améliorer les performances comme des modèles plus récents (DeepLab, PSPNet, etc.).