## Utilise une image Python 3.11 comme base
#FROM python:3.11-slim
#
## Définit le répertoire de travail
#WORKDIR /app
#
## Installe les dépendances système nécessaires
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libgl1 \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*
#
## Copie uniquement le fichier requirements.txt pour optimiser le cache Docker
#COPY requirements.txt /app/
#
## Installe les dépendances Python
#RUN pip install --no-cache-dir -r requirements.txt
#
## Copie tout le contenu du répertoire courant vers /app
#COPY . /app/
#
## Expose le port utilisé par l'API
#EXPOSE 8000
#
## Définit la commande de lancement de l'API
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Utilise une image Python 3.11 comme base
# Utilise une image Python 3.11 comme base
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie uniquement le fichier requirements.txt pour optimiser le cache Docker
COPY requirements.txt /app/

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt


# Copie tout le contenu du répertoire courant vers /app
COPY . /app/

# Expose le port utilisé par l'API
EXPOSE 8000

# Définit la commande de lancement de l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

