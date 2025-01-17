# Utilise une image Python 3.11
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie le fichier requirements.txt
COPY requirements.txt /app/

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le contenu du répertoire courant vers /app
COPY . /app/

# Expose le port utilisé par l'API
EXPOSE 8000

# Définit la commande de lancement de l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
