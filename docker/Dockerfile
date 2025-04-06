# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt
COPY ./docker/requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application
COPY ./flask ./flask
COPY ./model ./model

# Exposer le port sur lequel l'application va tourner
EXPOSE 5000

# Commande pour lancer l'application
CMD ["python", "./flask/app.py"]
