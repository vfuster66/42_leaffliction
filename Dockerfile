FROM python:3.10

# Installer les dépendances système nécessaires pour TensorFlow, OpenCV et autres
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /app

# Copier les dépendances Python
COPY requirements.txt .

# Installer les packages Python (TensorFlow inclus)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Commande par défaut
CMD ["bash"]
