FROM python:3.11-slim

# Installation de curl et d'Ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de ton code et de tes données
COPY app.py data.txt start.sh ./
RUN chmod +x start.sh

# Hugging Face Spaces utilise le port 7860
EXPOSE 7860

CMD ["./start.sh"]