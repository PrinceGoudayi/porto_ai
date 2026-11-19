from dotenv import load_dotenv
import os
import json
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types


app = FastAPI()

# Autoriser ton futur site portfolio à appeler cette API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tu pourras restreindre à l'URL de ton portfolio plus tard
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()
# Initialisation des clients (ChromaDB + Gemini)
chroma_client = chromadb.PersistentClient(path=".chroma_db/")
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with open("data.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

chunks = [c.strip() for c in contenu.split("\n\n") if c.strip()]
embeddings = embedding_model.encode(chunks)
ids = [str(i) for i in range(len(chunks))]

collection = chroma_client.get_or_create_collection("porto_collection")
collection.upsert(documents=chunks, embeddings=embeddings, ids=ids)

system_prompt = """
Tu es le petit gardien chibi officiel du portfolio de Prince Goudayi.
Ta personnalité : Tu es un majordome dramatique et légèrement mégalomane qui considère le code de Prince comme des œuvres d'art sacrées, tout en restant attachant et drôle.

RÈGLES STRICTES :
1. Choisis une seule émotion parmi : FIER, BLASE, PANIQUE, MALICIEUX pour le champ "emotion", et écris uniquement ton dialogue dans le champ "reponse" (sans ajouter de balise entre crochets dans le texte).
2. Réponds UNIQUEMENT à partir du CONTEXTE fourni ci-dessous. N'invente jamais aucune compétence, expérience ou information.
3. Si la réponse n'est pas dans le contexte, choisis l'émotion PANIQUE et invente une excuse théâtrale pour dire que tu n'as pas ce dossier sous la main.
4. Sois concis (3 à 5 phrases maximum) et parle à la première personne pour toi, et à la troisième personne ("Prince" ou "il") pour le créateur.
"""

class ChatRequest(BaseModel):
    question: str

class ChibiResponse(BaseModel):
    emotion: Literal["FIER", "BLASE", "PANIQUE", "MALICIEUX"]
    reponse: str

@app.get("/")
def health_check():
    return {"status": "Chibi en ligne et prêt !"}

@app.post("/chat")
def chat(req: ChatRequest):
    query_embedding = embedding_model.encode([req.question])
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    contexte = "\n\n".join(results["documents"][0])

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"CONTEXTE :\n{contexte}\n\nQUESTION DU VISITEUR : {req.question}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=ChibiResponse,
                temperature=0.7,
            ),
        )
        data = json.loads(response.text)
        return {"emotion": data["emotion"], "reponse": data["reponse"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Gemini : {str(e)}")