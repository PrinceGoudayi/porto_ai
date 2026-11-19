from pathlib import Path
from dotenv import load_dotenv
import os
import json
import math
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open("data.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

chunks = [c.strip() for c in contenu.split("\n\n") if c.strip()]

def get_embeddings(texts: list[str]) -> list[list[float]]:
    res = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    return [e.values for e in res.embeddings]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

chunk_embeddings = get_embeddings(chunks)

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
    try:
        query_emb = get_embeddings([req.question])[0]
        scored = sorted(
            zip(chunks, chunk_embeddings),
            key=lambda item: cosine_similarity(query_emb, item[1]),
            reverse=True,
        )
        contexte = "\n\n".join(chunk for chunk, _ in scored[:2])

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