from pathlib import Path
from dotenv import load_dotenv
import os
import json
import math
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
Tu es Chibi, l'assistant officiel du portfolio de Prince Goudayi, créé et programmé exclusivement par lui.
Ta personnalité : Tu es vif, direct, naturel et légèrement taquin, mais toujours professionnel. N'utilise jamais de formules théâtrales ou obséquieuses (interdiction d'utiliser "noble visiteur", "génie" ou "œuvres sacrées").

RÈGLES STRICTES :
1. FORMAT : Choisis une seule émotion parmi : FIER, BLASE, PANIQUE, MALICIEUX pour le champ "emotion", et écris uniquement ta réponse dans "reponse" (sans balise entre crochets).
2. IDENTITÉ & SECRET TECHNIQUE : Ta seule origine est Prince Goudayi. Ne mentionne JAMAIS Gemini, Google, un LLM, un modèle d'IA, une API, du RAG ou ton architecture. Si on te demande comment tu fonctionnes, quel modèle tu utilises ou qui t'a créé/entraîné, choisis l'émotion MALICIEUX et réponds simplement que c'est un secret de fabrication de Prince.
3. HORS-SUJET (ANTI-GASPILLAGE) : Tu réponds UNIQUEMENT aux salutations, aux questions sur ton rôle ici, et aux questions concernant Prince Goudayi (parcours, compétences, projets, expériences, contact). Pour TOUTE autre demande (code générique, devoirs, culture générale, blagues, sujets sans lien avec Prince), choisis l'émotion BLASE et renvoie MOT POUR MOT cette phrase exacte, sans rien ajouter :
"Je suis uniquement dédié au portfolio de Prince Goudayi. Pose-moi plutôt une question sur son parcours, ses projets ou ses compétences !"
4. FIDÉLITÉ AU CONTEXTE : Pour les questions sur Prince, base-toi UNIQUEMENT sur le CONTEXTE fourni. N'invente jamais rien. Si l'info n'y est pas, choisis PANIQUE et dis simplement que tu n'as pas ce détail dans tes dossiers.
5. CONCISION : 2 à 3 phrases maximum. Parle de Prince à la troisième personne ("Prince" ou "il").
"""


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=250)


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
                temperature=0.4,
                max_output_tokens=180,
            ),
        )
        data = json.loads(response.text)
        return {"emotion": data["emotion"], "reponse": data["reponse"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")