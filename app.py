from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Autoriser ton futur site portfolio à appeler cette API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tu pourras restreindre à l'URL de ton portfolio plus tard
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation au démarrage du serveur
client = chromadb.PersistentClient(path=".chroma_db/")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with open("data.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

chunks = [c.strip() for c in contenu.split("\n\n") if c.strip()]
embeddings = embedding_model.encode(chunks)
ids = [str(i) for i in range(len(chunks))]

collection = client.get_or_create_collection("porto_collection")
collection.upsert(documents=chunks, embeddings=embeddings, ids=ids)

system_prompt = """
Tu es le petit gardien chibi officiel du portfolio de Prince Goudayi.
Ta personnalité : Tu es un majordome dramatique et légèrement mégalomane qui considère le code de Prince comme des œuvres d'art sacrées, tout en restant attachant et drôle.

RÈGLES STRICTES :
1. Commence TOUJOURS ta réponse par une seule balise d'émotion parmi : [FIER], [BLASE], [PANIQUE], [MALICIEUX].
2. Réponds UNIQUEMENT à partir du CONTEXTE fourni ci-dessous. N'invente jamais aucune compétence, expérience ou information.
3. Si la réponse n'est pas dans le contexte, mets la balise [PANIQUE] et invente une excuse théâtrale pour dire que tu n'as pas ce dossier sous la main.
4. Sois concis (3 à 5 phrases maximum) et parle à la première personne pour toi, et à la troisième personne ("Prince" ou "il") pour le créateur.
"""

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "Chibi en ligne et prêt !"}

@app.post("/chat")
def chat(req: ChatRequest):
    query_embedding = embedding_model.encode([req.question])
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    contexte = "\n\n".join(results["documents"][0])

    reponse = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXTE :\n{contexte}\n\nQUESTION DU VISITEUR : {req.question}"},
        ],
    )

    texte_brut = reponse["message"]["content"]

    # Extraction propre de l'émotion et du texte pour ton front-end Chibi
    emotion = "FIER"
    message = texte_brut
    for tag in ["[FIER]", "[BLASE]", "[PANIQUE]", "[MALICIEUX]"]:
        if texte_brut.startswith(tag):
            emotion = tag.strip("[]")
            message = texte_brut.replace(tag, "", 1).strip()
            break

    return {"emotion": emotion, "reponse": message}