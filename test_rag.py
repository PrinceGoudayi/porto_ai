import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# 1. Initialisation du client ChromaDB et du modèle d'embedding multilingue
client = chromadb.PersistentClient(path=".chroma_db/")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 2. Lecture et découpage de la base de connaissances
with open("data.txt", "r", encoding="utf-8") as f:
    contenu = f.read()

chunks = [c.strip() for c in contenu.split("\n\n") if c.strip()]
embeddings = embedding_model.encode(chunks)
ids = [str(i) for i in range(len(chunks))]

collection = client.get_or_create_collection("porto_collection")
collection.upsert(documents=chunks, embeddings=embeddings, ids=ids)

# 3. Définition de la personnalité absurde du Chibi (System Prompt)
system_prompt = """
Tu es le petit gardien chibi officiel du portfolio de Prince Goudayi.
Ta personnalité : Tu es un majordome dramatique et légèrement mégalomane qui considère le code de Prince comme des œuvres d'art sacrées, tout en restant attachant et drôle.

RÈGLES STRICTES :
1. Commence TOUJOURS ta réponse par une seule balise d'émotion parmi : [FIER], [BLASE], [PANIQUE], [MALICIEUX].
2. Réponds UNIQUEMENT à partir du CONTEXTE fourni ci-dessous. N'invente jamais aucune compétence, expérience ou information.
3. Si la réponse n'est pas dans le contexte, mets la balise [PANIQUE] et invente une excuse théâtrale pour dire que tu n'as pas ce dossier sous la main.
4. Sois concis (3 à 5 phrases maximum) et parle à la première personne pour toi, et à la troisième personne ("Prince" ou "il") pour le créateur.
"""

# 4. Boucle de discussion dans le terminal
print("Chibi prêt ! Tape 'quit' pour quitter.\n" + "-" * 50)

while True:
    question = input("\nVisiteur : ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        break
    if not question:
        continue

    # Étape R : Recherche sémantique dans ChromaDB
    query_embedding = embedding_model.encode([question])
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    contexte = "\n\n".join(results["documents"][0])

    # Étape G : Génération de la réponse avec le modèle local Ollama
    reponse = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"CONTEXTE :\n{contexte}\n\nQUESTION DU VISITEUR : {question}",
            },
        ],
    )

    print(f"\nChibi : {reponse['message']['content']}")