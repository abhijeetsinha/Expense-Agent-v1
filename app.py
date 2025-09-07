import os
import sys
import pandas as pd
import uuid
from dotenv import load_dotenv
from typing import List, Dict

# Azure OpenAI imports
from openai import AzureOpenAI

# Chroma
import chromadb
from chromadb.config import Settings



# Load environment variables from .env file
load_dotenv()

AzureOpenAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AzureOpenAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AzureOpenAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AzureOpenAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
AzureOpenAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL") 

print("AZURE_OPENAI_ENDPOINT:", AzureOpenAI_ENDPOINT)
print("AZURE_OPENAI_API_KEY:", AzureOpenAI_API_KEY)
print("AZURE_OPENAI_API_VERSION:", AzureOpenAI_API_VERSION)
print("AZURE_OPENAI_CHAT_MODEL:", AzureOpenAI_CHAT_MODEL)
print("AZURE_OPENAI_EMBEDDING_MODEL:", AzureOpenAI_EMBEDDING_MODEL)

if not all([AzureOpenAI_ENDPOINT, AzureOpenAI_API_KEY, AzureOpenAI_API_VERSION, AzureOpenAI_CHAT_MODEL, AzureOpenAI_EMBEDDING_MODEL]):
    print("One or more environment variables are missing. Please check your .env file.")
    sys.exit(1)



# Initialize Azure OpenAI client
azureopenai_client = AzureOpenAI(
    api_key = AzureOpenAI_API_KEY,
    api_version = AzureOpenAI_API_VERSION,
    azure_endpoint = AzureOpenAI_ENDPOINT
)



# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path = "./chroma_db", settings = Settings(allow_reset = False))
COLLECTION_NAME = "expenses_collection"

def get_or_create_collection():
    try:
        return chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        return chroma_client.create_collection(COLLECTION_NAME)
    


# Utilities
def read_expenses(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    print(f"Reading file: {file_path} with extension: {ext}")
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        df =  pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    df.fillna("")
    if "id" not in df.columns:
        df.insert(0, "id", [str(uuid.uuid4()) for _ in range(len(df))])
    
    df["id"] = df ["id"].astype(str)
    return df



def row_to_text (row: pd.Series) -> str:
    parts = []    
    for col in row.index:
        if col == "id":
            continue
        val = str(row[col]).strip()
        if val != "":
            parts.append(f"{col}: {val}")
    return "\n".join(parts)



def embed_text(texts : List[str]) -> List[List[float]]:
    response = azureopenai_client.embeddings.create(model=AzureOpenAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]



# ingest data
def ingest_expenses(file_path: str):
    df= read_expenses(file_path)
    print(f"Dataframe shape: {df.shape}")
    docs = []
    metadatas = []
    ids = []

    
    for _, row in df.iterrows():
        ids.append(str(row["id"]))
        docs.append(row_to_text(row))  # what we embed & store for retrieval
        # Keep original fields as metadata for display/filters later (optional)
        meta = {k: (str(v) if pd.notna(v) else "") for k, v in row.to_dict().items()}
        metadatas.append(meta)

    print("docs:", docs)
    print("ids:", ids)
    print("metadatas:", metadatas)

    batch_size = 128
    vectors: List[List[float]] = []
    for i in range(0, len(docs), batch_size):
        batch_texts = docs[i:i + batch_size]
        batch_vectors = embed_text(batch_texts)
        vectors.extend(batch_vectors)
        print(f"Vectors: {vectors}")
        #print(f"Processed batch {i // batch_size + 1}, total vectors: {len(vectors)}")
    print(f"Generated {len(vectors)} embeddings.")

    collections = get_or_create_collection()
    collections.add(
        documents = docs,
        metadatas = metadatas,
        ids = ids,
        embeddings = vectors
    )
    print(f"Inserted {len(docs)} records into ChromaDB collection '{COLLECTION_NAME}'.")


# Ask questions

def retrieve(query: str, k: int = 5):
    q_vec = embed_text([query])[0]
    col = get_or_create_collection()
    res = col.query(query_embeddings=[q_vec], n_results=k)  # similarity search
    print("Similarity search result.")
    print("Raw query result:", res)
    # res contains "documents", "metadatas", "ids", "distances"
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })

    print(f"Retrieved {len(hits)} hits for query: '{query}'")
    print("Hits:", hits)
    return hits





def answer_with_context(question: str, hits: List[Dict]) -> str:
    # Build a tiny prompt with the retrieved rows
    snippets = []
    for h in hits:
        snippets.append(f"- ID {h['id']}\n{h['document']}")
    context = "\n\n".join(snippets) if snippets else "No matching expenses found."
    print("\n=== Context ===")
    print(context)
    print("=== End Context ===\n")

    system = (
        "You are a helpful assistant that answers questions about the user's expenses. "
        "Use ONLY the provided context to answer. If the answer isn't in the context, say you don't know."
    )

    print("\n=== System ===")
    print(system)
    print("=== End System ===\n")

    user = f"Question: {question}\n\nContext:\n{context}"
    print("\n=== Prompt ===")
    print(user)
    print("=== End Prompt ===\n")   



    resp = azureopenai_client.chat.completions.create(
        model=AzureOpenAI_CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0,
    )
    return resp.choices[0].message.content




def ask(question: str):
    hits = retrieve(question, k=5)
    ans = answer_with_context(question, hits)
    print("\n=== Answer ===")
    print(ans)
    print("\n=== Top matches (debug) ===")
    for h in hits:
        md = h["metadata"]
        basic = f"{md.get('date','?')} • {md.get('merchant','?')} • {md.get('category','?')} • {md.get('amount','?')}"
        print(f"{h['id']}: {basic}  (distance={h['distance']:.4f})")


# main chat function
def main():

    if len(sys.argv) < 2:
        print("Usage:\n  python app.py ingest <path_to_csv_or_xlsx>\n  python app.py ask \"your question\"")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("Please provide a file path, e.g. data/expenses.csv")
            sys.exit(1)
        ingest_expenses(sys.argv[2])
    elif cmd == "ask":
        question = " ".join(sys.argv[2:])
        ask(question)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
