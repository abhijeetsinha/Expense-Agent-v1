# Expense-Agent-v1

# Conceptual picture (plain English)
[Your expense files] --> [Embed each row with Azure OpenAI] --> [Chroma DB on disk]
                                                           |
User question --> [Embed question] --> [nearest rows from Chroma] --> [LLM writes answer]


Embeddings = numeric “GPS coordinates of meaning” for each row.
Vector DB (Chroma) = quickly finds the closest rows (by meaning).
LLM call = turns those rows into a natural‑language answer.
1) Prerequisites (one‑time)


## Create an Azure OpenAI resource in your Azure subscription, and deploy:

An embeddings model: text-embedding-3-small (cheap & good for starters).
A chat model: gpt-4o-mini (or any chat model you have deployed).
(How‑to for embeddings on Azure OpenAI: models and API details.) [Azure Open...Azure ...], [azure-sdk-...t main ...]


## Install tools
Shell# Create & activate a virtual env (macOS/Linux shown)python -m venv .venvsource .venv/bin/activate# Windows PowerShell# .venv\Scripts\Activate.ps1# Minimal packagespip install pandas python-dotenv openai==1.* chromadbShow more lines


## Prepare a .env file (in your project folder):
INI# Azure OpenAI (use your actual endpoint/key + your deployment names)AZURE_OPENAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com/AZURE_OPENAI_API_KEY=<your-aoai-key>AZURE_OPENAI_API_VERSION=2024-05-01-previewAZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-smallAZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-miniShow more lines



## Why these libraries?

openai v1+ includes the AzureOpenAI client used for embeddings and chat completions with Azure endpoints. (See the Azure client usage and embeddings in the official docs.) [Learn how...ure OpenAI], [Azure Open...Azure ...]
chromadb is a lightweight vector DB that persists to a folder—perfect for a first project. [AzureOpenA...LangChain]


# Project layout (copy/paste this tree)
expense-agent/
├─ .env
├─ data/
│  └─ expenses.csv            # or expenses.xlsx
├─ chroma/                    # Chroma will store vectors here (auto-created)
├─ app.py                     # main CLI: ingest & ask
├─ requirements.txt
└─ README.md

## requirements.txt
Plain Textpandaspython-dotenvopenai==1.*chromadb
