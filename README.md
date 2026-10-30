# Self-Healing RAG

## Project overview

This project is a self-healing Retrieval-Augmented Generation (RAG) system built in Python. It stores local text documents in ChromaDB, retrieves the most relevant content for a user query, generates an answer using a chat model, and then checks whether that answer is actually supported by the retrieved documents before returning it.

The main purpose of the project is to reduce hallucinations in a basic RAG workflow by adding a grading and retry loop. Instead of trusting the first generated answer, the system can rewrite the question and try again when the answer is not grounded in the source documents.

## Functionality

The project works in two stages: ingestion and question answering.

During ingestion, local `.txt` files are loaded, split into chunks, and stored in a persistent Chroma collection along with IDs and metadata.

During question answering, the system:
- Retrieves the top matching chunks from ChromaDB.
- Generates an answer from those retrieved chunks.
- Grades whether the answer is grounded in the documents.
- Returns the answer if it passes.
- Rewrites the question and retries if it fails.
- Falls back to a safe response if retries are exhausted.

## Technology stack

The project uses the following main technologies:

- **Python** for the application logic.
- **LangGraph** for graph-based workflow orchestration.
- **ChromaDB** for persistent local vector storage and retrieval.
- **LangChain integrations** for connecting to the language model.
- **Groq API** as the current LLM provider.
- **python-dotenv** for loading environment variables.

## File details

| File / Folder | Purpose |
|---|---|
| `docs/` | Stores the source text files used as the knowledge base. |
| `ingest.py` | Reads documents, splits them into chunks, and stores them in ChromaDB. |
| `chroma_db/` | Stores the persistent Chroma vector database created after ingestion. |
| `rag_agent.py` | Runs the retrieval, generation, grading, retry, and fallback workflow. |
| `.env` | Stores environment variables such as the Groq API key. |
| `.gitignore` | Prevents secrets and local generated files from being committed to Git. |
| `pyproject.toml` | Defines the Python project and dependencies created by `uv`. |
| `uv.lock` | Locks dependency versions for reproducible installs. |

## Setup steps

### 1. Create the project

```bash
uv init self-healing-rag
cd self-healing-rag
```

### 2. Install dependencies

```bash
uv add langgraph langchain-groq langchain-community langchain-text-splitters chromadb python-dotenv
```

### 3. Add the API key

Create a `.env` file and store the Groq API key:

```bash
echo 'GROQ_API_KEY=your_key_here' > .env
```

### 4. Protect local files

Add local and secret files to `.gitignore`:

```bash
echo ".env" >> .gitignore
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "chroma_db/" >> .gitignore
```

### 5. Add knowledge files

Create a `docs/` folder and place your `.txt` files inside it. These files will act as the knowledge base used for retrieval.

### 6. Run ingestion

```bash
uv run ingest.py
```

This step loads the documents, splits them into chunks, and stores them in the persistent Chroma collection.

### 7. Run the agent

```bash
uv run rag_agent.py
```

This starts the question-answering workflow using retrieval, answer generation, grading, and retry logic.