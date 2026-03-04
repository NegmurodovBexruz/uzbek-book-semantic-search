# Uzbek Book Semantic Search

This project implements a semantic search system for Uzbek-language books stored in `.docx` format.
Instead of traditional keyword search, it uses vector embeddings to retrieve passages that are semantically related to a user’s question.

The system is designed to work with a local embedding model and can be connected to a Telegram bot for public use.

## Features

* Semantic search over multiple `.docx` books
* Automatic extraction of headings from documents
* Text chunking for better retrieval accuracy
* Vector storage using ChromaDB
* Local embeddings via Ollama
* Telegram bot interface for querying the knowledge base

## Project Structure

```
Semantic_Search/
│
├── src/
│   ├── bot.py
│   ├── ingest_docx.py
│   ├── search.py
│   ├── search_engine.py
│   └── utils.py
│
├── books/        # place .docx books here
├── db/           # vector database (created after indexing)
│
├── requirements.txt
└── README.md
```

## Install Ollama and Embedding Model

This project uses a local embedding model through **Ollama**.

### Install Ollama

Download and install Ollama from the official website:

https://ollama.com/download

After installation, verify it works:

```
ollama --version
```

### Download the embedding model

Pull the embedding model used by this project:

```
ollama pull bge-m3
```

The first download may take some time depending on your internet connection.

### Verify the model

Run:

```
ollama list
```

You should see:

```
bge-m3
```

Once the model is installed, the indexing script will automatically use it when generating embeddings.


## Installation

Clone the repository:

```
git clone https://github.com/NegmurodovBexruz/uzbek-book-semantic-search.git
cd uzbek-book-semantic-search
```

Create a virtual environment:

```
python -m venv venv
```

Activate it:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

## Setup

Before running the project create two folders in the project root:

```
books
db
```

Place your `.docx` books inside the `books/` folder.

## Build the Vector Database

Run the indexing script:

```
python src/ingest_docx.py
```

This step reads the documents, splits them into chunks and stores their embeddings in ChromaDB.

## Local Search

To test the search from the terminal:

```
python src/search.py
```

Example query:

```
me'roj nima
```

The system will return the book title, section heading and the most relevant passage.

## Telegram Bot

Create a bot using BotFather and set your token as an environment variable:

```
BOT_TOKEN=your_token_here
```

Run the bot:

```
python src/bot.py
```

Users can then send questions directly to the bot.

## Notes

* The quality of search depends on the structure of the source documents.
* Headings are detected from the DOCX styles.
* Large collections of books may require rebuilding the vector database.

## License

MIT License
