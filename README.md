# YouTube Chatbot using LangChain

Building a Retrieval-Augmented Generation (RAG) System in LangChain

This repository contains a complete implementation of a YouTube Chatbot built using LangChain and the RAG (Retrieval-Augmented Generation) approach. The chatbot can answer questions based on the content of YouTube videos by extracting transcripts, generating embeddings, storing them in a vector database, and using an LLM to produce grounded responses.

The project is implemented entirely inside a single Jupyter Notebook.

---

## Overview

Large Language Models do not have direct access to specific external knowledge, such as the content of a YouTube video. RAG solves this problem by retrieving relevant information from an external source before generating an answer.
This project demonstrates how to build such a system using:

* LangChain for pipeline orchestration
* YouTube transcript loader
* Embedding models for vectorization
* FAISS or similar vector store for similarity search
* RetrievalQA for grounded response generation

---

## Included File

* **rag_using_langchain.ipynb**
  This notebook contains the full workflow to build the YouTube RAG chatbot from start to end.

---

## Features

* Fetches YouTube video transcripts automatically
* Cleans and preprocesses text for better retrieval
* Splits transcripts into chunks suitable for embedding
* Generates embeddings using a sentence-transformer model
* Stores embeddings in FAISS vector store
* Uses LangChain RetrievalQA for context-aware responses
* Provides an interactive query interface inside the notebook

---

## Workflow

1. **Load YouTube Transcript**
   The notebook extracts transcript text using LangChain’s YouTubeLoader.

2. **Text Preprocessing and Chunking**
   Transcript is cleaned and split into manageable chunks for embedding.

3. **Embedding Generation**
   Each chunk is converted into vector embeddings using a transformer model.

4. **Vector Store Creation**
   FAISS or another vector database is used for similarity search.

5. **Retriever Setup**
   The retriever finds the most relevant transcript parts for a user query.

6. **RAG Pipeline using RetrievalQA**
   LangChain combines retrieved context with an LLM to produce grounded answers.

7. **Chat Interface**
   Users ask questions and receive answers based only on the video content.

---

## Installation

Install required packages:

```
pip install langchain langchain-community langchain-core faiss-cpu
pip install sentence-transformers
pip install python-dotenv
pip install pytube
```

If you are using OpenAI or another model:

```
pip install openai
```

---

## How to Run

1. Clone or download this repository
2. Install the dependencies
3. Open the notebook:

```
jupyter notebook rag_using_langchain.ipynb
```

4. Add your API key inside `.env` or directly in the notebook
5. Input a YouTube video URL
6. Run all cells to build and test the chatbot

---

## Use Cases

* Summarizing long YouTube videos
* Creating educational chatbots based on courses or lectures
* Customer support over tutorial videos
* FAQ bots built from video content
* Research assistants for documentary or interview videos

---

## Limitations

* Accuracy depends on transcript quality
* Videos without transcripts cannot be processed
* Embedding quality impacts retrieval performance
* Requires an API key for LLM responses

---

## Future Improvements

* Support for multiple videos
* Storing transcripts persistently
* Adding a frontend UI
* Using a hosted vector database such as Pinecone or Weaviate
* Adding summarization layers


