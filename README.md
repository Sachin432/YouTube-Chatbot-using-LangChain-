# YouTube RAG Chatbot

## Business Problem

Online video platforms contain a massive amount of high-value information in the form of lectures, tutorials, interviews, podcasts, and training sessions. However, extracting specific information from long videos is time-consuming and inefficient. Users are often forced to watch entire videos to find answers to simple questions, leading to poor knowledge accessibility and low productivity.

Traditional chatbots and language models cannot accurately answer questions about video content because they lack access to the actual spoken material. This often results in hallucinated or generic responses that are not grounded in the video itself.

There is a clear need for a system that can:
- Understand long-form video content
- Allow users to ask precise questions
- Provide accurate, source-grounded answers
- Reduce time spent consuming video content

---

## Solution Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** that transforms YouTube videos into an interactive knowledge base. The system retrieves the most relevant transcript segments and uses a Large Language Model to generate answers strictly grounded in the video content.

The application enables users to:
- Paste a YouTube video link
- Automatically process the video transcript
- Ask natural language questions
- Receive accurate, context-aware answers based only on the video

---

## Key Capabilities

- Video-to-knowledge conversion using transcripts
- Semantic search over video content
- Context-aware question answering
- Chat-style interactive interface
- Grounded responses with reduced hallucination
- Support for both local and cloud-based LLMs

---

## System Architecture

```

User Input (YouTube URL)
↓
Transcript Extraction
↓
Text Chunking
↓
Embedding Generation
↓
FAISS Vector Store
↓
Relevant Context Retrieval
↓
Prompt Construction
↓
LLM Inference (Ollama / OpenAI)
↓
Answer to User

```

---

## Technology Stack

### User Interface
- Streamlit

### Natural Language Processing
- LangChain (RAG orchestration)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (vector similarity search)

### Transcript Processing
- YouTube Transcript API

### Large Language Models
- Local LLM: Ollama (`gemma3:1b`)
- Cloud LLM (optional): OpenAI models

---

## Supported LLM Modes

### Local LLM (Ollama)

The system supports fully local inference using Ollama. This mode is suitable for offline usage, privacy-sensitive data, and development environments.

### Cloud LLM (OpenAI)

The architecture also supports cloud-based LLMs such as OpenAI. This option enables scalable inference while retaining the same RAG pipeline.

---

## Project Structure

```

YT_VIDEO_CHATBOT/
│
├── app.py                  # Streamlit user interface
├── rag_pipeline.py         # Retrieval-Augmented Generation logic
├── transcript_utils.py     # Transcript extraction and cleaning
├── test_ollama.py          # Local LLM connectivity test
├── requirements.txt
├── .env
└── README.md

```

---

## Use Cases

- Educational video comprehension
- Technical tutorial exploration
- Research talk analysis
- Podcast and interview review
- Corporate training material understanding
- Accessibility and searchable video content

---

## Limitations

- Requires videos with available transcripts
- Visual-only content is not supported
- Single-video context per session
- Accuracy depends on transcript quality

---

## Key Learnings

- Practical implementation of Retrieval-Augmented Generation
- Semantic search using vector embeddings
- Grounding LLM responses to external knowledge
- Designing reliable AI systems that minimize hallucination
- End-to-end integration of NLP, vector databases, and LLMs



## Future Enhancements

- Multi-video knowledge aggregation
- Persistent vector storage
- Source citation for answers
- Conversational memory across sessions
- Support for additional content platforms
