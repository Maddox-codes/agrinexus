#  AgriNexus

> **Vision-Integrated Stateful Agentic Triage System for Sustainable Crop Disease Management**

AgriNexus is an AI-powered agricultural decision support system that combines computer vision, Retrieval-Augmented Generation (RAG), and stateful AI workflows to generate personalized crop disease management recommendations.

Unlike conventional plant disease detection applications that immediately recommend generic pesticides after image classification, AgriNexus introduces a **Deterministic State Machine** that pauses the diagnostic workflow to collect the farmer's budget constraints and recent pesticide usage history before generating recommendations. This additional context enables safer, more practical, and environmentally responsible treatment plans.

---

## Features

*  Multi-crop disease diagnosis using fine-tuned ResNet-18 models
*  Stateful workflow that collects farmer constraints before reasoning
*  Retrieval-Augmented Generation (RAG) using official agricultural manuals
*  Metadata-filtered semantic search with Pinecone
*  Context-aware recommendations powered by Cohere Command R+
*  Organic-first Integrated Pest Management (IPM) recommendations
*  Multi-turn conversational assistant for follow-up agricultural guidance

---

# System Architecture

```text
Leaf Image
     │
     ▼
┌──────────────────────────────┐
│  Crop-Specific ResNet-18     │
│  Disease Classification      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Deterministic State Machine  │
│ Budget + Spray History       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Pinecone Vector Database     │
│ Metadata Filtered Retrieval  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Cohere Command R+            │
│ Context-Aware IPM Generator  │
└──────────────┬───────────────┘
               │
               ▼
     Personalized 7-Day
      Disease Management
```

---

# Technology Stack

| Component            | Technology                           |
| -------------------- | ------------------------------------ |
| Frontend             | Streamlit                            |
| Deep Learning        | PyTorch, ResNet-18                   |
| Dataset              | PlantVillage                         |
| Vector Database      | Pinecone Serverless                  |
| Embedding Model      | Cohere `embed-english-v3.0`          |
| Large Language Model | Cohere `command-r-plus-08-2024`      |
| Retrieval            | Retrieval-Augmented Generation (RAG) |
| Document Processing  | PyPDF2                               |

---

# Repository Structure

```text
agrinexus/
│
├── app.py
├── train_resnet.py
├── ingest_bulk.py
├── requirements.txt
├── .env
├── models/
│   ├── agrinexus_tomato_resnet18.pth
│   ├── agrinexus_potato_resnet18.pth
│   └── agrinexus_pepper_resnet18.pth
└── README.md
```

| File              | Description                            |
| ----------------- | -------------------------------------- |
| `app.py`          | Main Streamlit application             |
| `train_resnet.py` | Model training pipeline                |
| `ingest_bulk.py`  | Document ingestion and vector indexing |
| `models/`         | Trained ResNet-18 model weights        |

---

# Installation

## Clone the repository

```bash
git clone https://github.com/Maddox-codes/agrinexus.git
cd agrinexus
```

---

## Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configure environment variables

Create a `.env` file in the project root.

```env
COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## Add trained model weights

Place the trained model files inside the `models/` directory.

```text
models/
├── agrinexus_tomato_resnet18.pth
├── agrinexus_potato_resnet18.pth
└── agrinexus_pepper_resnet18.pth
```

> **Note:** Model weights are not included because they exceed GitHub's file size limit.

---

# Running the Application

Launch the Streamlit application.

```bash
streamlit run app.py
```

The application will be available at

```text
http://localhost:8501
```

---

# Workflow

### 1. Upload a Leaf Image

Select the crop type and upload a diseased leaf image.

---

### 2. Disease Classification

The appropriate crop-specific ResNet-18 model predicts the disease.

---

### 3. Constraint Collection

The workflow pauses until the user provides:

* Budget category
* Recent pesticide usage (last 14 days)

---

### 4. Retrieval-Augmented Generation

The system retrieves relevant agricultural knowledge from Pinecone using metadata-filtered semantic search.

---

### 5. Recommendation Generation

The Cohere LLM generates a personalized **7-day Integrated Pest Management (IPM)** protocol based on:

* Disease prediction
* Retrieved agricultural literature
* Farmer constraints

---

### 6. Follow-up Assistance

Users can continue asking questions regarding:

* Soil management
* Irrigation
* Fertilizer usage
* Organic alternatives
* Preventive practices

The assistant maintains conversational context throughout the session.

---

# Knowledge Base

The RAG pipeline is built using official agricultural extension publications, including resources from:

* ICAR
* UC Davis
* Purdue University

Documents are:

* Parsed using PyPDF2
* Split into semantic chunks
* Embedded using Cohere `embed-english-v3.0`
* Indexed in Pinecone Serverless
* Retrieved through metadata-filtered semantic search

---

# Future Improvements

* Support for additional crops
* Mobile application
* Regional language support
* Weather-aware recommendations
* Satellite and IoT integration
* Offline deployment for low-connectivity regions

---

# License

This project was developed as part of a Bachelor of Technology (Computer Science & Engineering) project at **Model Institute of Engineering and Technology (MIET), Jammu**.

---

# Author

**Madhur Mahajan**
GitHub: https://github.com/Maddox-codes
