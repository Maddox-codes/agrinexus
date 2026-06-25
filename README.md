
# AgriNexus: Vision-Integrated Stateful Agentic Triage System

## Overview

AgriNexus is an advanced AI-driven agricultural decision-support system designed to solve the **"Stateless Diagnosis"** problem in precision agriculture. While traditional Ag-Tech applications act as stateless oracles—identifying diseases but recommending generic, potentially inappropriate chemical treatments—AgriNexus enforces a **Deterministic State Machine** that deliberately pauses the diagnostic workflow to collect a farmer's budget constraints and 14-day pesticide/spray history.

This persistent contextual information is then used to filter a **1024-dimensional Pinecone Serverless Vector Database** and ground the **Cohere `command-r-plus-08-2024`** large language model, enabling the generation of scientifically accurate, organic-first, 7-day **Integrated Pest Management (IPM)** protocols.

---

# Core Architecture

### 1. Multi-Crop Perception Engine (Model Router)

Three independently fine-tuned **PyTorch ResNet-18** models are isolated by crop domain (Tomato, Potato, and Bell Pepper), eliminating inter-crop class confusion while improving diagnostic accuracy.

### 2. Deterministic State Machine

A Streamlit-based state machine prevents premature LLM reasoning by enforcing contextual validation before Retrieval-Augmented Generation (RAG) begins. The application cannot proceed until the required user constraints have been collected.

### 3. Agentic RAG Layer

Official agricultural manuals from **ICAR**, **UC Davis**, and **Purdue University** were parsed into **981 semantic chunks**, embedded using Cohere's `embed-english-v3.0` model, and indexed inside a Pinecone Serverless vector database. Retrieval uses strict crop-based metadata filtering before context is passed to the language model.

### 4. Multi-Turn LLM Orchestration

The grounded LLM generates personalized IPM recommendations and supports agentic follow-up conversations covering topics such as soil health, nutrient management, irrigation, and additional agronomic practices.

---

# Repository Structure

```
.
├── app.py
├── train_resnet.py
├── ingest_bulk.py
├── requirements.txt
├── .env
└── models/
```

| File              | Description                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| `app.py`          | Main Streamlit application implementing the stateful workflow and user interface                       |
| `train_resnet.py` | PyTorch training script for fine-tuning the ResNet-18 perception models using the PlantVillage dataset |
| `ingest_bulk.py`  | Pipeline for PDF parsing, Cohere embedding generation, and Pinecone vector database ingestion          |
| `models/`         | Directory containing the trained `.pth` model weights (not included in repository)                     |

---

# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Maddox-codes/agrinexus.git
cd agrinexus
```

---

## 2. Create a Virtual Environment

To avoid dependency conflicts, create and activate a clean Python virtual environment.

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## 3. Install Dependencies

Install all required software libraries.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root.

```text
COHERE_API_KEY=your_production_cohere_api_key_here
PINECONE_API_KEY=your_production_pinecone_api_key_here
```

---

## 5. Add the Trained Models

GitHub file size restrictions prevent the trained model weights from being included in the repository.

Place the following files inside the `models/` directory:

```
models/
├── agrinexus_tomato_resnet18.pth
├── agrinexus_potato_resnet18.pth
└── agrinexus_pepper_resnet18.pth
```

---

# Deployment & Demo Instructions

## 1. Launch the Application

Start the Streamlit server.

```bash
streamlit run app.py
```

---

## 2. Open the Dashboard

After initialization, open the following address in your browser:

```
http://localhost:8501
```

---

## 3. Execute a Complete Diagnostic Workflow

### Step 1 – Perception Ingress

* Select the crop type (Tomato, Potato, or Bell Pepper).
* Upload a symptomatic leaf image.
* The application routes the image to the corresponding ResNet-18 model.
* A disease prediction is generated.

---

### Step 2 – Constraint Interview

The deterministic workflow pauses automatically.

Provide:

* Budget category
* Previous 14-day pesticide/spray history

These constraints unlock the Retrieval-Augmented Generation pipeline.

---

### Step 3 – Context-Aware Knowledge Generation

Click **Generate Context-Aware RAG Protocol**.

The system will:

1. Filter the Pinecone vector database using crop metadata.
2. Retrieve the most relevant agricultural documents.
3. Ground the Cohere LLM using the retrieved evidence.
4. Produce a personalized, organic-first, 7-day Integrated Pest Management protocol.

---

### Step 4 – Agentic Follow-up

Continue interacting with the system using the multi-turn chat interface.

Example follow-up questions include:

* Soil remediation strategies
* Irrigation planning
* Fertilizer recommendations
* Organic treatment alternatives
* Preventive disease management

The application maintains conversational session state, enabling context-aware follow-up responses without repeating previously supplied information.
