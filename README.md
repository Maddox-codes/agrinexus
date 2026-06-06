# AgriNexus: Vision-Integrated Stateful Agentic Triage System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Stateful-FF4B4B)
![Cohere](https://img.shields.io/badge/Cohere-Command_R+-39594D)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-000000)

## Overview
AgriNexus is an advanced AI-driven agricultural decision-support system designed to solve the "Stateless Diagnosis" problem in precision agriculture. While traditional Ag-Tech apps act as stateless oracles—identifying diseases but recommending generic, toxic chemicals—AgriNexus enforces a **Deterministic State Machine** that halts the workflow to capture a farmer's budget constraints and 14-day chemical spray history. 

This stateful context is used to filter a 1024-dimensional **Pinecone Serverless Vector Database** and ground the **Cohere command-r-plus-08-2024** LLM, generating scientifically accurate, organic-first, 7-day Integrated Pest Management (IPM) protocols.

## Core Architecture
1. **Multi-Crop Perception Engine (Model Router):** Three distinct, fine-tuned PyTorch ResNet-18 models isolated by crop domain (Tomato, Potato, Bell Pepper) to eliminate inter-crop class confusion.
2. **Deterministic State Machine:** A rigid Streamlit-based logic gate that prevents LLM hallucination by demanding contextual constraint validation before initiating RAG.
3. **Agentic RAG Layer:** 981 semantic chunks parsed from official agricultural manuals (ICAR, UC Davis, Purdue), embedded via `embed-english-v3.0`, and dynamically queried using strict crop-metadata filtering.
4. **Multi-Turn LLM Orchestration:** Generates tailored IPMs and allows for agentic deep-dives into soil management and irrigation practices.

## Repository Structure
- `app.py`: Main Streamlit application and Stateful UI logic.
- `train_resnet.py`: PyTorch script for fine-tuning the ResNet-18 models on the PlantVillage dataset.
- `ingest_bulk.py`: PDF parsing, Cohere embedding generation, and Pinecone database upsertion logic.
- `models/`: Directory containing the domain-isolated `.pth` weight files (omitted from repo due to size).

## Installation & Setup

1. **Clone the repository:**
```bash
   git clone [https://github.com/Maddox-codes/agrinexus.git](https://github.com/Maddox-codes/agrinexus.git)
   cd agrinexus
