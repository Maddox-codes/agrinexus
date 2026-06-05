import os
import PyPDF2
import cohere
from pinecone import Pinecone, ServerlessSpec
import time

# --- 1. INITIALIZATION & API SETUP ---
COHERE_API_KEY = ""
PINECONE_API_KEY = ""

print("Initializing AI APIs...")
co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "agrinexus-index"

# --- 2. CREATE FRESH INDEX ---
if index_name not in pc.list_indexes().names():
    print(f"Creating fresh Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1024, # Cohere embed-english-v3.0 dimension size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Waiting for server deployment (15 seconds)...")
    time.sleep(15)

index = pc.Index(index_name)

MANUALS_DIR = "manuals"
TARGET_CROPS = ["Tomato", "Potato", "Pepper"]

# --- 3. PDF PROCESSING ENGINE ---
def extract_text_from_pdf(pdf_path):
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

print("\n--- Starting Bulk Ingestion Process ---")

# --- 4. CRAWL, CHUNK, AND UPLOAD ---
total_vectors_uploaded = 0

for crop in TARGET_CROPS:
    crop_dir = os.path.join(MANUALS_DIR, crop)
    
    if not os.path.exists(crop_dir):
        continue
        
    pdfs = [f for f in os.listdir(crop_dir) if f.endswith('.pdf')]
    if not pdfs:
        continue

    print(f"\nProcessing Corpus for: {crop}")
    
    for pdf_file in pdfs:
        pdf_path = os.path.join(crop_dir, pdf_file)
        print(f"  -> Reading {pdf_file}...")
        
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            continue
            
        chunks = [raw_text[i:i+1000] for i in range(0, len(raw_text), 800)]
        print(f"  -> Generated {len(chunks)} contextual chunks. Embedding...")
        
        vectors = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50: 
                continue 
                
            try:
                embedding = co.embed(
                    texts=[chunk], 
                    model='embed-english-v3.0', 
                    input_type='search_document'
                ).embeddings[0]
                
                vector_id = f"{crop}_{pdf_file.replace('.pdf', '')}_chunk_{i}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "crop": crop, 
                        "source": pdf_file,
                        "text": chunk
                    }
                })
                
                if len(vectors) >= 40:
                    index.upsert(vectors=vectors)
                    total_vectors_uploaded += len(vectors)
                    vectors = []
                    time.sleep(1) # Extra delay for free-tier safety
            except Exception as e:
                print(f"  -> API Error on chunk {i}: {e}. Skipping chunk.")
                time.sleep(2) # Back off if API complains
                
        if vectors:
            index.upsert(vectors=vectors)
            total_vectors_uploaded += len(vectors)
            
        print(f"   Successfully indexed {pdf_file} into Pinecone!")

print(f"\n CORPUS INGESTION COMPLETE! Total vector knowledge nodes deployed: {total_vectors_uploaded}")