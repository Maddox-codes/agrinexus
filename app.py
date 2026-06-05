import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cohere
from pinecone import Pinecone
import time

# --- 1. INITIALIZATION & API SETUP ---
COHERE_API_KEY = ""
PINECONE_API_KEY = ""

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("agrinexus-index")

# --- 2. MULTI-CROP CONFIGURATION ---
CROP_CONFIG = {
    "Tomato": {
        "weights": "agrinexus_tomato_resnet18.pth",
        "classes": ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot", "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"],
        "num_classes": 10
    },
    "Potato": {
        "weights": "agrinexus_potato_resnet18.pth",
        "classes": ["Early Blight", "Healthy", "Late Blight"],
        "num_classes": 3
    },
    "Pepper": {
        "weights": "agrinexus_pepper_resnet18.pth",
        "classes": ["Bacterial Spot", "Healthy"],
        "num_classes": 2
    }
}

@st.cache_resource
def load_model(crop_name):
    config = CROP_CONFIG[crop_name]
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config["num_classes"])
    
    try:
        model.load_state_dict(torch.load(config["weights"], map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None

def predict_image(image, model, classes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted[0].item()]

# --- 3. ADVANCED STATE MANAGEMENT ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'upload'
if 'disease' not in st.session_state:
    st.session_state.disease = None
if 'current_crop' not in st.session_state:
    st.session_state.current_crop = "Tomato"
if 'main_plan' not in st.session_state:
    st.session_state.main_plan = None
if 'follow_up_answer' not in st.session_state:
    st.session_state.follow_up_answer = None

# --- 4. RAG QUERY FUNCTION ---
def query_agent(prompt_text, filter_crop):
    # 1. Convert prompt to vector
    query_embedding = co.embed(texts=[prompt_text], model='embed-english-v3.0', input_type='search_query').embeddings[0]
    
    # 2. Search Pinecone with strict Crop Metadata filtering
    search_results = index.query(vector=query_embedding, top_k=3, include_metadata=True, filter={"crop": filter_crop})
    
    if search_results['matches']:
        # 3. Extract text and sources
        context_chunks = "\n\n".join([match['metadata']['text'] for match in search_results['matches']])
        source_docs = list(set([match['metadata']['source'] for match in search_results['matches']]))
        
        # 4. Generate AI response using the upgraded command-r model
        system_prompt = f"""
        You are AgriNexus, an expert agricultural AI. 
        Answer the farmer's query using ONLY the official documentation provided below.
        Be concise, empathetic, and highly actionable.
        
        OFFICIAL DOCUMENTATION:
        {context_chunks}
        
        FARMER QUERY: {prompt_text}
        """
        generation = co.chat(message=system_prompt, model="command-r-plus-08-2024")
        return generation.text, source_docs
    return None, None

# --- 5. USER INTERFACE ---
st.set_page_config(page_title="AgriNexus Triage", layout="wide")

st.sidebar.title("AgriNexus Settings")
st.sidebar.markdown("### Model Router Configuration")
selected_crop = st.sidebar.selectbox("Select Target Crop", ["Tomato", "Potato", "Pepper"])

# Reset state if user changes crop mid-diagnosis
if selected_crop != st.session_state.current_crop:
    st.session_state.current_crop = selected_crop
    st.session_state.stage = 'upload'
    st.session_state.disease = None
    st.session_state.main_plan = None
    st.session_state.follow_up_answer = None
    st.rerun()

st.title("🌱 AgriNexus: Vision-Integrated Agentic Triage")
st.markdown("**Phase 2 Production Build** | Multi-Crop Router & Generative RAG Activated")

if st.session_state.stage == 'upload':
    st.header(f"Step 1: Visual Perception ({selected_crop} Model)")
    uploaded_file = st.file_uploader(f"Upload a {selected_crop} Leaf Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button(f"Run {selected_crop} AI Diagnosis"):
            with st.spinner(f"Loading specialized ResNet-18 model for {selected_crop}..."):
                model = load_model(selected_crop)
                if model is None:
                    st.error(f"Error: Could not find '{CROP_CONFIG[selected_crop]['weights']}'. Did you download it from Colab?")
                    st.stop()
                
                prediction = predict_image(image, model, CROP_CONFIG[selected_crop]["classes"])
                st.session_state.disease = prediction
                
                if "Healthy" in prediction:
                    st.success("Diagnosis: Healthy Leaf. No treatment required. Maintaining state.")
                else:
                    st.error(f"⚠️ Diagnosis: {prediction} detected.")
                    st.session_state.stage = 'interview'
                    st.rerun()

elif st.session_state.stage == 'interview':
    st.header("Step 2: Constraint Interview")
    st.warning("SYSTEM HALT: Cannot prescribe chemicals without financial and historical context.")
    
    budget = st.selectbox("Farmer Budget Level", ["Low (Organic/Local Focus)", "Medium", "High (Aggressive Intervention)"])
    history = st.text_input("Chemicals sprayed in last 14 days (Leave blank if none):")
    
    if st.button("Generate Context-Aware RAG Protocol"):
        with st.spinner("Querying Vector Database and Orchestrating LLM..."):
            prompt = f"Write a step-by-step 7-day action plan for {st.session_state.disease} on {selected_crop}. Budget: {budget}. Past chemicals: {history}."
            answer, sources = query_agent(prompt, selected_crop)
            
            if answer:
                st.session_state.main_plan = {"text": answer, "sources": sources}
                st.session_state.follow_up_answer = None 
            else:
                st.error("No relevant protocol found in the database for this specific crop and disease.")

    # --- THE AGENTIC DEEP DIVE SECTION ---
    if st.session_state.main_plan:
        st.success("✅ Contextual Matches Retrieved from Official Agricultural Corpus!")
        st.markdown(f"**Sources Referenced:** `{', '.join(st.session_state.main_plan['sources'])}`")
        
        st.markdown("### 📋 Your 7-Day Action Plan")
        st.info(st.session_state.main_plan["text"])
        
        st.markdown("---")
        st.markdown("### 🔍 Agentic Follow-Up: Explore More Options")
        st.write("What else would you like to know about managing this crop?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🌱 Soil & Fertilizer Management"):
                with st.spinner("Researching soil protocols..."):
                    ans, srcs = query_agent(f"What are the best soil preparation and fertilizer practices for {selected_crop} to prevent {st.session_state.disease}?", selected_crop)
                    st.session_state.follow_up_answer = {"title": "Soil & Fertilizer Management", "text": ans, "sources": srcs}
        with col2:
            if st.button("💧 Irrigation Best Practices"):
                with st.spinner("Researching irrigation protocols..."):
                    ans, srcs = query_agent(f"How should I manage watering and irrigation for {selected_crop} while fighting {st.session_state.disease}?", selected_crop)
                    st.session_state.follow_up_answer = {"title": "Irrigation Best Practices", "text": ans, "sources": srcs}
        with col3:
            if st.button("🛡️ Next Season Prevention"):
                with st.spinner("Researching preventative protocols..."):
                    ans, srcs = query_agent(f"What crop rotation or preventative steps should I take next season to avoid {st.session_state.disease} in {selected_crop}?", selected_crop)
                    st.session_state.follow_up_answer = {"title": "Next Season Prevention", "text": ans, "sources": srcs}
                    
        # Display the follow-up answer if a button was clicked
        if st.session_state.follow_up_answer:
            st.markdown(f"#### 💡 {st.session_state.follow_up_answer['title']}")
            st.success(st.session_state.follow_up_answer["text"])
            st.caption(f"Sources: {', '.join(st.session_state.follow_up_answer['sources'])}")

        st.markdown("---")
        if st.button("Reset System (New Scan)"):
            st.session_state.stage = 'upload'
            st.session_state.disease = None
            st.session_state.main_plan = None
            st.session_state.follow_up_answer = None
            st.rerun()