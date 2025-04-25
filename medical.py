#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from langchain.schema import Document
import re
import torch
import logging

# ---------------- Setup Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- Custom CSS Styling ------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #0A58CA;
        text-align: center;
        padding: 0.5rem 0;
    }
    .stMarkdown h2 {
        color: #004080;
        border-bottom: 2px solid #004080;
        padding-bottom: 5px;
        margin-top: 2rem;
    }
    input[type="text"] {
        border-radius: 10px;
        border: 1px solid #0A58CA;
        padding: 10px;
    }
    div.stButton > button:first-child {
        background-color: #0A58CA;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #084298;
    }
    .answer-box {
        background-color: #dbeafe;
        padding: 1rem;
        border-left: 5px solid #0A58CA;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 1.05rem;
    }
    .doc-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Models ------------------
@st.cache_resource
def load_models():
    try:
        logger.info("Loading SentenceTransformer...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu', trust_remote_code=False)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("./medical", trust_remote_code=False)
        logger.info("Loading Flan-T5 model...")
        model = AutoModelForSeq2SeqLM.from_pretrained("./medical", device_map="cpu", trust_remote_code=False)
        logger.info("Models loaded successfully.")
        return embedding_model, tokenizer, model
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        return None, None, None

embedding_model, tokenizer, generator_model = load_models()
if embedding_model is None:
    st.stop()

# ---------------- Load FAISS & Docs ------------------
@st.cache_resource
def load_index_and_docs():
    try:
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        faiss_index_path = "faiss_index_file.index"
        documents_path = "documents.pkl"
        
        logger.info(f"Checking for FAISS index at: {os.path.abspath(faiss_index_path)}...")
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index file not found at {os.path.abspath(faiss_index_path)}. Please run the pipeline to generate it.")
        faiss_index = faiss.read_index(faiss_index_path)
        
        logger.info(f"Checking for documents at: {os.path.abspath(documents_path)}...")
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Documents file not found at {os.path.abspath(documents_path)}. Please run the pipeline to generate documents.pkl.")
        with open(documents_path, "rb") as f:
            documents = pickle.load(f)
        
        index_size = faiss_index.ntotal
        doc_size = len(documents)
        logger.info(f"FAISS index size: {index_size}, Documents size: {doc_size}")
        if index_size == 0:
            raise ValueError("FAISS index is empty.")
        if doc_size == 0:
            raise ValueError("Documents list is empty.")
        if index_size != doc_size:
            raise ValueError(f"Mismatch: FAISS index has {index_size} vectors, but documents.pkl has {doc_size} documents.")
        return faiss_index, documents
    except Exception as e:
        logger.error(f"Error loading FAISS index or documents: {e}")
        st.error(f"Error loading FAISS index or documents: {e}")
        return None, None

faiss_index, documents = load_index_and_docs()
if faiss_index is None:
    st.stop()

# ---------------- Answer Generator ------------------
def clean_context(context):
    """Clean context while preserving diagnostic, symptom, cause, and treatment terms."""
    # Remove JSON symbols
    context = re.sub(r'\$[^\s]+', '', context)
    context = re.sub(r'[{}\[\]]', '', context)
    context = re.sub(r'\"[^\"]+\"[:,]', '', context)
    # Protect specific terms
    protected_terms = r'\b(FeNO|spirometry|peak\s+flow|polyuria|polydipsia|dyspnea|cough|fever|hypertension|fatigue|salt\s+intake|stress|genetic|coronary\s+artery\s+disease|blood\s+glucose|A1C|inhalers|bronchodilators|corticosteroids|bacterial\s+infection|viral\s+infection|edema|headache|dizziness|echocardiography|BNP|chest\s+X-ray|ECG|electrocardiogram|Holter\s+monitoring|antibiotics|oxygen\s+therapy|antihypertensive\s+medications|dietary\s+modifications|palpitations|confusion|rapid\s+heart\s+rate|weakness|slurred\s+speech|smoking|air\s+pollution|jaundice|abdominal\s+pain|pallor|blood\s+cultures|vital\s+signs|diuretics|beta-blockers|glomerulonephritis|diabetes\s+mellitus|thyroid\s+disorders|chronic\s+kidney\s+disease|CKD)\b'
    context = re.sub(rf'(?!(?:{protected_terms}))[A-Z]+-?\d*\.?\d+\b|\bn?[A-Z]+OS\b|[A-Z]{2,}\s*\+\b', '', context, flags=re.IGNORECASE)
    # Replace abbreviations
    context = re.sub(r'\bwwp\b', 'well without pathology', context, flags=re.IGNORECASE)
    context = re.sub(r'\bedeman\b', 'edema', context, flags=re.IGNORECASE)
    # Normalize whitespace
    context = re.sub(r'\s+', ' ', context).strip()
    logger.info(f"Cleaned context sample: {context[:200]}...")
    return context

def generate_answer(context, query, model, tokenizer):
    is_symptom_query = "symptom" in query.lower()
    is_diagnosis_query = "diagnosed" in query.lower()
    is_cause_query = "cause" in query.lower()
    is_treatment_query = "treatment" in query.lower()
    
    # Clean context
    cleaned_context = clean_context(context)
    logger.info(f"Cleaned context length: {len(cleaned_context)} characters")
    
    if is_symptom_query:
        prompt = f"""You are a clinical expert. Extract symptoms for '{query}' from the context. Output only a list of symptoms in bullet-point format, one per line (e.g., '- Fever'). Map medical terms to common terms: 'polyuria' to 'Frequent urination', 'polydipsia' to 'Increased thirst', 'loss weight rapidly' to 'Weight loss', 'edema' to 'Swelling in legs', 'palpitations' to 'Irregular heartbeat'. Exclude causes, treatments, or narrative text. If no relevant symptoms are found, return an appropriate message for the condition (e.g., 'No specific symptoms found for hypertension...'). Use bullet points ('-') for symptoms.

        Context: {cleaned_context}
        Question: {query}
        Answer:"""
    else:
        prompt = f"""You are a clinical expert. Answer '{query}' using only the context. Provide a clear, concise answer in complete sentences, focusing on the requested information (e.g., diagnostic methods like echocardiography; causes like diabetes; treatments like diuretics). Use plain language, avoiding abbreviations, lab results, or unrelated terms. If the context lacks specific details, say 'Not enough information to answer the query.' Do not use bullet points or include symptoms unless requested.

        Context: {cleaned_context}
        Question: {query}
        Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        no_repeat_ngram_size=2,
        temperature=0.2,
        top_p=0.8,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Raw model response: {response}")
    
    if is_symptom_query:
        symptom_mappings = {
            r"\bfever\b": "Fever",
            r"\bcough\b(?:\s+productive)?": "Cough",
            r"shortness\s+of\s+breath|\bdyspnea\b": "Shortness of breath",
            r"\bfatigue\b": "Fatigue",
            r"\bthirst\b|\bpolydipsia\b": "Increased thirst",
            r"\burination\b|\bpolyuria\b": "Frequent urination",
            r"(?:weight\s+loss|loss\s+weight|\blose\s+weight\b)": "Weight loss",
            r"\bedema\b": "Swelling in legs",
            r"\bheadache\b": "Headache",
            r"\bdizziness\b": "Dizziness",
            r"\bpalpitations\b": "Irregular heartbeat",
            r"\bconfusion\b": "Confusion",
            r"rapid\s+heart\s+rate|\btachycardia\b": "Rapid heart rate",
            r"\bweakness\b": "Weakness",
            r"slurred\s+speech": "Slurred speech",
            r"\bjaundice\b": "Jaundice",
            r"\babdominal\s+pain\b": "Abdominal pain",
            r"\bpallor\b": "Pallor"
        }
        
        valid_symptoms = set()
        context_lower = cleaned_context.lower()
        
        # Diabetes-specific logic
        if "diabetes" in query.lower():
            diabetes_symptoms = {
                r"\bpolyuria\b": "Frequent urination",
                r"\bpolydipsia\b": "Increased thirst",
                r"(?:loss\s+weight|\blose\s+weight\b|weight\s+loss)": "Weight loss"
            }
            for pattern, display_name in diabetes_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for diabetes."

        # Pneumonia-specific logic
        if "pneumonia" in query.lower():
            pneumonia_symptoms = {
                r"\bcough\b": "Cough",
                r"\bfever\b": "Fever",
                r"shortness\s+of\s+breath|\bdyspnea\b": "Shortness of breath"
            }
            for pattern, display_name in pneumonia_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for pneumonia."

        # COPD-specific logic
        if "copd" in query.lower() or "chronic obstructive pulmonary disease" in query.lower():
            copd_symptoms = {
                r"\bcough\b": "Cough",
                r"shortness\s+of\s+breath|\bdyspnea\b": "Shortness of breath",
                r"\bfatigue\b": "Fatigue"
            }
            for pattern, display_name in copd_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for COPD."

        # Heart failure-specific logic
        if "heart failure" in query.lower():
            heart_failure_symptoms = {
                r"shortness\s+of\s+breath|\bdyspnea\b": "Shortness of breath",
                r"\bfatigue\b": "Fatigue",
                r"\bedema\b": "Swelling in legs"
            }
            for pattern, display_name in heart_failure_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for heart failure."

        # Hypertension-specific logic
        if "hypertension" in query.lower():
            hypertension_symptoms = {
                r"\bheadache\b": "Headache",
                r"\bdizziness\b": "Dizziness"
            }
            for pattern, display_name in hypertension_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for hypertension. It is often asymptomatic but may include headache or dizziness in severe cases."

        # Atrial fibrillation-specific logic
        if "atrial fibrillation" in query.lower():
            afib_symptoms = {
                r"\bpalpitations\b": "Irregular heartbeat",
                r"shortness\s+of\s+breath|\bdyspnea\b": "Shortness of breath",
                r"\bfatigue\b": "Fatigue"
            }
            for pattern, display_name in afib_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for atrial fibrillation."

        # Sepsis-specific logic
        if "sepsis" in query.lower():
            sepsis_symptoms = {
                r"\bfever\b": "Fever",
                r"\bconfusion\b": "Confusion",
                r"rapid\s+heart\s+rate|\btachycardia\b": "Rapid heart rate"
            }
            for pattern, display_name in sepsis_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for sepsis."

        # Kidney disease-specific logic
        if "kidney disease" in query.lower():
            kidney_symptoms = {
                r"\bedema\b": "Swelling in legs",
                r"\bfatigue\b": "Fatigue",
                r"\burination\b|\bpolyuria\b": "Frequent urination"
            }
            for pattern, display_name in kidney_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for kidney disease."

        # Stroke-specific logic
        if "stroke" in query.lower():
            stroke_symptoms = {
                r"\bweakness\b": "Weakness",
                r"slurred\s+speech": "Slurred speech",
                r"\bconfusion\b": "Confusion"
            }
            for pattern, display_name in stroke_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for stroke."

        # Liver disease-specific logic
        if "liver disease" in query.lower():
            liver_symptoms = {
                r"\bjaundice\b": "Jaundice",
                r"\bfatigue\b": "Fatigue",
                r"\babdominal\s+pain\b": "Abdominal pain"
            }
            for pattern, display_name in liver_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for liver disease."

        # Anemia-specific logic
        if "anemia" in query.lower():
            anemia_symptoms = {
                r"\bfatigue\b": "Fatigue",
                r"\bpallor\b": "Pallor",
                r"shortness\s+of\s+breath|\bdyspnea\b": "Shortness of breath"
            }
            for pattern, display_name in anemia_symptoms.items():
                if re.search(pattern, context_lower, re.IGNORECASE):
                    valid_symptoms.add(display_name)
            return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else "No specific symptoms found for anemia."

        # General symptom extraction (fallback)
        for pattern, display_name in symptom_mappings.items():
            if re.search(pattern, context_lower, re.IGNORECASE):
                valid_symptoms.add(display_name)
        
        # Model response as fallback
        answer_lines = [line.strip() for line in response.split('\n') if line.strip().startswith('-')]
        for line in answer_lines:
            symptom = line[2:].strip().lower()
            for pattern, display_name in symptom_mappings.items():
                if re.search(pattern, symptom, re.IGNORECASE):
                    valid_symptoms.add(display_name)
        
        return "\n".join([f"- {s}" for s in sorted(valid_symptoms)]) if valid_symptoms else f"No specific symptoms found for {query.split(' of ')[1]}."

    # Diagnosis fallback
    if is_diagnosis_query:
        diagnosis_terms = {
            r"\bspirometry\b": "lung function tests like spirometry",
            r"\bpeak\s+flow\b": "peak flow monitoring",
            r"\bfeno\b|\bexhaled\s+nitric\s+oxide\b": "exhaled nitric oxide testing",
            r"\ballergic\s+rhinitis\b": "allergy history",
            r"\bblood\s+glucose\b|\bfasting\s+blood\s+sugar\b|\bA1C\b": "blood glucose tests such as fasting blood sugar or A1C tests",
            r"\bechocardiography\b": "echocardiography",
            r"\bBNP\b|\bbrain\s+natriuretic\s+peptide\b": "blood tests like BNP",
            r"\bchest\s+X-ray\b": "chest X-ray",
            r"\bECG\b|\belectrocardiogram\b": "electrocardiogram (ECG)",
            r"\bHolter\s+monitoring\b": "Holter monitoring",
            r"\bblood\s+cultures\b": "blood cultures",
            r"\bvital\s+signs\b": "vital signs monitoring"
        }
        found_terms = []
        context_lower = cleaned_context.lower()
        for pattern, description in diagnosis_terms.items():
            if re.search(pattern, context_lower, re.IGNORECASE):
                found_terms.append(description)
        
        if found_terms:
            condition = query.split(' is ')[1].split(' diagnosed')[0].capitalize()
            response = f"{condition} is diagnosed through {', '.join(found_terms)} and patient history."
        else:
            response = response if response and "information" not in response.lower() else "Not enough information to answer the query."

    # Cause fallback
    if is_cause_query:
        cause_terms = {
            r"\bgenetic\b|\bhereditary\b": "genetic predisposition",
            r"\bsalt\s+intake\b|\bdiet\b": "high salt intake",
            r"\bstress\b": "stress",
            r"\bdiabetes\b|\bdiabetes\s+mellitus\b": "diabetes",
            r"\bheart\s+disease\b|\bcoronary\s+artery\s+disease\b": "heart disease",
            r"\bbacterial\s+infection\b": "bacterial infections",
            r"\bviral\s+infection\b": "viral infections",
            r"\bfungus\b|\bfungi\b": "fungi",
            r"\bsmoking\b": "smoking",
            r"\bair\s+pollution\b": "air pollution",
            r"\bglomerulonephritis\b": "glomerulonephritis",
            r"\bthyroid\s+disorders\b|\bhyperthyroidism\b|\bhypothyroidism\b": "thyroid disorders"
        }
        found_terms = []
        context_lower = cleaned_context.lower()
        for pattern, description in cause_terms.items():
            if re.search(pattern, context_lower, re.IGNORECASE):
                found_terms.append(description)
        
        if found_terms:
            condition = re.search(r'causes\s+(.+?)(\?|$)', query.lower()).group(1).capitalize()
            response = f"{condition} is caused by factors like {', '.join(found_terms)}. Not enough information for full details."
        else:
            response = response if response and "information" not in response.lower() else "Not enough information to answer the query."

    # Treatment fallback
    if is_treatment_query:
        treatment_terms = {
            r"\binhalers\b|\bbronchodilators\b": "inhalers such as bronchodilators",
            r"\bcorticosteroids\b": "corticosteroids",
            r"\blifestyle\s+changes\b": "lifestyle changes",
            r"\bantibiotics\b": "antibiotics",
            r"\boxygen\s+therapy\b": "oxygen therapy",
            r"\bantihypertensive\s+medications\b": "antihypertensive medications",
            r"\bdietary\s+modifications\b": "dietary modifications",
            r"\bdiuretics\b": "diuretics",
            r"\bbeta-blockers\b": "beta-blockers"
        }
        found_terms = []
        context_lower = cleaned_context.lower()
        for pattern, description in treatment_terms.items():
            if re.search(pattern, context_lower, re.IGNORECASE):
                found_terms.append(description)
        
        if found_terms:
            condition = query.split(' for ')[1].split('?')[0].capitalize()
            response = f"{condition} is treated with {', '.join(found_terms)}. Not enough information for full details."
        else:
            response = response if response and "information" not in response.lower() else "Not enough information to answer the query."

    # Clean narrative response
    response = re.sub(r'\b(?!FeNO\b|BNP\b|CKD\b)[A-Z]+-?\d*\.?\d+\b|\bn?[A-Z]+OS\b|[A-Z]{2,}\s*\+\b', '', response)
    response = response.strip()
    return response if response else "Not enough information to answer the query."

# ---------------- UI ------------------
st.title("🧠 Medical RAG Assistant")
st.markdown("Enter your **clinical query** below and get context-aware medical answers backed by document retrieval.", unsafe_allow_html=True)

query = st.text_input("Enter Clinical Query:", placeholder="e.g., Describe the symptoms of Type 2 Diabetes")

if st.button("Generate Answer"):
    if query.strip() == "":
        st.warning("Please enter a clinical query.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            # Encode + Retrieve
            logger.info("Encoding query...")
            query_embedding = embedding_model.encode([query], convert_to_tensor=False)
            logger.info("Performing FAISS search...")
            D, I = faiss_index.search(np.array(query_embedding, dtype=np.float32), k=15)  # Increased k
            valid_indices = [i for i in I[0] if i >= 0 and i < len(documents)]
            logger.info(f"FAISS returned indices: {I[0].tolist()}, Valid indices: {valid_indices}")
            if not valid_indices:
                st.error("No valid documents found in FAISS search. Please check FAISS index and documents.pkl.")
                st.stop()
            top_docs = [documents[i] for i in valid_indices]
            logger.info(f"Retrieved {len(top_docs)} valid documents: indices {valid_indices}")

            # Display documents
            st.markdown("## 📄 Retrieved Documents")
            for idx, doc in enumerate(top_docs, start=1):
                st.markdown(f"""
                    <div class="doc-box">
                        <b>Document {idx}:</b><br>
                        {doc.page_content[:1000]}
                    </div>
                """, unsafe_allow_html=True)

            # Generate answer
            logger.info("Generating answer...")
            context = " ".join([doc.page_content for doc in top_docs])[:4000]
            answer = generate_answer(context, query, generator_model, tokenizer)
            logger.info("Answer generated successfully.")

            # Display final answer
            st.markdown("## 🤖 Generated Answer")
            st.markdown(f"""<div class="answer-box">{answer}</div>""", unsafe_allow_html=True)
