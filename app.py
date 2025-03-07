import streamlit as st
import spacy
import pandas as pd
import json
import os
import time
from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import psutil
import gc

# Set page configuration
st.set_page_config(
    page_title="Medical Transcript Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTextInput, .stTextArea {
        padding: 1rem;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .header-text {
        color: #0066cc;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader-text {
        color: #0066cc;
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='header-text'>Medical Transcript Analyzer</div>", unsafe_allow_html=True)
st.markdown("""
This application analyzes medical conversation transcripts to extract structured information, 
perform sentiment analysis, and generate SOAP notes for physicians.
""")

# Initialize configuration in session state
if 'config' not in st.session_state:
    st.session_state.config = {
        'spacy_model': 'en_core_web_sm',  # Use the smaller model by default to save memory
        'sentiment_model': 'distilbert-base-uncased',
        'batch_size': 8,  # For processing larger batches
        'max_text_length': 512  # Limit text length for sentiment analysis
    }

# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.spacy_model = None
    st.session_state.sentiment_model = None

# Memory monitoring function
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    return memory_usage_mb

# Function to ensure SpaCy model is installed
@st.cache_resource
def ensure_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"SpaCy model '{model_name}' not found. Downloading now...")
        os.system(f"python -m spacy download {model_name}")
        return spacy.load(model_name)

# Load NLP models with caching
@st.cache_resource
def load_sentiment_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def load_models():
    with st.spinner('Loading NLP models... This may take a minute.'):
        progress_bar = st.progress(0)
        
        # Monitor initial memory usage
        initial_memory = get_memory_usage()
        st.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        try:
            # Load SpaCy model
            progress_bar.progress(10)
            st.session_state.spacy_model = ensure_spacy_model(st.session_state.config['spacy_model'])
            progress_bar.progress(50)
            
            # Load Hugging Face sentiment model
            progress_bar.progress(60)
            st.session_state.sentiment_model = load_sentiment_model(st.session_state.config['sentiment_model'])
            progress_bar.progress(90)
            
            # Force garbage collection
            gc.collect()
            
            current_memory = get_memory_usage()
            st.info(f"Memory usage after loading models: {current_memory:.2f} MB (Increase: {current_memory - initial_memory:.2f} MB)")
            
            st.session_state.models_loaded = True
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            st.success("Models loaded successfully!")
            
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.session_state.models_loaded = False

# Optimized extraction function with LRU cache for repeated texts
@lru_cache(maxsize=128)
def extract_medical_details_cached(transcript_text):
    # Process the document with SpaCy
    doc = st.session_state.spacy_model(transcript_text)
    
    symptoms = []
    treatments = []
    diagnosis = []
    timeframes = []

    # Extract entities
    for ent in doc.ents:
        if ent.label_ in ["DATE", "TIME"]:
            timeframes.append(ent.text)

    # Rule-based extraction using lower-cased text for efficiency
    text_lower = transcript_text.lower()
    
    # Keyword dictionaries for more organized extraction
    symptom_keywords = {
        "pain": "Pain/Discomfort",
        "discomfort": "Pain/Discomfort",
        "neck pain": "Neck pain",
        "back pain": "Back pain",
        "hit my head": "Head impact"
    }
    
    treatment_keywords = {
        "physiotherapy": "Physiotherapy sessions",
        "painkiller": "Painkillers"
    }
    
    diagnosis_keywords = {
        "whiplash": "Whiplash injury"
    }
    
    # Check for symptoms
    for keyword, symptom in symptom_keywords.items():
        if keyword in text_lower:
            symptoms.append(symptom)
    
    # Check for treatments
    for keyword, treatment in treatment_keywords.items():
        if keyword in text_lower:
            treatments.append(treatment)
    
    # Check for diagnoses
    for keyword, diag in diagnosis_keywords.items():
        if keyword in text_lower:
            diagnosis.append(diag)

    return {
        "Symptoms": list(set(symptoms)),
        "Treatment": list(set(treatments)),
        "Diagnosis": list(set(diagnosis)),
        "Timeframes": list(set(timeframes))
    }

# Non-cached wrapper for the cached function
def extract_medical_details(transcript_text):
    # Trim text if it's too long to avoid memory issues
    if len(transcript_text) > 10000:
        transcript_text = transcript_text[:10000]
    return extract_medical_details_cached(transcript_text)

# Structured summary function
def structured_summary(medical_details, transcript_text):
    text_lower = transcript_text.lower()
    
    # Default values
    patient_name = "Unknown"
    current_status = "Unknown"
    prognosis = "Unknown"
    
    # Extract patient name - using simple pattern matching
    if "Ms. Jones" in transcript_text:
        patient_name = "Ms. Jones"
    
    # Determine current status with more efficient checks
    if "occasional backache" in text_lower:
        current_status = "Occasional backache"
    elif "better" in text_lower:
        current_status = "Improving"
    
    # Determine prognosis
    if "improving" in text_lower:
        prognosis = "Improving, full recovery expected"
    
    return {
        "Patient_Name": patient_name,
        "Symptoms": medical_details["Symptoms"],
        "Diagnosis": medical_details["Diagnosis"][0] if medical_details["Diagnosis"] else "Not specified",
        "Treatment": medical_details["Treatment"],
        "Current_Status": current_status,
        "Prognosis": prognosis
    }

# Optimized sentiment analysis
def analyze_sentiment_and_intent(patient_text):
    # Extract only patient statements to reduce processing
    patient_statements = []
    for line in patient_text.split('\n'):
        if line.strip().startswith("Patient:"):
            patient_statements.append(line.strip()[8:].strip())
    
    patient_combined = " ".join(patient_statements)
    
    # Limit text length to prevent memory issues
    if len(patient_combined) > st.session_state.config['max_text_length']:
        patient_combined = patient_combined[:st.session_state.config['max_text_length']]
    
    # Sentiment analysis using the transformer model
    classification = st.session_state.sentiment_model(patient_combined)
    raw_label = classification[0]['label']
    
    # Map raw sentiment label to desired format
    sentiment_map = {
        'POSITIVE': "Reassured",
        'NEGATIVE': "Anxious",
        'NEUTRAL': "Neutral"
    }
    sentiment = sentiment_map.get(raw_label, "Neutral")
    
    # Rule-based intent detection - more efficient with a single pass
    intent = "Providing information"  # Default
    lowered_text = patient_combined.lower()
    
    # Check for intents in priority order
    if any(word in lowered_text for word in ["worry", "anxious", "concern"]):
        intent = "Seeking reassurance"
    elif any(word in lowered_text for word in ["better", "improving", "helped"]):
        intent = "Reporting improvement"
    elif any(word in lowered_text for word in ["pain", "symptom"]):
        intent = "Reporting symptoms"
    
    return {
        "Sentiment": sentiment,
        "Intent": intent,
        "Confidence": classification[0]['score']
    }

# SOAP Note Generation with more efficient text processing
def generate_soap_note(summary, transcript_text):
    text_lower = transcript_text.lower()
    
    # Default values
    history = "Unknown"
    physical_exam = "Unknown"
    assessment = "Unknown"
    plan = "Unknown"
    
    # Extract history information
    if "car accident" in text_lower:
        history = "Patient involved in a car accident"
        if "September" in transcript_text:
            history += " in September"
    
    # Check for physical examination
    if "physical examination" in text_lower:
        physical_exam = "Physical examination mentioned, details not provided"
    
    # Get assessment from summary
    if summary["Diagnosis"] != "Not specified":
        assessment = summary["Diagnosis"]
    
    # Treatment plan components
    plan_components = []
    if "physiotherapy" in text_lower:
        plan_components.append("Continue physiotherapy")
    if "painkiller" in text_lower:
        plan_components.append("Use painkillers as needed")
    
    if plan_components:
        plan = ", ".join(plan_components)
    
    return {
        "Subjective": {
            "Chief_Complaint": ", ".join(summary["Symptoms"]) if summary["Symptoms"] else "Not specified",
            "History_of_Present_Illness": history
        },
        "Objective": {
            "Physical_Exam": physical_exam,
            "Observations": "Based on patient's statements"
        },
        "Assessment": {
            "Diagnosis": summary["Diagnosis"],
            "Severity": "Improving based on patient statements" if "improving" in text_lower else "Unknown"
        },
        "Plan": {
            "Treatment": plan,
            "Follow_Up": "As needed based on symptom progression"
        }
    }

# Batch processing function with memory optimization
def process_transcripts_in_batches(transcripts, batch_size=8):
    results = []
    total = len(transcripts)
    
    for i in range(0, total, batch_size):
        batch = transcripts[i:i+batch_size]
        batch_results = []
        
        for transcript_data in batch:
            content = transcript_data["content"]
            filename = transcript_data["filename"]
            
            # Run analysis pipeline
            medical_details = extract_medical_details(content)
            summary = structured_summary(medical_details, content)
            sentiment_analysis = analyze_sentiment_and_intent(content)
            soap_note = generate_soap_note(summary, content)
            
            # Compile results
            batch_results.append({
                "filename": filename,
                "medical_details": medical_details,
                "summary": summary,
                "sentiment_analysis": sentiment_analysis,
                "soap_note": soap_note
            })
        
        # Append batch results and force garbage collection
        results.extend(batch_results)
        gc.collect()
        
        # Update progress
        progress = min(1.0, (i + len(batch)) / total)
        st.progress(progress)
    
    return results

# Advanced settings sidebar
with st.sidebar:
    st.header("Advanced Settings")
    
    st.subheader("Model Configuration")
    spacy_model_option = st.selectbox(
        "SpaCy Model",
        ["en_core_web_sm"],
        index=0,
        help="Smaller model uses less memory but may be less accurate"
    )
    
    batch_size = st.slider(
        "Batch Processing Size",
        min_value=1,
        max_value=20,
        value=8,
        help="Larger batch sizes process faster but use more memory"
    )
    
    max_text_length = st.slider(
        "Max Text Length for Analysis",
        min_value=128,
        max_value=1024,
        value=512,
        help="Limit text length to prevent memory issues"
    )
    
    # Update configuration if changed
    if (spacy_model_option != st.session_state.config['spacy_model'] or
        batch_size != st.session_state.config['batch_size'] or
        max_text_length != st.session_state.config['max_text_length']):
        
        st.session_state.config['spacy_model'] = spacy_model_option
        st.session_state.config['batch_size'] = batch_size
        st.session_state.config['max_text_length'] = max_text_length
        
        # Reset models if SpaCy model changed
        if spacy_model_option != st.session_state.config['spacy_model']:
            st.session_state.models_loaded = False
            st.warning("Model configuration changed. Models will reload on next analysis.")
    
    # Memory monitoring
    if st.button("Check Memory Usage"):
        memory_usage = get_memory_usage()
        st.info(f"Current memory usage: {memory_usage:.2f} MB")
    
    # Force garbage collection
    if st.button("Free Memory"):
        before = get_memory_usage()
        gc.collect()
        after = get_memory_usage()
        st.success(f"Memory freed: {before - after:.2f} MB")

# Main application logic
tab1, tab2= st.tabs(["📋 Transcript Analysis", "📊 Batch Processing"])

with tab1:
    # Input section
    st.markdown("<div class='subheader-text'>Input Medical Transcript</div>", unsafe_allow_html=True)
    
    # Sample transcript option
    sample_transcript = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when 
I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain 
in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but 
they didn't do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take 
painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help 
with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty 
concentrating?
Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from 
doing anything.
Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
"""
    
    use_sample = st.checkbox("Use sample transcript", value=True)
    
    if use_sample:
        transcript = st.text_area("Medical Transcript", value=sample_transcript, height=300)
    else:
        transcript = st.text_area("Medical Transcript", height=300, 
                                placeholder="Enter physician-patient conversation...")
    
    # Load models when needed
    if not st.session_state.models_loaded and (transcript.strip() != ""):
        load_models()
    
    # Process transcript button
    analyze_button = st.button("Analyze Transcript", type="primary", disabled=not st.session_state.models_loaded)
    
    if analyze_button and transcript:
        with st.spinner('Analyzing transcript...'):
            # Run the analysis pipeline
            medical_details = extract_medical_details(transcript)
            summary = structured_summary(medical_details, transcript)
            sentiment_analysis = analyze_sentiment_and_intent(transcript)
            soap_note = generate_soap_note(summary, transcript)
            
            # Display results
            st.markdown("<div class='subheader-text'>Analysis Results</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Medical Details")
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.write("**Symptoms:**", ", ".join(medical_details["Symptoms"]) if medical_details["Symptoms"] else "None detected")
                st.write("**Treatments:**", ", ".join(medical_details["Treatment"]) if medical_details["Treatment"] else "None detected")
                st.write("**Diagnosis:**", ", ".join(medical_details["Diagnosis"]) if medical_details["Diagnosis"] else "None detected")
                st.write("**Timeframes:**", ", ".join(medical_details["Timeframes"]) if medical_details["Timeframes"] else "None detected")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("#### 📊 Sentiment Analysis")
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.write("**Patient Sentiment:**", sentiment_analysis["Sentiment"])
                st.write("**Patient Intent:**", sentiment_analysis["Intent"])
                st.write("**Confidence Score:**", f"{sentiment_analysis['Confidence']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📝 Patient Summary")
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                for key, value in summary.items():
                    if isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ')}:**", ", ".join(value) if value else "Not specified")
                    else:
                        st.write(f"**{key.replace('_', ' ')}:**", value)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("#### 📑 SOAP Note")
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                
                # Subjective
                st.markdown("**S - Subjective**")
                st.write("Chief Complaint:", soap_note["Subjective"]["Chief_Complaint"])
                st.write("History of Present Illness:", soap_note["Subjective"]["History_of_Present_Illness"])
                
                # Objective
                st.markdown("**O - Objective**")
                st.write("Physical Exam:", soap_note["Objective"]["Physical_Exam"])
                st.write("Observations:", soap_note["Objective"]["Observations"])
                
                # Assessment
                st.markdown("**A - Assessment**")
                st.write("Diagnosis:", soap_note["Assessment"]["Diagnosis"])
                st.write("Severity:", soap_note["Assessment"]["Severity"])
                
                # Plan
                st.markdown("**P - Plan**")
                st.write("Treatment:", soap_note["Plan"]["Treatment"])
                st.write("Follow-Up:", soap_note["Plan"]["Follow_Up"])
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Option to download results as JSON
            result_json = {
                "medical_details": medical_details,
                "summary": summary,
                "sentiment_analysis": sentiment_analysis,
                "soap_note": soap_note
            }
            
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps(result_json, indent=4),
                file_name="medical_analysis_results.json",
                mime="application/json",
            )

with tab2:
    st.markdown("<div class='subheader-text'>Batch Processing</div>", unsafe_allow_html=True)
    st.markdown("""
    Upload multiple transcripts for batch processing. Each transcript should be a separate text file.
    """)
    
    uploaded_files = st.file_uploader("Upload transcript files", accept_multiple_files=True, type=['txt'])
    
    if uploaded_files:
        if not st.session_state.models_loaded:
            load_models()
        
        process_batch = st.button("Process Batch", type="primary", disabled=not st.session_state.models_loaded)
        
        if process_batch:
            with st.spinner('Processing files...'):
                # Prepare transcripts for batch processing
                transcripts = []
                for uploaded_file in uploaded_files:
                    content = uploaded_file.read().decode("utf-8")
                    filename = uploaded_file.name.split('.')[0]
                    transcripts.append({"filename": filename, "content": content})
                
                # Process in batches
                results = process_transcripts_in_batches(transcripts, st.session_state.config['batch_size'])
                
                # Display batch results
                st.success(f"Processed {len(results)} files successfully!")
                
                # Create a DataFrame for display
                results_df = pd.DataFrame({
                    "Filename": [r["filename"] for r in results],
                    "Patient": [r["summary"]["Patient_Name"] for r in results],
                    "Diagnosis": [r["summary"]["Diagnosis"] for r in results],
                    "Sentiment": [r["sentiment_analysis"]["Sentiment"] for r in results],
                    "Current Status": [r["summary"]["Current_Status"] for r in results]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Option to download batch results
                st.download_button(
                    label="Download Batch Results as JSON",
                    data=json.dumps(results, indent=4),
                    file_name="batch_analysis_results.json",
                    mime="application/json",
                )
