

# Medical Transcript Analyzer 🏥

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red) ![License](https://img.shields.io/badge/License-MIT-green)

A powerful web application built with **Streamlit** to analyze medical conversation transcripts, extract structured information, perform sentiment analysis, and generate SOAP notes for physicians.

🔗 **Live Demo**: [Medical Transcript Analyzer](https://physician-tracker.streamlit.app/)

---

## 📖 Overview

The **Medical Transcript Analyzer** is designed to assist healthcare professionals by automating the analysis of physician-patient conversation transcripts. It leverages natural language processing (NLP) techniques using SpaCy and Hugging Face transformers to extract key medical details, assess patient sentiment, and produce structured clinical documentation (SOAP notes).

This tool is ideal for physicians, medical researchers, or anyone looking to streamline the processing of clinical conversations while optimizing memory usage and performance.

---

## ✨ Features

- **Transcript Analysis**: Extract symptoms, treatments, diagnoses, and timeframes from medical conversations.
- **Sentiment Analysis**: Detect patient sentiment (e.g., Reassured, Anxious, Neutral) and intent using DistilBERT.
- **SOAP Note Generation**: Automatically generate Subjective, Objective, Assessment, and Plan (SOAP) notes.
- **Batch Processing**: Analyze multiple transcripts simultaneously with memory-efficient batching.
- **Memory Optimization**: Includes memory usage monitoring, garbage collection, and configurable batch sizes.
- **User-Friendly Interface**: Built with Streamlit for an intuitive, responsive web experience.
- **Downloadable Results**: Export analysis results as JSON files for further use.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning the repository)

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/vempatisaivishal/physician_tracker.git
  
   ```

2. **Create a Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   streamlit>=1.0.0
   spacy>=3.0.0
   transformers>=4.0.0
   torch>=1.9.0
   pandas>=1.3.0
   psutil>=5.8.0
   ```

4. **Download SpaCy Model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

   Open your browser at `http://localhost:8501` to access the app.

---

## 🚀 Usage

1. **Single Transcript Analysis**
   - Navigate to the "Transcript Analysis" tab.
   - Use the sample transcript or paste your own physician-patient conversation.
   - Click "Analyze Transcript" to view extracted details, sentiment, summary, and SOAP note.
   - Download results as a JSON file.

2. **Batch Processing**
   - Go to the "Batch Processing" tab.
   - Upload multiple `.txt` files containing transcripts.
   - Click "Process Batch" to analyze all files and view results in a table.
   - Export batch results as a JSON file.

3. **Advanced Settings**
   - Adjust SpaCy model, batch size, and max text length in the sidebar.
   - Monitor memory usage and free memory as needed.

---

## ⚙️ Configuration

The app includes configurable settings in the sidebar:

| Setting             | Description                                      | Default Value      |
|---------------------|--------------------------------------------------|--------------------|
| SpaCy Model         | NLP model for entity extraction                  | `en_core_web_sm`   |
| Batch Size          | Number of transcripts processed per batch        | 8                  |
| Max Text Length     | Maximum characters for sentiment analysis        | 512                |

Changes to these settings are applied dynamically, with model reloading triggered only when necessary.

---

## 📋 Example Transcript

```plaintext
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st. Another car hit me from behind...
```

**Output:**
- **Symptoms**: Pain/Discomfort, Neck pain, Back pain
- **Diagnosis**: Whiplash injury
- **Sentiment**: Reassured
- **SOAP Note**: Structured clinical summary

---

## 📈 Performance Optimization

- **Caching**: Uses `@st.cache_resource` and `@lru_cache` to minimize redundant computations.
- **Memory Management**: Monitors memory usage with `psutil` and frees memory with garbage collection (`gc`).
- **Batching**: Processes large datasets in configurable batches to balance speed and resource usage.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.


