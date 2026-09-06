# REC-Voice-AI-Receptionist

A **Voice-Based AI Receptionist for Smart College Communication** developed as a final-year project at **Rajalakshmi Engineering College**.

The system combines **Speech Recognition, Retrieval-Augmented Generation (RAG), Large Language Models, Text-to-Speech, and SIP-based Telephony** to automatically answer institutional queries through voice calls.

The project is designed in two phases:

- **Phase 1:** AI receptionist for publicly available college information.
- **Phase 2:** Agentic AI extension for authenticated student-specific queries, DigiCampus/API integration, and human receptionist escalation.

---

## Authors

1. **Akash N** - 230701019
2. **Kamalesh S P** - 230701138
3. **Subramani A** - 230701347
4. **Guru Raja T** - 230701502

**Department:** Computer Science and Engineering  
**Institution:** Rajalakshmi Engineering College

---

## Project Overview

College reception desks receive repetitive telephone inquiries related to admissions, academics, departments, faculty, placements, facilities, regulations, and other institutional information.

Handling all such queries manually can increase receptionist workload and make information delivery dependent on staff availability.

The **REC Voice AI Receptionist** provides a voice-based interface that can understand a caller's question, retrieve relevant institutional information, generate a grounded response, and communicate the answer back to the caller through speech.

The Phase 1 system operates completely on publicly available institutional information and does not access personal student records.

---

## Project Architecture

The implemented Phase 1 architecture is:

```text
MicroSIP / SIP Caller
        │
        ▼
Asterisk 22.10.1
        │
        ▼
AudioSocket
        │
        ▼
Python Audio Gateway
        │
        ▼
WebRTC Voice Activity Detection
        │
        ▼
Faster-Whisper STT
        │
        ▼
RAG Service
        │
        ▼
Sentence Transformer Embeddings
        │
        ▼
ChromaDB
        │
        ▼
Ollama
        │
        ▼
Qwen3:4B
        │
        ▼
Piper TTS
        │
        ▼
8 kHz PCM Audio
        │
        ▼
AudioSocket
        │
        ▼
Asterisk
        │
        ▼
MicroSIP / Caller
```

---

## Phase 1

Phase 1 implements a complete voice-based AI receptionist for answering publicly available institutional queries.

### Phase 1 Features

- SIP-based voice calling
- Automatic call handling using Asterisk
- Audio communication using Asterisk AudioSocket
- Voice Activity Detection using WebRTC VAD
- Speech-to-Text using Faster-Whisper
- Institutional knowledge retrieval using RAG
- Sentence Transformer embeddings
- ChromaDB vector database
- Locally hosted LLM using Ollama
- Qwen3:4B response generation
- Piper Text-to-Speech
- Context-grounded institutional responses
- Out-of-scope query refusal
- Multi-query evaluation
- Component-level latency evaluation

---

### Phase 1 Call Flow

When a caller interacts with the system:

1. The caller initiates a SIP call using MicroSIP.
2. Asterisk receives and answers the call.
3. Asterisk creates an AudioSocket connection with the Python audio gateway.
4. Incoming audio is transmitted as PCM audio.
5. WebRTC VAD detects active speech.
6. Faster-Whisper converts the speech into text.
7. The transcribed query is passed to the RAG pipeline.
8. Relevant institutional information is retrieved from ChromaDB.
9. Retrieved information is supplied as context to Qwen3:4B through Ollama.
10. The LLM generates a knowledge-grounded response.
11. Piper TTS converts the response into speech.
12. The generated audio is converted to telephony-compatible 8 kHz PCM.
13. AudioSocket sends the response back to Asterisk.
14. Asterisk plays the response to the caller.

---

## Technology Stack

| Component                | Technology                 |
| ------------------------ | -------------------------- |
| Programming Language     | Python                     |
| Telephony Server         | Asterisk 22.10.1           |
| Telephony Protocol       | SIP                        |
| Softphone                | MicroSIP                   |
| Audio Transport          | Asterisk AudioSocket       |
| Voice Activity Detection | WebRTC VAD                 |
| Speech-to-Text           | Faster-Whisper             |
| STT Model                | Whisper Small              |
| GPU Acceleration         | CUDA / float16             |
| Embeddings               | Sentence Transformers      |
| Vector Database          | ChromaDB                   |
| LLM Runtime              | Ollama                     |
| Large Language Model     | Qwen3:4B                   |
| Text-to-Speech           | Piper TTS                  |
| TTS Voice                | en_US-lessac-medium        |
| Web Scraping             | Requests + BeautifulSoup   |
| PDF Processing           | PyMuPDF                    |
| OCR                      | Tesseract OCR              |
| Evaluation               | Python, Pandas, Matplotlib |

---

## Knowledge Base

The Phase 1 knowledge base contains publicly available information from the official Rajalakshmi Engineering College website and selected institutional documents.

### Website Information

The ingestion pipeline includes relevant public information related to:

- About the College
- Academics
- Admissions
- Departments
- Faculty
- Placements
- Research
- Facilities
- Events
- Department-specific pages
- Other relevant institutional information

### PDF Documents

The current knowledge base includes:

1. Certificate for CGPA to Percentage Conversion
2. Certificate for Medium of Instruction
3. R2023 CSE Curriculum and Syllabus
4. REC-2023 UG Regulations

Scanned documents are processed using Tesseract OCR when machine-readable text is not directly available.

---

## Knowledge Ingestion Pipeline

```text
College Website + PDF Documents
              │
              ▼
     Content Extraction
              │
              ▼
      OCR where required
              │
              ▼
         Text Cleaning
              │
              ▼
           Chunking
              │
              ▼
 Sentence Transformer Embeddings
              │
              ▼
          ChromaDB
              │
              ▼
 Semantic Retrieval + Reranking
```

The final Phase 1 knowledge base contains 767 indexed knowledge chunks.

---

## Retrieval-Augmented Generation

The project uses Retrieval-Augmented Generation to reduce unsupported LLM responses.

For each user query:

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
ChromaDB Search
      │
      ▼
Candidate Knowledge Chunks
      │
      ▼
Relevance Reranking
      │
      ▼
Retrieved Context
      │
      ▼
Qwen3:4B
      │
      ▼
Grounded Response
```

The retriever uses semantic similarity together with additional relevance scoring for factual queries and course-related information.

Source information such as document name, URL, page number, and chunk information is retained where applicable.

---

## Speech-to-Text

The project uses Faster-Whisper for local speech recognition.

Current configuration:

- Model: `small`
- CUDA acceleration
- float16 inference
- Local processing

The use of a local STT model avoids dependency on an external speech-recognition API during Phase 1 development.

---

## Voice Activity Detection

WebRTC VAD is used to determine when the caller is actively speaking.

This allows the audio gateway to identify complete utterances before sending them to the speech-recognition pipeline.

---

## Large Language Model

The project uses:

```test
Ollama
   +
Qwen3:4B
```

The LLM runs locally during Phase 1.

Retrieved institutional information is supplied to the model as context so that responses remain aligned with the available college knowledge base.

---

## Text-to-Speech

The generated response is converted into speech using Piper TTS.

Current voice model:

```text
en_US-lessac-medium
```

The synthesized audio is converted into 8 kHz, mono, 16-bit PCM before being transmitted through AudioSocket to Asterisk.

The Piper model files are not stored in the Git repository and must be downloaded separately.

---

## Telephony Integration

The Phase 1 telephony system uses:

- Asterisk 22.10.1
- SIP
- MicroSIP
- AudioSocket
- Python socket programming

Asterisk handles the SIP call while the Python audio gateway processes the AI voice pipeline.

### Telephony Flow

```text
MicroSIP
   │
   ▼
Asterisk
   │
   ▼
AudioSocket TCP Connection
   │
   ▼
Python AI Backend
   │
   ▼
AudioSocket
   │
   ▼
Asterisk
   │
   ▼
MicroSIP
```

---

## Phase 1 Evaluation

The RAG system was evaluated using 20 test queries.

The test set consisted of:

- 18 normal institutional queries
- 2 negative/out-of-scope queries

### Evaluation Results

| Metric            | Result |
| ----------------- | ------ |
| Normal Queries    |     18 |
| Retrieval Correct |  18/18 |
| Hit@5             |   100% |
| Answer Correct    |  18/18 |
| Answer Accuracy   |   100% |
| Negative Queries  |      2 |
| Correct Refusals  |    2/2 |
| Refusal Accuracy  |   100% |

The system achieved **100% retrieval Hit@5, answer accuracy, and refusal accuracy on the defined Phase 1 evaluation dataset**.

These results represent performance on the evaluated test set and should not be interpreted as universal accuracy across all possible queries.

---

## Latency Evaluation

A separate evaluation measures the processing latency of the major AI components.

### Average Processing Latency

| Component               | Average Time |
| ----------------------- | ------------ |
| STT                     |       0.25 s |
| RAG                     |       0.07 s |
| LLM                     |      18.55 s |
| TTS                     |       2.46 s |
| **Total AI Processing** |  **21.35 s** |

The evaluation identifies **LLM generation as the primary computational bottleneck**.

Approximately **86.9**% of the measured AI processing latency is contributed by LLM generation.

The measurements were obtained using the current local development environment. Hardware acceleration, runtime optimization, model optimization, and cloud GPU deployment are possible future methods for reducing response latency.

The reported total represents AI processing from STT processing through TTS audio preparation and is not intended to represent complete caller-perceived telephony latency.

---

## Evaluation Outputs

The evaluation scripts generate CSV files and visualizations for analyzing system performance.

Typical results include:

```text
evaluation_results/
├── rag_evaluation_results.csv
├── evaluation_metrics.csv
├── evaluation_summary.txt
├── category_accuracy.png
├── overall_accuracy.png
├── question_wise_accuracy.png
├── voice_latency_results.csv
├── voice_latency_graph.png
└── voice_latency_average_graph.png
```

---

## Phase 1 Milestones

| Milestone | Description                                        | Status      |
|-----------|--------------------------------------------------- |-------------|
| M1        | Website and PDF ingestion pipeline with ChromaDB   | ✅ Completed|
| M2        | RAG backend + locally hosted LLM                   | ✅ Completed|
| M3        | STT → RAG → LLM → TTS pipeline                     | ✅ Completed|
| M4        | Asterisk/SIP/AudioSocket telephony integration     | ✅ Completed|
| M5        | Accuracy, retrieval, refusal and latency eval.     | ✅ Completed|
| M6        | Mentor review and approval of Phase 2 scope        | ⏳ Pending  |

---

## Phase 2 — Planned Extension

Phase 2 extends the existing Phase 1 architecture instead of replacing it.

The planned goal is to transform the public-information receptionist into an Agentic AI-based institutional receptionist capable of securely handling personalized student queries.

---

## Planned Phase 2 Architecture

```text
Caller
   │
   ▼
Existing Voice Pipeline
   │
   ▼
Query Understanding
   │
   ▼
Agentic Decision Layer
   │
   ├─────────────────────────────┐
   │                             │
   ▼                             ▼
Public Information          Personal Information
   │                             │
   ▼                             ▼
RAG Knowledge              Authentication
                                 │
                                 ▼
                        DigiCampus / API
                                 │
                                 ▼
                       Student Information
   │                             │
   └──────────────┬──────────────┘
                  │
                  ▼
              LLM Response
                  │
                  ▼
                 TTS
                  │
                  ▼
                Caller
```

---

## Planned Phase 2 Features

Phase 2 may include:

- Agentic AI decision making
- Tool/function calling
- Student authentication
- Student ID verification
- PIN/OTP/DTMF authentication
- DigiCampus or approved institutional API integration
- Subject-wise attendance retrieval
- Internal assessment marks
- Examination results
- Fee-payment status
- Outstanding dues
- Placement eligibility
- Intelligent query routing
- Human receptionist escalation
- Conversation summary generation
- Secure logging and audit support

These features are **planned and are not part of the completed Phase 1 implementation**.

---

## Agentic AI

The proposed Agentic AI layer will determine how a query should be handled.

Possible decisions include:

```text
Incoming Query
      │
      ▼
Agentic AI
      │
      ├──► Answer using RAG
      │
      ├──► Request authentication
      │
      ├──► Call institutional API
      │
      ├──► Ask for clarification
      │
      ├──► Refuse unsupported operation
      │
      └──► Escalate to human receptionist
```

The agent will operate within predefined institutional permissions and approved tools.

---

## Personalized Student Queries

Subject to institutional API access and approval, Phase 2 is intended to support queries such as:

- "What is my attendance percentage?"
- "What are my internal marks?"
- "Have my semester results been published?"
- "Is my fee payment complete?"
- "Am I eligible for the upcoming placement drive?"

Personal student information will not be stored in the public RAG knowledge base.

Instead, it should be retrieved securely from an approved institutional service after successful authentication.

---

## Planned Authentication

Possible authentication mechanisms include:

- Student ID
- PIN
- OTP
- DTMF keypad input
- Institution-approved identity verification

The final authentication mechanism depends on institutional security requirements and API access.

---

## Planned Human Escalation

The system may escalate a call when:

- the caller requests a human receptionist;
- the query requires administrative approval;
- the query requires human judgement;
- authentication fails;
- the requested information cannot be accessed;
- the system cannot reliably answer the query;
- the matter is outside the AI's defined scope.

Planned flow:

```text
AI identifies escalation
        │
        ▼
Inform Caller
        │
        ▼
Generate Query / Conversation Summary
        │
        ▼
Asterisk / SIP Call Transfer
        │
        ▼
Human Receptionist
```

---

## Phase 2 Milestones

| Milestone | Description                                         | Status  |
| --------- | --------------------------------------------------- | ------- |
| M7        | Caller authentication flow                          | Planned |
| M8        | DigiCampus/API integration                          | Planned |
| M9        | Agentic routing and human receptionist escalation   | Planned |
| M10       | Complete Phase 2 evaluation and final demonstration | Planned |

## Project Directory Structure

The project is organized into separate modules for ingestion, retrieval, speech processing, telephony, evaluation, models, and knowledge data.

```text
REC-Voice-AI-Receptionist/
│
├── asterisk/
│   ├── config/
│   ├── extensions/
│   └── scripts/
│
├── backend/
│   └── app/
│       ├── evaluation/
│       ├── ingestion/
│       ├── rag/
│       ├── telephony/
│       └── voice/
│
├── data/
│   └── raw/
│       ├── pdf/
│       └── web/
│
├── evaluation_results/
│
├── models/
│   └── tts/
│
├── vectorstore/
│
└── requirements.txt
```

Some generated directories such as the virtual environment, local vector database, temporary audio files, model binaries, and Python cache directories are excluded from Git.

---

## Main Modules

`backend/app/ingestion`

Responsible for building the institutional knowledge base.

Important components include:

```text
chunker.py
cleaner.py
pdf_parser.py
pipeline.py
web_scraper.py
```

---

`backend/app/rag`

Responsible for retrieval and response generation.

Important components include:

```text
embeddings.py
llm_service.py
prompt.py
rag_service.py
retriever.py
vector_store.py
```

---

`backend/app/voice`

Responsible for speech-processing services.

Important components include:

```text
stt_service.py
tts_service.py
cuda_setup.py
conversation_service.py
```

---

`backend/app/telephony`

Responsible for connecting the AI backend to Asterisk.

```text
audio_gateway.py
```

---

`backend/app/evaluation`

Contains evaluation scripts for RAG performance and voice-pipeline latency.

```text
evaluate_rag.py
evaluate_voice_latency.py
visualize_results.py
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kamalesh-Suresh-Kumar/REC-Voice-AI-Receptionist.git
cd REC-Voice-AI-Receptionist/REC-Voice-AI-Receptionist
```

---

### 2. Create a Python Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

### 3. Install Python Dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## External Requirements

Some components are not installed directly through `requirements.txt`.

### Ollama

Install Ollama separately and make sure the required LLM is available:

```powershell
ollama pull qwen3:4b
```

### Verify

```powershell
ollama list
```

---

## Tesseract OCR

Tesseract OCR is required for scanned PDF processing.

Ensure that Tesseract is installed locally and accessible to the PDF/OCR pipeline.

---

## Piper TTS Model

The Piper model files are intentionally excluded from Git.

Place the following files in:

```text
models/tts/
```

Required files:

```text
en_US-lessac-medium.onnx
en_US-lessac-medium.onnx.json
```

---

## Asterisk

Asterisk is required for the telephony layer.

The current Phase 1 environment uses:

```text
Asterisk 22.10.1
```

with SIP and AudioSocket support.

---

## MicroSIP

MicroSIP is used as the SIP softphone for Phase 1 development and testing.

---

## Build the Knowledge Base

From the `backend` directory:

```powershell
cd backend
$env:PYTHONPATH="$PWD"
python -m app.ingestion.pipeline
```

This processes the configured website and PDF sources and populates the local ChromaDB knowledge base.

---

## Run the Voice AI Gateway

From the `backend` directory:

```text
backend/
```

run:

```powershell
$env:PYTHONPATH="$PWD"; python .\app\telephony\audio_gateway.py
```

The Python gateway will load the required AI services and listen for the Asterisk AudioSocket connection.

---

## Run RAG Evaluation

From the `backend` directory:

```powershell
$env:PYTHONPATH="$PWD"; python .\app\evaluation\evaluate_rag.py
```

---

## Run Voice Latency Evaluation

From the `backend` directory:

```powershell
$env:PYTHONPATH="$PWD"; python .\app\evaluation\evaluate_voice_latency.py
```

The results are written to the evaluation results directory and can be used to analyze STT, RAG, LLM, TTS, and total processing latency.

---

## Current Project Status

```text
Phase 1
├── M1 Knowledge Ingestion          ✅
├── M2 RAG + LLM                    ✅
├── M3 Voice AI Pipeline            ✅
├── M4 Telephony Integration        ✅
├── M5 Evaluation                   ✅
└── M6 Mentor Review                ⏳

Phase 2
├── M7 Authentication               Planned
├── M8 DigiCampus/API               Planned
├── M9 Agentic AI + Escalation      Planned
└── M10 Final Evaluation            Planned
```

---

## Key Phase 1 Outcomes

The completed Phase 1 prototype demonstrates the integration of:

### Telephony + Speech Recognition + RAG + Local LLM + Text-to-Speech

into a single institutional voice AI system.

The current system can:

- receive SIP-based voice calls;
- detect caller speech;
- transcribe spoken questions;
- retrieve relevant college information;
- generate grounded responses;
- reject tested out-of-scope queries;
- synthesize spoken responses;
- return audio to the caller through Asterisk.

Phase 1 evaluation has also identified the LLM inference stage as the major performance bottleneck, providing a clear direction for future optimization.

---

## Future Improvements

Possible future improvements include:

- GPU/cloud deployment for reduced LLM latency
- model inference optimization
- streaming LLM generation
- streaming TTS
- improved speech recognition under noisy conditions
- multilingual or Tamil-English support
- automated knowledge-base updating
- improved conversation-state handling
- institutional API integration
- authenticated student services
- human receptionist escalation
- production SIP/PSTN deployment
- secure call logging and analytics

---

## SDG Alignment

The project aligns with the following United Nations Sustainable Development Goals:

### SDG 04 — Quality Education

Improves accessibility to institutional and academic information through an intelligent voice interface.

### SDG 09 — Industry, Innovation and Infrastructure

Applies AI, speech processing, RAG, and telecommunications technologies to build intelligent institutional communication infrastructure.

### SDG 16 — Peace, Justice and Strong Institutions

Supports consistent, transparent, and accessible delivery of institutional information.

---

## Disclaimer

This repository represents an academic final-year project prototype.

Phase 1 works only with publicly available institutional information.

Personalized student-data access, authentication, DigiCampus integration, Agentic AI tool use, and human receptionist escalation belong to the planned Phase 2 scope and must not be considered part of the completed Phase 1 implementation.
