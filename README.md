# Graduate Programme Python Assessment

This project is a robust, modular Python application for document and timesheet retrieval, prompt engineering, and vector database management. It features:

- **Document Ingestion & Retrieval:**

  - Supports `.txt`, `.pdf`, and `.docx` files.
  - Uses FAISS and HuggingFace embeddings for semantic search.
  - Metadata and similarity scores are included in retrieval results.

- **Timesheet Querying:**

  - Dynamic filter construction for timesheet records.
  - Returns results as JSON arrays suitable for DataFrame display.

- **Prompt Engineering:**

  - Generates context-aware system prompts for downstream AI models.
  - Adapts to different prompt types (e.g., document/document, timesheet/timesheet).

- **Streamlit Interface:**
  - Chat and document Q&A modes.
  - DataFrame display for structured results.
  - Session management and inactivity timeout.

## Setup

1. **Clone the repository**

2. **Install dependencies**

   - Create a virtual environment (recommended):
     ```sh
     python -m venv pdenv
     source pdenv/bin/activate  # On Windows: pdenv\Scripts\activate
     ```
   - Install required packages:
     ```sh
     pip install -r requirements.txt
     ```
   - For document ingestion, you may also need:
     ```sh
     pip install pdfminer.six pi-heif unstructured-inference
     ```

3. **Run the Streamlit app**
   ```sh
   streamlit run scripts/interface.py
   ```

## Project Structure

- `scripts/` — Main Python modules (agent, retriever, database, interface, etc.)
- `docs/` — Prompt templates, sample documents
- `data/faiss_index/` — Vector database indexes
- `notebooks/` — Jupyter notebooks for experimentation
- `logs/` — Log files
- `requirements.txt` — Python dependencies

## Usage

- **PD Chat:**

  - Ask timesheet-related questions in natural language.
  - Results are shown as a DataFrame.

- **Document Q&A:**
  - Upload a document and ask questions about its content.
  - Answers include relevant metadata and similarity scores.

## Notes

- All responses are designed for easy DataFrame conversion and downstream analysis.
- Prompts and tool instructions are customizable in `docs/prompts.txt`.

## Troubleshooting

- If you encounter missing dependency errors for document loaders, install the suggested packages above.
- For any issues, check the log files in the `logs/` directory.

---

© 2025 Graduate Programme Python Assessment
