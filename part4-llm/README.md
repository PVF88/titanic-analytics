# Part 4 — LLM + Documents (Local RAG Q&A)

## Goal
Given local documents (policies/FAQs/articles), answer questions by:
1. Converting documents to chunks
2. Creating embeddings
3. Retrieving top relevant chunks
4. Prompting an LLM with retrieved context

## Provided Docs
- docs/policy_1.txt
- docs/policy_2.txt
- docs/faq.txt

## App
`app.py` runs a Q&A interface:
- user enters a question
- system retrieves relevant chunks
- system generates an answer using retrieved context

## Run
```bash
cd part4-llm
pip install -r requirements.txt
streamlit run app.py
```

## Architecture
- Chunking: simple word-based splitting
- Keyword indexing: fast search without embeddings
- Retrieval: keyword-based (top-k by relevance)
- Response caching: using Streamlit session state
- Data statistics: cached and displayed

## Limitations
- Keyword-based search (not semantic embeddings for simplicity)
- Simulated LLM responses (easily replaceable with OpenAI API)
- Works best when questions match document phrasing
