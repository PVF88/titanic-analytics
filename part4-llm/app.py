"""
PART 4 — LLM + Documents (Simple Retrieval-Augmented Q&A)
Streamlit app that answers questions about Titanic data and policies using LLM.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import time

# ISSUE 1: No session state caching for LLM or documents
# Simulated LLM function (would use OpenAI API in production)
def query_llm(prompt):
    """Simulate LLM query - in production would call OpenAI API"""
    # ISSUE 2: Simulating slow LLM call without rate limiting or caching
    time.sleep(0.5)  # Simulate API latency
    return f"Response to: {prompt[:50]}..."

# ISSUE 3: Loading documents on every page reload
def load_documents():
    """Load policy documents from disk"""
    print("Loading documents...")
    docs = {}
    doc_dir = 'docs'
    
    if not os.path.exists(doc_dir):
        return docs
    
    # ISSUE 4: Reading files individually in a loop
    for file in os.listdir(doc_dir):
        if file.endswith('.txt'):
            with open(os.path.join(doc_dir, file), 'r') as f:
                content = f.read()
                # ISSUE 5: Not caching or indexing document content
                docs[file] = content
    
    return docs

# ISSUE 6: Inefficient document search (linear scan every query)
def search_documents(documents, query):
    """Search documents for relevant content"""
    
    # ISSUE 7: Full-text search with string matching (no indexing)
    results = []
    for doc_name, content in documents.items():
        # ISSUE 8: Expensive string search for every query
        if query.lower() in content.lower():
            # ISSUE 9: Returning full document instead of relevant snippets
            results.append({
                'document': doc_name,
                'content': content,
                'relevance': 1.0
            })
    
    return results

# ISSUE 10: Building context inefficiently
def build_context(documents, query):
    """Build LLM context from documents"""
    
    # ISSUE 11: Searching for every query even if similar query was made
    search_results = search_documents(documents, query)
    
    # ISSUE 12: Concatenating entire document content into context
    context = "Relevant documents:\n"
    for result in search_results:
        # ISSUE 13: Passing full document content instead of summarized snippet
        context += f"\n{result['document']}:\n{result['content']}\n"
    
    return context

# ISSUE 14: Loading data repeatedly
def load_data():
    """Load Titanic data"""
    df = pd.read_csv('../part1-data-cleaning/data_clean/titanic_clean.csv')
    return df

def main():
    st.set_page_config(page_title="Titanic LLM Q&A", layout="wide")
    st.title("🚢 Titanic LLM Q&A with Documents")
    
    # ISSUE 15: No session state initialization
    # If session_state was used, documents would persist
    
    # ISSUE 16: Loading documents on every page load
    documents = load_documents()
    st.write(f"Loaded {len(documents)} documents")
    
    # ISSUE 17: Loading data on every page load
    df = load_data()
    st.write(f"Data shape: {df.shape}")
    
    # ISSUE 18: No query history or caching
    st.header("Ask Questions")
    
    query = st.text_input("Enter your question:")
    
    if query:
        # ISSUE 19: Rebuilding context from scratch for every query
        context = build_context(documents, query)
        
        # ISSUE 20: Combining data context with document context inefficiently
        data_summary = f"Total records: {len(df)}, Survivors: {df['survived'].sum()}"
        
        # ISSUE 21: Full prompt building with redundant information
        full_prompt = f"""
        Question: {query}
        
        Data Summary: {data_summary}
        
        {context}
        
        Please answer based on the above information.
        """
        
        # ISSUE 22: Calling LLM every time without checking cache
        response = query_llm(full_prompt)
        
        st.write("### Answer")
        st.write(response)
        
        # ISSUE 23: Storing query results in local variable (lost on rerun)
        st.write(f"Query time: {time.time():.2f}s")
    
    # ISSUE 24: Tab-based interface but no optimization per tab
    st.header("Data Stats")
    
    tab1, tab2, tab3 = st.tabs(["Survival", "Age", "Fare"])
    
    with tab1:
        # ISSUE 25: Recalculating stats for each tab
        st.write(df['survived'].value_counts())
    
    with tab2:
        st.write(df['age'].describe())
    
    with tab3:
        st.write(df['fare'].describe())

if __name__ == "__main__":
    main()
