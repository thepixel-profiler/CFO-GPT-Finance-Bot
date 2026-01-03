import streamlit as st
import os
# Import BOTH functions now
from rag_engine import process_pdf_to_vector_db, answer_query

st.set_page_config(page_title="The CFO-GPT", page_icon="💰")

# Sidebar Logic
with st.sidebar:
    st.header("📂 Financial Docs")
    uploaded_file = st.file_uploader("Upload Report", type="pdf")
    
    if uploaded_file is not None:
        save_path = os.path.join("data_source", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("Analyze Document ⚡"):
            with st.spinner("Processing..."):
                count = process_pdf_to_vector_db(save_path)
                st.success(f"✅ Memory Created! ({count} chunks)")

# Main Chat Logic
st.title("💰 The CFO-GPT")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle New Question
prompt = st.chat_input("Ask about the report...")
if prompt:
    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # CALL THE BRAIN HERE 🧠
            final_answer = answer_query(prompt)
            st.markdown(final_answer)
            
    st.session_state.messages.append({"role": "assistant", "content": final_answer})