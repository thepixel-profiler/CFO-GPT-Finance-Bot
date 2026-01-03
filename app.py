import streamlit as st
import os
from rag_engine import process_pdf_to_vector_db, answer_query

# 1. Page Configuration
st.set_page_config(page_title="The CFO-GPT", page_icon="💰")

# 2. Sidebar: File Upload & Monetization
with st.sidebar:
    st.header("📂 Financial Docs")
    uploaded_file = st.file_uploader("Upload an Annual Report (PDF)", type="pdf")
    
    if uploaded_file is not None:
        # --- FIX: Create folder if it doesn't exist on the server ---
        if not os.path.exists("data_source"):
            os.makedirs("data_source")
        # ------------------------------------------------------------

        save_path = os.path.join("data_source", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File Uploaded: {uploaded_file.name}")
        
        if st.button("Analyze Document ⚡"):
            with st.spinner("Processing PDF... (Creating Embeddings)"):
                count = process_pdf_to_vector_db(save_path)
                st.write(f"✅ Processed & Saved {count} text chunks to FAISS.")
                st.info("Ready for Chat! (AI Memory Created)")
    
    # --- MONETIZATION SECTION ---
    st.markdown("---")
    st.header("🚀 Build This Yourself")
    st.write("Get the complete Source Code & Setup Guide.")
    # REPLACE THE LINK BELOW WITH YOUR REAL GUMROAD LINK
    st.link_button("📥 Download Starter Kit (₹299)", "https://framedmonochrome.gumroad.com/l/cfogptstarterkit") 
    st.info("Perfect for students building a portfolio.")
    # ----------------------------

# 3. Main Chat Interface
st.title("💰 The CFO-GPT")
st.markdown("Your **Private Financial Analyst**. Upload a report to begin.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
prompt = st.chat_input("Ask a question about the report...")
if prompt:
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Get AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_query(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


