import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import pandas as pd

# Set page config
st.set_page_config(page_title="Langchain Agent", layout="wide")

# Role definitions
ROLES = {
    "Data Scientist": """You are an expert Data Scientist who helps with:
    - Data Analysis and Statistics
    - Machine Learning Models
    - Feature Engineering
    - Data Visualization
    - Model Evaluation""",
    
    "ML Engineer": """You are an expert ML Engineer who helps with:
    - Model Development and Deployment
    - MLOps and Production Systems
    - Performance Optimization
    - System Architecture
    - Best Practices"""
}

def initialize_agent(api_key, role):
    """Initialize LangChain agent with selected role"""
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
    
    memory = ConversationBufferMemory()
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Set the role context
    chain.predict(input=f"From now on, {ROLES[role]}")
    return chain

def main():
    # Sidebar
    with st.sidebar:
        st.title("AI Assistant Setup")
        api_key = st.text_input("Google API Key", type="password")
        role = st.selectbox("Select Role", list(ROLES.keys()))
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
    
    # Main chat interface
    st.title(f"💬 {role} Assistant")
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize agent
    if api_key:
        if "agent" not in st.session_state:
            st.session_state.agent = initialize_agent(api_key, role)
    else:
        st.warning("Please enter your API key to start")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.predict(input=prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Simple data analysis tool for Data Scientist role
    if role == "Data Scientist":
        st.divider()
        st.subheader("📊 Quick Data Analysis")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:", df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Basic Info:")
                st.write(f"- Rows: {df.shape[0]}")
                st.write(f"- Columns: {df.shape[1]}")
                st.write("- Missing Values:", df.isnull().sum().sum())
            
            with col2:
                st.write("Data Types:")
                st.write(df.dtypes)

if __name__ == "__main__":
    main()