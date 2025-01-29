import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import plotly.express as px

class BaseAgent(ABC):
    """Base agent class with common functionality"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model_name
        self.role_context = ""
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.history: List[Dict[str, str]] = []
    
    @abstractmethod
    def get_role_prompt(self) -> str:
        """Return the role-specific prompt"""
        pass
    
    def chat(self, message: str) -> str:
        try:
            full_prompt = f"{self.role_context}\nUser Query: {message}"
            chat = self.model.start_chat(history=self.history)
            response = chat.send_message(full_prompt)
            
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": response.text})
            
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

class DataScientistAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.role_context = self.get_role_prompt()
        
    def get_role_prompt(self) -> str:
        return """You are an expert Data Scientist with the following capabilities:
        1. Data Analysis: Proficient in exploratory data analysis, statistical analysis, and hypothesis testing
        2. Machine Learning: Experienced in supervised and unsupervised learning techniques
        3. Data Visualization: Expert in creating insightful visualizations and dashboards
        4. Feature Engineering: Skilled in creating and selecting meaningful features
        5. Model Evaluation: Proficient in model performance metrics and validation techniques"""
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform quick EDA on a dataset"""
        analysis = {}
        
        # Basic statistics
        analysis['shape'] = df.shape
        analysis['missing_values'] = df.isnull().sum().to_dict()
        analysis['dtypes'] = df.dtypes.to_dict()
        analysis['summary'] = df.describe()
        
        # Additional analysis for numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            analysis['correlations'] = df[num_cols].corr()
        
        return analysis

class MLEngineerAgent(BaseAgent):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.role_context = self.get_role_prompt()
    
    def get_role_prompt(self) -> str:
        return """You are an expert Machine Learning Engineer with the following capabilities:
        1. Model Development: Proficient in designing and implementing ML models
        2. MLOps: Expert in ML deployment, monitoring, and maintenance
        3. Performance Optimization: Skilled in model optimization and scalability
        4. Production Systems: Experienced in building production-ready ML systems
        5. Best Practices: Knowledge of ML engineering best practices and design patterns"""
    
    def get_deployment_checklist(self) -> List[str]:
        return [
            "Model versioning configured",
            "Performance metrics monitoring set up",
            "API endpoints documented",
            "Error handling implemented",
            "Load testing completed",
            "Rollback strategy defined",
            "Data validation pipeline configured",
            "Monitoring alerts set up"
        ]

def main():
    st.set_page_config(page_title="AI Agent Interface", layout="wide")
    
    # Sidebar for API key input
    with st.sidebar:
        st.title("Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        
        if not api_key:
            st.warning("Please enter your API key to continue")
            return
        
        agent_type = st.radio(
            "Select Agent Type",
            ["Data Scientist", "ML Engineer"]
        )
    
    # Initialize the selected agent
    try:
        if agent_type == "Data Scientist":
            agent = DataScientistAgent(api_key)
            st.title("🔬 Data Scientist Agent")
        else:
            agent = MLEngineerAgent(api_key)
            st.title("⚙️ ML Engineer Agent")
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return
    
    # Main interface
    tab1, tab2 = st.tabs(["Chat Interface", "Special Tools"])
    
    # Chat Interface
    with tab1:
        st.subheader("Chat with your AI Agent")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                response = agent.chat(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Special Tools
    with tab2:
        if agent_type == "Data Scientist":
            st.subheader("Data Analysis Tools")
            
            # File uploader
            uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                # Display dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
                
                # Analyze dataset
                analysis = agent.analyze_dataset(df)
                
                # Display analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Basic Information")
                    st.write(f"Shape: {analysis['shape']}")
                    st.write("Missing Values:")
                    st.write(analysis['missing_values'])
                
                with col2:
                    st.subheader("Summary Statistics")
                    st.write(analysis['summary'])
                
                # Correlation heatmap
                if 'correlations' in analysis:
                    st.subheader("Correlation Heatmap")
                    fig = px.imshow(analysis['correlations'],
                                  labels=dict(color="Correlation"),
                                  color_continuous_scale="RdBu")
                    st.plotly_chart(fig)
        
        else:  # ML Engineer
            st.subheader("ML Engineering Tools")
            
            # Display deployment checklist
            st.write("### Deployment Checklist")
            checklist = agent.get_deployment_checklist()
            for item in checklist:
                st.checkbox(item, key=item)
            
            # Add more ML engineering tools here
            st.write("### Model Monitoring Setup")
            monitoring_metric = st.selectbox(
                "Select Primary Monitoring Metric",
                ["Accuracy", "F1 Score", "AUC-ROC", "MAE", "RMSE"]
            )
            
            alert_threshold = st.slider(
                "Alert Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05
            )
            
            if st.button("Configure Monitoring"):
                st.success("Monitoring configuration saved!")

if __name__ == "__main__":
    main()