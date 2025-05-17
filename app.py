import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
from glob import glob

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Constants
os.environ["GOOGLE_API_KEY"] = "AIzaSyBmUYQdImYbjPJesYFoMHVEfibp5l1CKBc"  # Replace with your real key
DB_PATH = "incident_faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize vector DB if available
def initialize_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if os.path.exists(DB_PATH):
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store, embeddings
    else:
        st.error("Vector database not found. Please run the `build_vectordb()` script first.")
        st.stop()

# Initialize RAG chain
@st.cache_resource(show_spinner=False)
def initialize_rag():
    vector_store, _ = initialize_vector_db()
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3),
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

# Safety functions
def analyze_video_feed():
    return ["No helmet detected near Furnace 3", "Unauthorized entry detected in Zone B"]

def check_sensor_data(sensor_df):
    alerts = []
    if sensor_df['gas_level'].iloc[-1] > 300:
        alerts.append("High gas level detected")
    if sensor_df['temperature'].iloc[-1] > 80:
        alerts.append("High temperature in Boiler Room")
    if sensor_df['noise_level'].iloc[-1] > 85:
        alerts.append("Noise level exceeds safety threshold")
    return alerts

def generate_prevention_checklist():
    return ["Wear helmet and safety gear", "Check gas detector calibration", "Inspect fire extinguishers"]

def generate_compliance_report():
    return "Safety compliance is at 92% this month. Helmet violations decreased by 15%."

# RAG search
qa_chain = initialize_rag()
def retrieve_similar_incidents(query):
    try:
        return [qa_chain.run(query)]
    except AssertionError as e:
        st.error("Embedding dimension mismatch. Please rebuild the vector DB.")
        st.stop()

# UI Layout
st.set_page_config(page_title="AI-Powered Safety Monitoring", layout="wide")
st.title("AI-Powered Industrial Safety Monitoring")

st.sidebar.header("Control Panel")
shift_start = st.sidebar.time_input("Shift Start Time", value=datetime.time(8, 0))
selected_area = st.sidebar.selectbox("Select Area", ["Furnace", "Boiler Room", "Assembly Line"])

st.header("Real-time Hazard Alerts")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Video Surveillance Alerts")
    for alert in analyze_video_feed():
        st.error(f"📹 {alert}")

with col2:
    st.subheader("Sensor Alerts")
    sensor_df = pd.DataFrame({
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='min'),
        'gas_level': np.random.randint(250, 350, 10),
        'temperature': np.random.randint(60, 90, 10),
        'noise_level': np.random.randint(70, 95, 10)
    })
    for alert in check_sensor_data(sensor_df):
        st.error(f"📊 {alert}")

st.header("Prevention Checklist")
for item in generate_prevention_checklist():
    st.checkbox(item, value=False)

st.header("Historical Incident Analysis (RAG)")
user_query = st.text_input("Describe current risk or incident", value="helmet violation near furnace")
if user_query:
    with st.spinner("Searching similar historical incidents..."):
        for res in retrieve_similar_incidents(user_query):
            st.info(res)

st.header("Safety Compliance Report")
st.success(generate_compliance_report())

st.header("Sensor Readings (Last 10 min)")
st.dataframe(sensor_df.set_index('timestamp'))

st.header("Add New Incident")
with st.form("incident_form"):
    incident_date = st.date_input("Incident Date")
    incident_desc = st.text_area("Incident Description")
    action_taken = st.text_area("Action Taken")

    if st.form_submit_button("Add to Knowledge Base"):
        new_incident = f"{incident_date}: {incident_desc} Action: {action_taken}"
        os.makedirs("incident_docs", exist_ok=True)
        incident_count = len(glob("incident_docs/*.txt")) + 1
        incident_path = f"incident_docs/incident_{incident_count}.txt"

        with open(incident_path, "w") as f:
            f.write(new_incident)

        # Load and embed new document
        loader = TextLoader(incident_path)
        new_doc = loader.load()
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        if os.path.exists(DB_PATH):
            vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(new_doc)
            vector_store.save_local(DB_PATH)
            st.success("Incident added to knowledge base!")
        else:
            st.error("Vector DB missing. Please rerun `build_vectordb()` to rebuild.")
