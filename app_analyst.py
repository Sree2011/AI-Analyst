import io
import os
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel
from PIL import Image


# Initialize the ChatGoogleGenerativeAI object with the following parameters:

# Args:
#     model (str): Specifies the model to be used, in this case, "gemini-2.0-flash".
#     google_api_key (str): The API key required to authenticate and access the Google Generative AI services.
#     streaming (bool): A boolean flag indicating whether to enable streaming responses.

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    streaming=True,
)

#
# Memory Log for Conversation & File Data


if "memory" not in st.session_state:
    st.session_state.memory = {"chat_history": [], "file_data": None, "file_name": None}

# --- Shared State Definition ---
class LLMState(BaseModel):
    """Represents the state of the Language Learning Model (LLM).

    Attributes:
        query (str): The user's query.
        response (str): The response from the LLM.
        file (Optional[Any]): The file associated with the query, if any.
        memory (Dict[str, Any]): The memory state, including chat history and file data.
    """
    query: str
    response: str = ""
    #agent: str = "general_ai_agent"
    file: Optional[Any] = None
    memory: Dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}

def upload_file():
    """Uploads a file and stores the data in the session state memory.
    Args:
        None
    Returns: 
        str: A message indicating the success or failure of the file upload.
    """
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx","json","xml"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file,sep=";")
        elif uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith("json"):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith("xml"):
            df = pd.read_xml(uploaded_file,parser="etree")
        
        st.session_state.memory["file_data"] = df
        st.session_state.memory["file_name"] = uploaded_file.name
        return f"File '{uploaded_file.name}' uploaded successfully."

def plot_heatmap():
    """Generates a heatmap plot for the correlation matrix of the uploaded file data."
    Args:
        None
    Returns:
        str: A message indicating the success or failure of the plot generation
    """
    if st.session_state.memory["file_data"] is not None:
        st.write("Generating heatmap...")
        if st.session_state.memory["file_data"].select_dtypes(include=["number"]).empty:
            return "No numerical columns available for correlation analysis."
        else:
            hd = st.session_state.memory["file_data"].select_dtypes(include=["number"])
            fig, ax = plt.subplots()
            sns.heatmap(hd.corr(), annot=True, ax=ax)
            # st.pyplot(fig)
            return st.pyplot(fig)
    else:
        return "No file data available for plotting."
    
def plot_trends():
    """Generates a trends plot for the uploaded file data.""
    Args:
        None"
    Returns:
        str: A message indicating the success or failure of the plot generation"
    """
    if st.session_state.memory["file_data"] is not None:
        st.write("Generating trends plot...")
        fig, ax = plt.subplots()
        st.session_state.memory["file_data"].plot(ax=ax)
        st.pyplot(fig)
        return "Here's a plot for the data trends."
    else:
        return "No file data available for plotting."


def load_data():
    """Loads the uploaded file data and displays the file information.
    Args:
        None
    Returns:
        str: A message indicating the success or failure of the data loading.
    """
    if st.session_state.memory["file_data"] is not None:
        return f"File '{st.session_state.memory['file_name']}' loaded successfully."
    else:
        return "No file data available. Please upload a file."

def summarise_data():
    """Summarizes the uploaded file data.
    Args:
        None
    Returns:
        str: A message containing the summary statistics of the data.
    """
    if st.session_state.memory["file_data"] is not None:
        return st.table(st.session_state.memory["file_data"].describe())
    else:
        return "No file data available for summarization."

# --------------------------------------------
# Main Application
# --------------------------------------------

st.title("AI Analyst Assistant")
st.write("This AI assistant can help you with data analysis, visualization, and more!")

upload_file()
query = st.text_input("Ask me anything:", key="query")
submission = st.button("Submit")
# --- User Query Input ---

if "load" in query.lower():
    response = load_data()
elif "summarise" in query.lower():
    response = summarise_data()
elif "heatmap" in query.lower():
    response = plot_heatmap()
elif "plot" in query.lower():
    response = plot_trends()

else:
    response = "".join(chunk.content for chunk in llm.stream(query)).strip().lower()


# --- Query Processing ---
if submission:
    state = LLMState(query=query, memory=st.session_state.memory)
    st.session_state.memory["chat_history"].append({"user": state.query, "bot": response})
    st.write(f"User: {state.query}")
    st.write(f"AI: {response}")

st.write("Chat History:", st.session_state.memory["chat_history"])