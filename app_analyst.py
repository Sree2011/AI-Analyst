import io
from dotenv import load_dotenv
import os
import base64
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from pydantic import BaseModel
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
 
 
 
# Initialize the ChatGoogleGenerativeAI object with the following parameters:
 
# Args:
#     model (str): Specifies the model to be used, in this case, "gemini-2.0-flash".
#     google_api_key (str): The API key required to authenticate and access the Google Generative AI services.
#     streaming (bool): A boolean flag indicating whether to enable streaming responses.
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key = "AIzaSyDrmiz2LgB8lWR1T3OmJM9kp9VnrUFIr50",
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
 
 
def normalize_and_plot():
    df = st.session_state.memory["file_data"]
 
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns
 
    # Normalize numeric columns
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(df[numeric_cols])
    df_normalized = pd.DataFrame(normalized_values, columns=[f"{col}_normalized" for col in numeric_cols])
 
    # Merge normalized columns back to original dataframe
    df_combined = pd.concat([df.reset_index(drop=True), df_normalized], axis=1)
 
    # Save normalized + original dataframe to session state
    st.session_state.memory["normalized_full_data"] = df_combined
 
    # Display preview
    st.subheader("Original + Normalized Data Preview")
    st.dataframe(df_combined.head())
 
    # Plot: Only normalized columns to avoid clutter
    st.subheader("Plot of Normalized Columns")
    fig, ax = plt.subplots()
    df_combined.plot(ax=ax, legend=True, alpha=0.8)
    st.pyplot(fig)
 
    return df_combined
 
def visualise_data():
    df = st.session_state.memory["file_data"]
    columns = df.columns
 
    for i in columns:
        for j in columns:
            if i != j:
                st.write(f"Crosstab between **{i}** and **{j}**")
                ct = pd.crosstab(df[i], df[j], rownames=[i], colnames=[j])
                fig, ax = plt.subplots()
                ct.plot(kind='line', ax=ax)
 
                # Show plot in Streamlit
                st.pyplot(fig)
 
                df_summary = ct.describe().to_string()
 
                prompt = f"""
                You are a data analyst. Explain {ct} in simple, plain language.
 
                Here is a summary of the data used for the plot:
                {df_summary}
 
                Also infer trends, patterns, or key insights from the plot.
                """
 
                # Generate the response from LLM
                response = "".join(chunk.content for chunk in llm.stream(prompt)).strip()
 
                # Display the response in Streamlit
                st.write(response)
 
 
def inference_data():
    df = st.session_state.memory["file_data"]
    df_summary = df.describe().to_string()
 
    prompt = f"""
    You are a data analyst. Explain the dataset in simple, plain language.
 
    Here is a summary of the data used for the plot:
    {df_summary}
 
    Also infer trends, patterns, or key insights from the plot.
    """
 
    # Generate the response from LLM
    response = "".join(chunk.content for chunk in llm.stream(prompt)).strip()
 
    # Display the response in Streamlit
    st.write(response)
 
def balance_analysis():
    df = st.session_state.memory["file_data"]
 
    st.subheader("Balance Analysis")
 
    # Let user choose a target column
    target_col = st.selectbox("Select Target Column for Balance Analysis", df.columns)
 
    if df[target_col].dtype == "object" or df[target_col].nunique() <= 20:
        balance_counts = df[target_col].value_counts()
 
        fig, ax = plt.subplots()
        balance_counts.plot(kind="bar", ax=ax)
        plt.title(f"Balance of Target Variable: {target_col}")
        st.pyplot(fig)
 
        st.write("Distribution of target values:")
        st.write(balance_counts.to_frame())
    else:
        st.write("Selected column does not look like a classification target.")
 
def train_predict_model():
    df = st.session_state.memory["file_data"]
 
    st.subheader("Train & Predict Using ML Models")
 
    target_col = st.selectbox("Select Target Column", df.columns)
    model_name = st.selectbox("Select ML Model", ["Random Forest", "Logistic Regression", "XGBoost"])
 
    # Basic input validation
    if df[target_col].nunique() < 2:
        st.write("Target must have at least two classes.")
        return
 
    X = df.drop(columns=[target_col])
    y = df[target_col]
 
    # Handle categorical features
    X = pd.get_dummies(X)
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    if model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))
 
    st.write("### Accuracy Score")
    st.write(accuracy_score(y_test, y_pred))
 
    st.write("### Sample Predictions")
    prediction_df = pd.DataFrame({ "Actual": y_test, "Predicted": y_pred })
    st.write(prediction_df.head(10))
 
 
 
def load_data():
    """Loads the uploaded file data and displays the file information.
    Args:
        None
    Returns:
        str: A message indicating the success or failure of the data loading.
    """
    df = st.session_state.memory["file_data"]
    if df is not None:
        return f"File '{st.session_state.memory['file_name']}' loaded successfully."+ f"Columns: {df.columns}"
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
 
 
def find_targets():
 
    data = st.session_state.memory["file_data"]
 
    # Threshold to consider a column as a potential target
    threshold = 2  # For example, columns with <=10 unique values
 
    # Identify candidate target columns
    candidate_targets = [
        column for column in data.columns if data[column].nunique() <= threshold
    ]
 
    # Display candidate target columns
    print("Candidate target columns based on unique values:")
    print(candidate_targets)
 
# Analyze further if necessary
 
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
elif "compare" in query.lower():
    response = find_targets()
elif "plot" in query.lower():
    response = plot_trends()
elif "visualise" in query.lower():
    response = visualise_data()
elif "infer" in query.lower():
    response = inference_data()
elif "balance" in query.lower():
    response = balance_analysis()
elif "predict" in query.lower():
    response = train_predict_model()
else:
    response = "".join(chunk.content for chunk in llm.stream(query)).strip().lower()
 
 
# --- Query Processing ---
if submission:
    state = LLMState(query=query, memory=st.session_state.memory)
    st.session_state.memory["chat_history"].append({"user": state.query, "bot": response})
    st.write(f"User: {state.query}")
    st.write(f"AI: {response}")
 
st.write("Chat History:", st.session_state.memory["chat_history"])