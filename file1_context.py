import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from PIL import Image

# Load pre-trained BERT model and tokenizer for Question Answering
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=500)
    with torch.no_grad():
        outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx]))
    return answer if answer.strip() else "I couldn't find an answer. Try rephrasing the question!"

# Streamlit UI Configuration
st.set_page_config(page_title="BERT QA", page_icon="📖", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        .main {background-color: #f0f2f6;}
        h1 {color: #4A90E2; text-align: center;}
        .stTextArea, .stTextInput, .stButton {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Logo
st.image("https://upload.wikimedia.org/wikipedia/commons/2/23/BERT_logo.svg", width=200)
st.title("📖 BERT-Based Question Answering")
st.markdown("Enter a context passage and ask a question to get an answer from BERT.")

# Input Fields
context = st.text_area("📜 Enter Context:", height=200, placeholder="Paste your context here...")
question = st.text_input("❓ Enter Your Question:", placeholder="Type your question here...")

# Button and Answer Display
if st.button("💡 Get Answer"):
    if context and question:
        answer = get_answer(question, context)
        st.success("**Answer:** " + answer)
    elif not context:
        st.warning("⚠️ Please enter a context.")
    else:
        st.warning("⚠️ Please enter a question.")