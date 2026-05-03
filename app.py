import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import PyPDF2

# ---------------- MODEL LOADING ---------------- #

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# ---------------- FUNCTIONS ---------------- #

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def get_answer(question, context):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)

    # ❗ Fix: invalid span check
    if end_idx < start_idx:
        return "I couldn't find a proper answer."

    # ❗ Fix: confidence check
    score = torch.max(start_scores) + torch.max(end_scores)
    if score < 0:   # threshold (can tune)
        return "Answer not confident enough."

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_idx:end_idx + 1],
        skip_special_tokens=True
    )

    return answer.strip()
# ---------------- UI CONFIG ---------------- #

st.set_page_config(page_title="Smart PDF QA", page_icon="📄", layout="wide")

# ---------------- CLEAN CSS ---------------- #


st.markdown("""
<style>

/* Dark background */
body {
    background-color: #0f172a;
    color: white;
}

/* Header */
.header h1 {
    text-align: center;
    margin-top: 30px;
    margin-bottom: 20px;
    
}
.header p {
    color: #cbd5f5;
    text-align: center;
}

/* Labels (like Ask your question) */
label {
    color: white !important;
    font-weight: 500;
}

/* Input box text */
input, textarea {
    color: white !important;
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
}

/* Placeholder text */
input::placeholder {
    color: #94a3b8 !important;
}

/* Button */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-size: 16px;
    border: none;
}

/* Answer box */
.answer {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #38bdf8;
    font-size: 16px;
    color: white;
}

/* Expander text */
.streamlit-expanderHeader {
    color: white !important;
}

</style>

""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown("""
<div class="header">
    <h1>📄 Smart PDF QA</h1>
    <p>Upload your PDF and get instant answers using BERT 🤖</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- MAIN LAYOUT ---------------- #

col1, col2 = st.columns([1, 2])

# Upload section
with col1:
    st.subheader("📂 Upload PDF")
    pdf_file = st.file_uploader("Select a file", type=["pdf"])

# Q&A section
with col2:
    if pdf_file:
        context = extract_text_from_pdf(pdf_file)

        st.success("✅ PDF loaded successfully")

        with st.expander("📜 Show extracted text"):
            st.text_area("", context, height=250)

        # FORM (Enter key support)
        with st.form(key="qa_form"):
            question = st.text_input("❓ Ask your question")
            submit = st.form_submit_button("🚀 Get Answer")

        if submit:
            if question:
                with st.spinner("🤖 Thinking..."):
                    answer = get_answer(question, context)

                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a question")

# ---------------- FOOTER ---------------- #

st.markdown("---")
st.markdown("<p style='text-align:center;color:#64748b;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)