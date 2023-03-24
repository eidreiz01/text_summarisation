# Import required libraries
import streamlit as st
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from gensim.utils import simple_preprocess

st.set_page_config(page_title="Abstractive News Summarization Dashboard", page_icon=":newspaper:")

st.title("Abstractive News Summarization Dashboard")
st.markdown("Welcome to the Abstractive News Summarization Dashboard. Please provide news text to be summarized below:")

# load model
model = AutoModelForSeq2SeqLM.from_pretrained("results\checkpoint-6000")
tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=2000)

def summarize_text(input_text):
    device = model.device
    tokenized = tokenizer([input_text], truncation =True, padding ='longest',return_tensors='pt')
    tokenized_dictionary = {k: v.to(device) for k, v in tokenized.items()}
    tokenized_result = model.generate(**tokenized_dictionary, max_length=128)
    tokenized_result = tokenized_result.to('cpu')
    predicted_summary = tokenizer.decode(tokenized_result[0])
    predicted_summary = predicted_summary.replace("pad", "")
    predicted_summary = (" ".join(simple_preprocess(predicted_summary))).title()
    return predicted_summary

# Define the Streamlit app
def main():

    # Input text area for the user
    input_text = st.text_area("Enter your news text here:", height=200)

    # Summarize button
    if st.button("Summarize"):
        if input_text.strip() != "":
            with st.spinner("Summarizing the news text..."):
                summary = summarize_text(input_text)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.error("Please enter some news text before clicking the 'Summarize' button.")

    st.markdown("---")
    st.markdown("© 2023 Abstractive News Summarization Dashboard")

if __name__ == "__main__":
    main()
