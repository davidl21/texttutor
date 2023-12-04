import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        # PdfReader object with pages variable containing pages of text...
        pdf_reader = PdfReader(pdf)

        all_pages = pdf_reader.pages
        for page in all_pages:
            text += page.extract_text()
    
    return text

def get_vector_store(text_chunks):
    """ Gets text embeddings for text chunks using Instructor model"""
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_text_chunks(raw_text):
    # Use LangChain character text splitter to split raw pdf text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )

    # list of chunks with above specifications
    chunks_array = text_splitter.split_text(raw_text)
    return chunks_array

def get_convo_chain(vector_store):
    """ Returns a conversation chain utilizing LangChain with Flan-t5 LLM"""
    # uses Google Flan-t5 xxl model
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # conversation chian that works with memory, initialize memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return convo_chain

def handle_user_input(user_input):
    # utilize persistent conversation chain variable
    # returns a list, containing response and chat history
    if not st.session_state.convo_chain:
        st.write("Please upload a PDF first!")
    else:
        response = st.session_state.convo_chain({'question': user_input})
            # format LangChain chat history object
            # must make chat_history persistent
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    #st.write(response)

def main():
    st.set_page_config(page_title="TextTutor")

    # Front-end Streamlit setup
    st.header("Talk with your textbooks and PDFs.")
    user_input = st.text_input("Ask TextTutor...")

    # only triggered if user submits a question
    if user_input:
        handle_user_input(user_input)

    st.write(css, unsafe_allow_html=True)

    # loads API keys for LangChain access
    load_dotenv()

    # streamlit variable consistency. must initialize at start of program
    if "convo_chain" not in st.session_state:
        st.session_state.convo_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.convo_chain = None
    
    # st.write(user_template.replace("{{MSG}}", "Hello, TextTutor!"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello!"), unsafe_allow_html=True)

    # File upload sidebar
    with st.sidebar:
        st.subheader("Import your documents")
        pdf_files = st.file_uploader(
            "Upload your texts or PDFs", accept_multiple_files=True)
        
        # button becomes True only if user clicks on it.
        if st.button("Upload"):
            # use a Streamlit spinner for better UI
            with st.spinner("Processing may take a few minutes"):
                # get pdf text
                raw_text = get_pdf_text(pdf_files)

                # divide text into chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vector_store = get_vector_store(text_chunks)

                # LangChain conversation chain
                # it is important to use st.session_state since st will reload variables 
                # when interacted with.... If converting to Flask note this.
                st.session_state.convo_chain = get_convo_chain(vector_store)


if __name__ == '__main__':
    main()