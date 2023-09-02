from tempfile import NamedTemporaryFile
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
import os

load_dotenv()
# st.cache_data.clear() 
summary_data = ""
summarized_text_global = ""
llm = OpenAI(temperature=0)

@st.cache_data#(func_or_key="summarize_pdf", persist=True)
def summarize_pdf(pdf_file_path, custom_prompt=""):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    if custom_prompt!="":
        prompt_template = custom_prompt + """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        custom_summary = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
    else:
        custom_summary = ""
    summarized_text_global = summary
    return summary#, custom_summary

def custom_summary(pdf_file_path, custom_prompt):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    prompt_template = custom_prompt + """

    {text}

    SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                map_prompt=PROMPT, combine_prompt=PROMPT)
    summary_output = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
    
    return summary_output

# Sidebar content
with st.sidebar:
    st.title('Chat PDF App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - Streamlit
    - LangChain
    - OpenAI LLM Model
    ''')
    add_vertical_space(5)
    st.write('Made with love by XIDOA')
    


# summarized = False

def main():
    st.header("Chat with PDF")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type = 'pdf')

    # # Summarizing PDF
    # if pdf is not None:
    #     with NamedTemporaryFile(dir='./tmppdfs', suffix='.pdf') as tmpfile:
    #         tmpfile.write(pdf.getbuffer())
    #         # Check if the PDF summary is in the cache
    #         summary_data = st.cache_data(summarize_pdf)
    #         if summary_data is None:
    #             # If not in the cache, run summarize_pdf and cache the result
    #             summary_data = summarize_pdf(tmpfile.name, custom_prompt="")
    #             # summarized_text = summary_data[0]
    #             st.write("PDF Summary: ")
    #             st.write(summary_data)
    #         else:
    #             # summarized_text_global = summarize_pdf(tmpfile.name, custom_prompt="")
    #             st.write("PDF Summary: ")
    #             st.write(summarized_text_global)


    
    # st.write(pdf)
    if  pdf is not None:
        st.write (pdf.name)
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =2000,
            chunk_overlap = 100,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)
        
        # # embeddings
        embeddings = OpenAIEmbeddings(model="text-davinci-003", chunk_size=2000,)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded From the Disk')
        else:
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore,f)

        # Accept user questions/queries
        query = st.text_input("Ask questions to your PDF file:")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            # st.write(docs)

            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9, model_kwargs = {"temperature": 0.9, "max_tokens": 3000,}, max_tokens=3000)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)




if __name__ == '__main__':
    main()