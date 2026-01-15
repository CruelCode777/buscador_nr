# --- FIX PARA O STREAMLIT CLOUD ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------

import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Assistente NR", page_icon="🦺")

st.title("🦺 Assistente de NRs (Versão Estável)")
st.caption("Rodando com FastEmbed + Groq (Llama 3.3).")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configurações")
    uploaded_file = st.file_uploader("Envie o PDF da NR", type="pdf")
    
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Chave API Groq:", type="password")

# --- PROCESSAMENTO ---
@st.cache_resource
def get_vectorstore(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    if not docs:
        st.error("Não foi possível ler o texto do PDF. Verifique se não é uma imagem escaneada.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # FastEmbed (Leve e rápido)
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# --- LÓGICA DO APP ---
if uploaded_file and api_key:
    # Salva arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # Só processa se houver mudança de arquivo
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Processando norma..."):
                vectorstore = get_vectorstore(tmp_path)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_file = uploaded_file.name
                    st.success("Norma carregada!")
                else:
                    st.stop() # Para a execução se o PDF for inválido

        # Chat
        question = st.chat_input("Pergunte sobre a norma...")
        
        if question:
            st.chat_message("user").write(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        # ATUALIZAÇÃO IMPORTANTE: Trocamos o modelo para o Llama 3.3 (Mais novo e estável)
                        llm = ChatGroq(
                            temperature=0, 
                            model_name="llama-3.3-70b-versatile", 
                            groq_api_key=api_key
                        )
                        
                        retriever = st.session_state.vectorstore.as_retriever()
                        docs = retriever.invoke(question)
                        
                        # Proteção contra contexto vazio
                        if not docs:
                            st.warning("Não encontrei informações relevantes nesse documento para sua pergunta.")
                            st.stop()

                        context = "\n\n".join([d.page_content for d in docs])
                        
                        prompt = ChatPromptTemplate.from_template("""
                        Você é um Especialista em Segurança do Trabalho.
                        Responda com base no contexto abaixo.
                        
                        Contexto: {context}
                        Pergunta: {question}
                        """)
                        
                        chain = prompt | llm
                        res = chain.invoke({"context": context, "question": question})
                        st.write(res.content)
                        
                    except Exception as e:
                        # Mostra o erro real na tela para facilitar o diagnóstico
                        st.error(f"Erro na comunicação com a IA: {e}")
                    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

elif not uploaded_file:
    st.info("Envie um PDF para começar.")
