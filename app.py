# --- FIX PARA O STREAMLIT CLOUD (ChromaDB requer SQLite > 3.35) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ------------------------------------------------------------------

import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Assistente de NR", page_icon="🛡️", layout="wide")

st.title("🛡️ Assistente de Normas Regulamentadoras")
st.markdown("""
Este sistema utiliza Inteligência Artificial (Llama 3 via Groq) para ler PDFs de NRs 
e responder perguntas técnicas com base **exclusivamente** no documento enviado.
""")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Upload e Configuração")
    uploaded_file = st.file_uploader("Envie o PDF da NR aqui", type="pdf")
    
    # Tenta pegar a chave dos Segredos do Streamlit
    # Se não achar, abre uma caixa para o usuário digitar (bom para testes)
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.warning("Chave de API não configurada nos segredos.")
        api_key = st.text_input("Insira sua Chave API da Groq:", type="password")
        st.caption("Obtenha gratuitamente em: https://console.groq.com")

# --- PROCESSAMENTO DO PDF ---
def processar_pdf(uploaded_file):
    # Cria um arquivo temporário para o PyPDFLoader ler
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Divide o texto em pedaços para a IA conseguir ler tudo
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Cria o banco de dados vetorial (na memória RAM)
        # Usamos um modelo leve da HuggingFace para não estourar a memória do servidor grátis
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vectorstore
        
    finally:
        # Garante que o arquivo temporário seja deletado
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- LÓGICA DO CHAT ---
if uploaded_file and api_key:
    # Só processa o PDF se ele mudou ou se é a primeira vez
    if "vectorstore" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Lendo e indexando a norma... (Isso pode levar alguns segundos)"):
            st.session_state.vectorstore = processar_pdf(uploaded_file)
            st.session_state.last_file = uploaded_file.name
        st.success("Norma processada com sucesso!")

    # Campo de pergunta
    question = st.chat_input("Ex: Qual a periodicidade do treinamento desta NR?")
    
    if question:
        # Mostra a mensagem do usuário
        st.chat_message("user").write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Consultando a norma e formulando resposta..."):
                try:
                    # Configura o Modelo Llama 3 (Versão 70B é mais inteligente)
                    llm = ChatGroq(
                        temperature=0, 
                        model_name="llama3-70b-8192", 
                        groq_api_key=api_key
                    )
                    
                    # Recupera os trechos relevantes
                    retriever = st.session_state.vectorstore.as_retriever()
                    context_docs = retriever.invoke(question)
                    context_text = "\n\n".join([doc.page_content for doc in context_docs])
                    
                    # O Prompt que define a personalidade da IA
                    prompt_template = ChatPromptTemplate.from_template("""
                    Você é um Especialista Sênior em Segurança do Trabalho no Brasil.
                    Responda à pergunta com base APENAS no contexto fornecido abaixo (trechos da NR).
                    
                    Diretrizes:
                    1. Seja direto e técnico.
                    2. Cite o número do item da norma sempre que possível (ex: "Conforme item 35.2.1...").
                    3. Se a informação não estiver no contexto, diga: "A norma fornecida não menciona este ponto específico."
                    4. Responda em Português do Brasil.
                    
                    Contexto da NR:
                    {context}
                    
                    Pergunta do Usuário: {question}
                    """)
                    
                    # Cria e executa a corrente
                    chain = prompt_template | llm
                    resposta = chain.invoke({"context": context_text, "question": question})
                    
                    st.write(resposta.content)
                    
                    # Mostra as fontes (opcional, bom para auditoria)
                    with st.expander("Ver trechos da norma utilizados como base"):
                        st.write(context_text)
                        
                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")

elif not uploaded_file:
    st.info("👈 Faça o upload de um PDF na barra lateral para começar.")