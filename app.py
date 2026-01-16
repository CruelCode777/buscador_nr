import streamlit as st
import os
# --- A CORREÇÃO ESTÁ AQUI: Removemos o import do 'google.generativeai' que dava erro ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("⚠️ Erro: Adicione a GOOGLE_API_KEY nos Secrets do Streamlit.")
    st.stop()

if "PINECONE_API_KEY" not in st.secrets:
    st.error("⚠️ Erro: Adicione a PINECONE_API_KEY nos Secrets do Streamlit.")
    st.stop()

# Configura variáveis de ambiente
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

st.title("👷 Consultor de NRs (IA)")
st.caption("Base de conhecimento unificada (Google Gemini 1.5 Flash)")

# --- CONEXÃO COM O BANCO DE DADOS ---
@st.cache_resource
def get_vectorstore():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Conexão Pinecone
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="base-nrs",
        embedding=embeddings
    )
    return vectorstore

try:
    vectorstore = get_vectorstore()
except Exception as e:
    st.error(f"Erro ao conectar no Pinecone: {e}")
    st.stop()

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Quais os exames obrigatórios para trabalho em altura?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando normas..."):
            try:
                # 1. Recuperação (Retrieval)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(prompt)
                
                if not docs:
                    response_text = "Não encontrei informações relevantes na base de dados."
                else:
                    context_text = "\n\n".join([f"{d.page_content} (Fonte: {d.metadata.get('source', 'NR')})" for d in docs])

                    # 2. Prompt
                    template = """
                    Você é um Especialista em Segurança do Trabalho. Responda com base no contexto abaixo.
                    
                    Contexto:
                    {context}
                    
                    Pergunta: {question}
                    """
                    prompt_template = ChatPromptTemplate.from_template(template)
                    
                    # 3. Modelo Google (Configuração Simplificada)
                    # Usamos 'gemini-1.5-flash'. Se este modelo der erro 404 futuramente,
                    # basta trocar o texto abaixo para 'gemini-pro'.
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash", 
                        temperature=0.1
                    )
                    
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context_text, "question": prompt})
                    response_text = response.content
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
