import streamlit as st
import os
import google.generativeai as genai # Biblioteca oficial do Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("⚠️ Erro: Adicione a GOOGLE_API_KEY nos Secrets.")
    st.stop()

if "PINECONE_API_KEY" not in st.secrets:
    st.error("⚠️ Erro: Adicione a PINECONE_API_KEY nos Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

st.title("👷 Consultor de NRs (IA)")

# --- LÓGICA DE AUTO-DETECÇÃO DE MODELO (A SOLUÇÃO) ---
@st.cache_resource
def descobrir_modelo_valido(api_key):
    """
    Lista os modelos disponíveis na conta do usuário e retorna o primeiro que funciona.
    """
    genai.configure(api_key=api_key)
    try:
        modelos = genai.list_models()
        # Procura qualquer modelo que seja 'gemini' e suporte gerar texto
        for m in modelos:
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini' in m.name:
                    # Retorna o nome limpo (ex: models/gemini-1.5-flash -> gemini-1.5-flash)
                    return m.name.replace("models/", "")
        return "gemini-pro" # Fallback se não achar nada
    except Exception as e:
        return "gemini-pro"

# Descobre o modelo agora
modelo_atual = descobrir_modelo_valido(st.secrets["GOOGLE_API_KEY"])
st.caption(f"Base de conhecimento unificada (Modelo detectado: {modelo_atual})")
# -----------------------------------------------------

# --- CONEXÃO COM O BANCO DE DADOS ---
@st.cache_resource
def get_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name="base-nrs",
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        # Se falhar a importação do sentence-transformers, avisa
        st.error("Erro no Pinecone/Embeddings. Verifique se 'sentence-transformers' está no requirements.txt")
        st.stop()

vectorstore = get_vectorstore()

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
                # 1. Busca
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
                    
                    # 3. Chama a IA usando o modelo DESCOBERTO automaticamente
                    llm = ChatGoogleGenerativeAI(
                        model=modelo_atual, 
                        temperature=0.1
                    )
                    
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context_text, "question": prompt})
                    response_text = response.content
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
