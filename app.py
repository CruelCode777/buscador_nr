import streamlit as st
import os
import google.generativeai as genai # Biblioteca para listar modelos
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
if "GOOGLE_API_KEY" in st.secrets:
    google_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Erro: GOOGLE_API_KEY não encontrada nos Secrets.")
    st.stop()

if "PINECONE_API_KEY" in st.secrets:
    pinecone_key = st.secrets["PINECONE_API_KEY"]
else:
    st.error("Erro: PINECONE_API_KEY não encontrada nos Secrets.")
    st.stop()

st.title("👷 Consultor de NRs (IA)")

# --- SOLUÇÃO DO ERRO 404 (Auto-Detecção) ---
# Em vez de adivinhar o nome, perguntamos ao Google o que você tem direito de usar
@st.cache_resource
def obter_modelo_valido(api_key):
    genai.configure(api_key=api_key)
    try:
        # Lista modelos que suportam gerar texto
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'gemini-1.5-flash' in m.name: return m.name # Prioridade: Rápido
                if 'gemini-1.5-pro' in m.name: return m.name   # Prioridade: Inteligente
                if 'gemini-pro' in m.name: return m.name       # Fallback: Clássico
        return "models/gemini-1.5-flash" # Chute final se a lista falhar
    except:
        return "gemini-1.5-flash"

# Descobre o modelo agora
nome_modelo = obter_modelo_valido(google_key)
st.caption(f"Base de conhecimento unificada (Usando: {nome_modelo})")
# --------------------------------------------

# --- CONEXÃO COM A BASE DE DADOS (PINECONE) ---
@st.cache_resource
def get_knowledge_base():
    os.environ['PINECONE_API_KEY'] = pinecone_key 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="base-nrs",
        embedding=embeddings
    )
    return vectorstore

try:
    vectorstore = get_knowledge_base()
except Exception as e:
    st.error(f"Erro ao conectar no banco de dados: {e}")
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
        with st.spinner("Consultando a base unificada de normas..."):
            try:
                # 1. Busca
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(prompt)
                
                if not docs:
                    response_text = "Não encontrei informações sobre isso na base de dados das NRs."
                else:
                    context_text = ""
                    sources = set()
                    for doc in docs:
                        src = doc.metadata.get('source', 'Desconhecido')
                        context_text += f"{doc.page_content}\n(Fonte: {src})\n---\n"
                        sources.add(src)

                    # 2. Prompt
                    system_prompt = """
                    Você é um Consultor Sênior em Segurança do Trabalho (HSE).
                    Responda com base estrita nas Normas Regulamentadoras (NRs).
                    Contexto: {context}
                    Pergunta: {question}
                    """
                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    
                    # 3. Chama a IA usando o modelo que descobrimos lá em cima
                    llm = ChatGoogleGenerativeAI(
                        model=nome_modelo, 
                        temperature=0.1,
                        google_api_key=google_key
                    )
                    
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context_text, "question": prompt})
                    response_text = response.content + f"\n\n\n*Fontes: {', '.join(sources)}*"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error(f"Erro: {e}")
