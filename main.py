import streamlit as st
import os
import google.generativeai as genai # Usando direto, sem LangChain no meio
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("⚠️ Erro: Adicione a GOOGLE_API_KEY nos Secrets.")
    st.stop()

if "PINECONE_API_KEY" not in st.secrets:
    st.error("⚠️ Erro: Adicione a PINECONE_API_KEY nos Secrets.")
    st.stop()

# Configurações
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]) # Configura o Google direto

st.title("👷 Consultor de NRs (IA)")
st.caption("Base de conhecimento unificada (Gemini Direct)")

# --- CONEXÃO PINECONE ---
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
        st.error(f"Erro ao conectar no Pinecone: {e}")
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
                # 1. Busca no Pinecone (LangChain faz isso bem)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(prompt)
                
                if not docs:
                    response_text = "Não encontrei informações relevantes na base de dados."
                else:
                    # Monta o texto de contexto
                    context_text = "\n\n".join([f"{d.page_content} (Fonte: {d.metadata.get('source', 'NR')})" for d in docs])

                    # 2. Monta o Prompt Manualmente
                    full_prompt = f"""
                    Você é um Especialista em Segurança do Trabalho. Responda à pergunta com base APENAS no contexto abaixo.
                    
                    CONTEXTO DAS NORMAS:
                    {context_text}
                    
                    PERGUNTA DO USUÁRIO:
                    {prompt}
                    """
                    
                    # 3. Chama o Google DIRETAMENTE (Sem LangChain atrapalhando)
                    # O 'gemini-pro' costuma ser o nome universal na API direta
                    model = genai.GenerativeModel('gemini-pro') 
                    response = model.generate_content(full_prompt)
                    
                    response_text = response.text
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                # Se falhar o gemini-pro, tenta o flash
                if "404" in str(e) or "not found" in str(e).lower():
                    try:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(full_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except:
                         st.error(f"Erro no Google: {e}")
                else:
                    st.error(f"Ocorreu um erro: {e}")
