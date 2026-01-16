import streamlit as st
import os
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- 1. CONFIGURAÇÃO DE CHAVES (Manual e Direta) ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("⚠️ Erro: Falta GOOGLE_API_KEY nos Secrets.")
    st.stop()

if "PINECONE_API_KEY" not in st.secrets:
    st.error("⚠️ Erro: Falta PINECONE_API_KEY nos Secrets.")
    st.stop()

# Configura o Google
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- 2. CARREGAMENTO DOS MODELOS (Cacheado) ---
@st.cache_resource
def carregar_modelos():
    # Carrega o modelo de embeddings (transforma texto em numeros)
    # Isso substitui o HuggingFaceEmbeddings do LangChain
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model

@st.cache_resource
def conectar_pinecone():
    # Conecta direto no Pinecone (sem LangChain)
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    # SEU INDEX SE CHAMA 'base-nrs', certo?
    index = pc.Index("base-nrs") 
    return index

try:
    embedding_model = carregar_modelos()
    pinecone_index = conectar_pinecone()
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}")
    st.stop()

# --- 3. FUNÇÃO DE BUSCA E RESPOSTA ---
def buscar_e_responder(pergunta):
    # A. Transforma a pergunta em números
    vector = embedding_model.encode(pergunta).tolist()
    
    # B. Busca no Pinecone
    resultados = pinecone_index.query(vector=vector, top_k=5, include_metadata=True)
    
    # C. Monta o Contexto
    contexto = ""
    fontes = set()
    for match in resultados['matches']:
        if match['score'] > 0.3: # Filtra coisas pouco relevantes
            texto = match['metadata'].get('text', '') # O Pinecone guarda o texto no campo 'text' ou 'page_content'
            if not texto: texto = match['metadata'].get('page_content', '')
            
            fonte = match['metadata'].get('source', 'NR')
            contexto += f"- {texto}\n(Fonte: {fonte})\n---\n"
            fontes.add(fonte)
    
    if not contexto:
        return "Não encontrei informações suficientes nas NRs processadas.", []

    # D. Manda para o Google (Direto)
    prompt_final = f"""
    Você é um Engenheiro de Segurança do Trabalho.
    Responda à pergunta do usuário usando APENAS o contexto abaixo.
    Se a resposta não estiver no contexto, diga que a norma não cita.

    CONTEXTO:
    {contexto}

    PERGUNTA:
    {pergunta}
    """
    
    # Tenta modelos em ordem de prioridade
    modelos_para_testar = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
    
    resposta_texto = "Erro ao conectar com o Google."
    
    for nome_modelo in modelos_para_testar:
        try:
            model = genai.GenerativeModel(nome_modelo)
            response = model.generate_content(prompt_final)
            resposta_texto = response.text
            break # Se funcionou, para o loop
        except Exception:
            continue # Se deu erro, tenta o proximo da lista

    return resposta_texto, list(fontes)

# --- 4. INTERFACE DE CHAT ---
st.title("👷 Consultor de NRs (Modo Raiz)")
st.caption("Sem LangChain | Conexão Direta")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: O que a NR diz sobre escadas?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processando..."):
            resposta, fontes = buscar_e_responder(prompt)
            
            if fontes:
                resposta += f"\n\n*Fontes: {', '.join(fontes)}*"
            
            st.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})
