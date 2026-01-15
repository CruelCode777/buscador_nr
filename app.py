import streamlit as st
import os
import uuid
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    pinecone_key = st.secrets["PINECONE_API_KEY"]
except FileNotFoundError:
    st.warning("⚠️ Chaves de API não configuradas.")
    st.stop()

st.title("👷 Consultor de NRs (IA)")
st.caption("Base de conhecimento unificada de todas as Normas Regulamentadoras.")

# --- GERENCIAMENTO DE SESSÃO E HISTÓRICO ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {} 

if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.chat_history[new_id] = {'title': 'Nova Conversa', 'messages': []}

def criar_nova_conversa():
    new_id = str(uuid.uuid4())
    st.session_state.chat_history[new_id] = {'title': 'Nova Conversa', 'messages': []}
    st.session_state.current_chat_id = new_id

def selecionar_conversa(chat_id):
    st.session_state.current_chat_id = chat_id

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Histórico de Conversas")
    if st.button("➕ Nova Conversa", use_container_width=True, type="primary"):
        criar_nova_conversa()
    st.divider()
    
    ids_conversas = list(st.session_state.chat_history.keys())
    for chat_id in reversed(ids_conversas):
        conversa = st.session_state.chat_history[chat_id]
        titulo = conversa['title']
        if len(titulo) > 25: titulo = titulo[:25] + "..."
        if chat_id == st.session_state.current_chat_id: titulo = f"🟢 {titulo}"
        if st.button(titulo, key=chat_id, use_container_width=True):
            selecionar_conversa(chat_id)

# --- CONEXÃO PINECONE ---
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

# --- CHAT PRINCIPAL ---
chat_id_atual = st.session_state.current_chat_id
dados_conversa_atual = st.session_state.chat_history[chat_id_atual]
mensagens_atuais = dados_conversa_atual['messages']

for message in mensagens_atuais:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Quais os exames obrigatórios para trabalho em altura?"):
    if len(mensagens_atuais) == 0:
        dados_conversa_atual['title'] = prompt

    mensagens_atuais.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando normas..."):
            try:
                # 1. Busca documentos
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
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

                    system_prompt = """
                    Você é um Consultor Sênior em Segurança do Trabalho (HSE).
                    Sua missão é orientar profissionais com base estrita nas Normas Regulamentadoras (NRs).
                    Diretrizes: Use tópicos, cite a NR e seja técnico.
                    Contexto: {context}
                    Pergunta: {question}
                    """
                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    
                    # --- LÓGICA DE FALLBACK (PLANO B) ---
                    try:
                        # Tenta o modelo Potente (70B)
                        llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
                        chain = prompt_template | llm
                        response = chain.invoke({"context": context_text, "question": prompt})
                        
                    except Exception as e_groq:
                        if "429" in str(e_groq):
                            # Se der erro de limite, avisa (opcional) e troca para o modelo Leve (8B)
                            st.toast("⚠️ Alto tráfego no modelo principal. Alternando para modo rápido (8B).")
                            llm_backup = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", groq_api_key=groq_key)
                            chain = prompt_template | llm_backup
                            response = chain.invoke({"context": context_text, "question": prompt})
                        else:
                            raise e_groq # Se for outro erro, repassa

                    response_text = response.content + f"\n\n\n*Fontes: {', '.join(sources)}*"
                
                st.markdown(response_text)
                mensagens_atuais.append({"role": "assistant", "content": response_text})
                
                if len(mensagens_atuais) == 2:
                    st.rerun()
            
            except Exception as e:
                st.error(f"Erro no sistema: {e}")
