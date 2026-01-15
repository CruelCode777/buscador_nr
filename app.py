import streamlit as st
import os  # <--- Importante adicionar isso
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Configuração da Página
st.set_page_config(page_title="Consultor SST", page_icon="👷", layout="centered")

# --- ESTILO GOOGLE (CSS INJETADO) ---
def local_css():
    st.markdown("""
    <style>
    /* Importando fonte estilo Google */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Esconde menus padrões do Streamlit para limpar a tela */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Centraliza o título */
    .title-container {
        text-align: center;
        margin-top: 50px;
        margin-bottom: 30px;
    }
    
    /* Estilo das letras do Google */
    .g-blue {color: #4285F4;}
    .g-red {color: #EA4335;}
    .g-yellow {color: #FBBC05;}
    .g-green {color: #34A853;}
    
    .big-font {
        font-size: 60px;
        font-weight: bold;
    }

    /* Ajuste dos cards de mensagem */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #dfe1e5;
    }
    </style>
    """, unsafe_allow_html=True)

def google_logo():
    st.markdown("""
    <div class="title-container">
        <span class="big-font g-blue">B</span>
        <span class="big-font g-red">u</span>
        <span class="big-font g-yellow">s</span>
        <span class="big-font g-blue">c</span>
        <span class="big-font g-green">a</span>
        <span class="big-font g-red">r</span>
        <span class="big-font g-blue" style="margin-left: 15px;">N</span>
        <span class="big-font g-green">R</span>
    </div>
    <div style="text-align: center; color: #5f6368; margin-bottom: 40px;">
        Inteligência Artificial aplicada à Segurança do Trabalho
    </div>
    """, unsafe_allow_html=True)

# --- FUNÇÃO PRINCIPAL ---
def main_app():
    local_css() # Aplica o visual Google

# --- SEGREDOS ---
groq_key = st.secrets["GROQ_API_KEY"]
pinecone_key = st.secrets["PINECONE_API_KEY"]

st.title("👷 Consultor de NRs")
st.caption("Base de conhecimento unificada de todas as Normas Regulamentadoras publicadas no site do MTE (gov.br database.)")

# --- CONEXÃO COM A BASE DE DADOS (PINECONE) ---
@st.cache_resource
def get_knowledge_base():
    # Define a chave no ambiente (é assim que a nova biblioteca procura)
    os.environ['PINECONE_API_KEY'] = pinecone_key 

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Conecta ao índice (agora sem passar a chave explicitamente aqui dentro)
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

# Mostra histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de pergunta
if prompt := st.chat_input("Ex: Quais os exames obrigatórios para trabalho em altura?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a base unificada de normas..."):
            
            try:
                # 1. Busca os trechos mais relevantes no Pinecone
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                docs = retriever.invoke(prompt)
                
                if not docs:
                    response_text = "Não encontrei informações sobre isso na base de dados das NRs."
                else:
                    # Formata o contexto
                    context_text = ""
                    sources = set()
                    for doc in docs:
                        # Proteção caso o metadado 'source' esteja vazio
                        src = doc.metadata.get('source', 'Desconhecido')
                        context_text += f"{doc.page_content}\n(Fonte: {src})\n---\n"
                        sources.add(src)

                    # 2. O Prompt
                    system_prompt = """
                    Você é um Consultor Sênior em Segurança do Trabalho (HSE).
                    Sua missão é orientar profissionais com base estrita nas Normas Regulamentadoras (NRs).
                    
                    Diretrizes:
                    1. Use tópicos para listas.
                    2. Cite qual NR e item embasa a resposta.
                    3. Se não estiver no contexto, diga que a norma não especifica.
                    
                    Contexto das Normas:
                    {context}
                    
                    Pergunta do Usuário: {question}
                    """
                    
                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    
                    # 3. Chama a IA (Groq) - Usando modelo estável
                    llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
                    chain = prompt_template | llm
                    
                    response = chain.invoke({"context": context_text, "question": prompt})
                    
                    response_text = response.content + f"\n\n\n*Fontes consultadas: {', '.join(sources)}*"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error(f"Ocorreu um erro durante a resposta: {e}")


