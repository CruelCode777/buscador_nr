import streamlit as st
import os
import uuid # Biblioteca para gerar IDs únicos para as conversas
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

# --- GERENCIAMENTO DE SESSÃO E HISTÓRICO (Lógica do Menu Lateral) ---

# 1. Cria o banco de dados de conversas se não existir
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {} # Estrutura: {id_unico: {'title': 'Titulo', 'messages': []}}

# 2. Cria a primeira conversa se não houver nenhuma
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.chat_history[new_id] = {'title': 'Nova Conversa', 'messages': []}

# Função para criar nova conversa (botão do menu)
def criar_nova_conversa():
    new_id = str(uuid.uuid4())
    st.session_state.chat_history[new_id] = {'title': 'Nova Conversa', 'messages': []}
    st.session_state.current_chat_id = new_id

# Função para trocar de conversa
def selecionar_conversa(chat_id):
    st.session_state.current_chat_id = chat_id

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("Histórico de Conversas")
    
    # Botão Principal
    if st.button("➕ Nova Conversa", use_container_width=True, type="primary"):
        criar_nova_conversa()
    
    st.divider()
    
    # Lista as conversas existentes (da mais recente para a mais antiga)
    # Pegamos os IDs e invertemos a ordem
    ids_conversas = list(st.session_state.chat_history.keys())
    
    for chat_id in reversed(ids_conversas):
        conversa = st.session_state.chat_history[chat_id]
        titulo = conversa['title']
        
        # Limita o tamanho do texto no botão para não quebrar
        if len(titulo) > 25:
            titulo = titulo[:25] + "..."
            
        # Marca visualmente qual conversa está ativa
        if chat_id == st.session_state.current_chat_id:
            titulo = f"🟢 {titulo}"
        
        if st.button(titulo, key=chat_id, use_container_width=True):
            selecionar_conversa(chat_id)

# --- CONEXÃO COM A BASE DE DADOS (PINECONE) ---
@st.cache_resource
def get_knowledge_base():
    # Define a chave no ambiente
    os.environ['PINECONE_API_KEY'] = pinecone_key 

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Conecta ao índice
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

# Identifica qual conversa está ativa agora
chat_id_atual = st.session_state.current_chat_id
dados_conversa_atual = st.session_state.chat_history[chat_id_atual]
mensagens_atuais = dados_conversa_atual['messages']

# Mostra histórico da conversa ATUAL
for message in mensagens_atuais:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de pergunta
if prompt := st.chat_input("Ex: Quais os exames obrigatórios para trabalho em altura?"):
    
    # Se for a primeira mensagem dessa conversa, renomeia o título no menu
    if len(mensagens_atuais) == 0:
        dados_conversa_atual['title'] = prompt
        # (O Streamlit atualizará o nome no menu na próxima interação)

    # Adiciona mensagem do usuário na memória da conversa ATUAL
    mensagens_atuais.append({"role": "user", "content": prompt})
    
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
                
                # Salva resposta na memória da conversa ATUAL
                mensagens_atuais.append({"role": "assistant", "content": response_text})
                
                # Se for a primeira mensagem, força recarregar para atualizar o nome no menu lateral imediatamente
                if len(mensagens_atuais) == 2:
                    st.rerun()
            
            except Exception as e:
                st.error(f"Ocorreu um erro durante a resposta: {e}")
