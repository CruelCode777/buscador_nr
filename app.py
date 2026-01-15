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

# --- BARRA LATERAL COM OPÇÃO DE DEBUG ---
with st.sidebar:
    st.header("Configurações")
    modo_debug = st.toggle("🕵️ Ativar Modo Debug", value=True)
    st.divider()
    # (Aqui iria seu código de histórico de conversas, mantive simplificado para focar no erro)

# --- CONEXÃO PINECONE ---
@st.cache_resource
def get_knowledge_base():
    os.environ['PINECONE_API_KEY'] = pinecone_key 
    # IMPORTANTE: O modelo aqui TEM que ser o mesmo usado no ingestao.py
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

if prompt := st.chat_input("Pergunte sobre a norma..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando normas..."):
            try:
                # 1. AUMENTAMOS O K PARA 6 (Mais contexto)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                docs = retriever.invoke(prompt)
                
                # --- ÁREA DE DEBUG (Onde vamos descobrir o erro) ---
                if modo_debug:
                    with st.expander("🕵️ O que a IA encontrou no banco de dados?"):
                        for i, doc in enumerate(docs):
                            st.write(f"**Trecho {i+1} (Fonte: {doc.metadata.get('source', '?')})**")
                            st.info(doc.page_content[:300] + "...") # Mostra só o começo
                            st.divider()

                if not docs:
                    response_text = "ERRO: O banco de dados não retornou nenhum trecho. Verifique a ingestão."
                else:
                    context_text = "\n\n".join([d.page_content for d in docs])
                    
                    # 2. Prompt mais agressivo
                    system_prompt = """
                    Você é um Especialista Sênior em NRs. Responda APENAS com base no contexto abaixo.
                    
                    Passo a Passo:
                    1. Analise o contexto procurando a resposta para a pergunta: "{question}".
                    2. Se encontrar, responda citando o item da norma.
                    3. Se a informação NÃO estiver explicitamente no contexto abaixo, diga: "Não encontrei essa informação específica nos documentos processados". NÃO INVENTE.
                    
                    Contexto:
                    {context}
                    """
                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    
                    # Tenta Llama 3.3 (70B)
                    try:
                        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
                        chain = prompt_template | llm
                        response = chain.invoke({"context": context_text, "question": prompt})
                    except:
                        # Fallback
                        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=groq_key)
                        chain = prompt_template | llm
                        response = chain.invoke({"context": context_text, "question": prompt})

                    response_text = response.content
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error(f"Erro: {e}")
