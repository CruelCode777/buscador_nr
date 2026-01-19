import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
groq_key = st.secrets["GROQ_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

st.title("👷 Consultor de NRs (IA)")
st.caption("Base de conhecimento unificada de todas as Normas Regulamentadoras.")

# --- CONEXÃO COM A BASE DE DADOS (PINECONE) ---
@st.cache_resource
def get_knowledge_base():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="base-nrs",
        embedding=embeddings,
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
                # Busca mais documentos para melhor cobertura
                retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
                docs = retriever.invoke(prompt)
                
                if not docs:
                    st.warning("Não encontrei informações sobre isso na base de dados das NRs.")
                else:
                    # Formata contexto
                    context_text = ""
                    sources = set()
                    for doc in docs:
                        src = doc.metadata.get('source', 'Desconhecido')
                        context_text += f"{doc.page_content}\n(Fonte: {src})\n\n"
                        sources.add(src)

                    # Prompt otimizado
                    system_prompt = """
                    Você é um Consultor Sênior em Segurança do Trabalho (HSE).
                    Sua missão é orientar profissionais com base estrita nas Normas Regulamentadoras (NRs).
                    
                    Diretrizes:
                    - Use tópicos (• ou 1.) para listas
                    - Sempre cite NR e item específico
                    - Se não achar no contexto, diga "norma não especifica"
                    
                    CONTEÚDO DAS NORMAS:
                    {context}
                    
                    PERGUNTA: {question}
                    """

                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    
                    # Groq Llama3 (rápido e preciso)
                    llm = ChatGroq(
                        temperature=0.1, 
                        model_name="llama-3.3-70b-versatile", 
                        groq_api_key=groq_key
                    )
                    chain = prompt_template | llm
                    
                    response = chain.invoke({"context": context_text, "question": prompt})
                    
                    response_text = (
                        response.content + 
                        f"\n\n**📚 Fontes:** {', '.join(list(sources)[:3])}"
                    )

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"Erro na resposta: {e}")
