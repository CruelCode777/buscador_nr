import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # <--- Gemini via LangChain [web:13]

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
# Adicione no .streamlit/secrets.toml:
# GEMINI_API_KEY = "sua-chave-aqui"
gemini_key = st.secrets["GEMINI_API_KEY"]

# Opcional: setar também como variável de ambiente (algumas libs usam isso)
os.environ["GOOGLE_API_KEY"] = gemini_key

st.title("👷 Consultor de NRs (IA)")
st.caption("Base de conhecimento unificada de todas as Normas Regulamentadoras.")

# --- CONEXÃO COM A BASE DE DADOS (PINECONE) ---
@st.cache_resource
def get_knowledge_base():
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

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
                        src = doc.metadata.get("source", "Desconhecido")
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

                    # 3. Chama a IA (Gemini)
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",  # ou outro modelo disponível [web:13]
                        temperature=0.1,
                        google_api_key=gemini_key,
                    )

                    chain = prompt_template | llm

                    response = chain.invoke(
                        {"context": context_text, "question": prompt}
                    )

                    # Em ChatGoogleGenerativeAI, o conteúdo vem em response.content
                    response_text = (
                        response.content
                        + f"\n\n\n*Fontes consultadas: {', '.join(sources)}*"
                    )

                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

            except Exception as e:
                st.error(f"Ocorreu um erro durante a resposta: {e}")
