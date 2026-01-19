import streamlit as st
import os
from google import genai  # SDK oficial Gemini [web:17]
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

st.title("👷 Consultor de NRs (IA)")
st.caption("Base de conhecimento unificada de todas as Normas Regulamentadoras.")

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

# Inicializa cliente Gemini
client = genai.Client(api_key=GEMINI_API_KEY)  # [web:17]

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
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                docs = retriever.invoke(prompt)

                if not docs:
                    response_text = "Não encontrei informações sobre isso na base de dados das NRs."
                else:
                    context_text = ""
                    sources = set()
                    for doc in docs:
                        src = doc.metadata.get("source", "Desconhecido")
                        context_text += f"{doc.page_content}\n(Fonte: {src})\n---\n"
                        sources.add(src)

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

                    filled_prompt = system_prompt.format(
                        context=context_text, question=prompt
                    )

                    # Chamada direta ao modelo Gemini 2.5 Flash [web:17]
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=filled_prompt,
                    )

                    # Normalmente o texto da resposta vem em response.text [web:17]
                    response_text = response.text + f"\n\n\n*Fontes consultadas: {', '.join(sources)}*"

                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

            except Exception as e:
                st.error(f"Ocorreu um erro durante a resposta: {e}")
