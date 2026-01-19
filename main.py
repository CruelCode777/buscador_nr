import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="wide")

# --- SEGREDOS ---
gemini_key = st.secrets["GEMINI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = gemini_key
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
    st.success("✅ Pinecone conectado!")
except Exception as e:
    st.error(f"❌ Erro ao conectar no banco de dados: {e}")
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
                # 1. Busca MAIS trechos (k=10 para melhor cobertura)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                docs = retriever.invoke(prompt)

                if not docs:
                    st.error("❌ Não encontrei informações sobre isso na base de dados das NRs.")
                else:
                    # DEBUG SIMPLES: Lista de fontes encontradas
                    st.info(f"**📚 Encontrados {len(docs)} trechos**")
                    fontes_unicas = list(set(doc.metadata.get("source", "Desconhecido") for doc in docs))
                    for i, fonte in enumerate(fontes_unicas[:3], 1):
                        st.caption(f"{i}. {os.path.basename(fonte)}")

                    # 2. Formata contexto para IA
                    context_text = ""
                    sources = set()
                    for doc in docs:
                        src = doc.metadata.get("source", "Desconhecido")
                        context_text += f"{doc.page_content}\n(Fonte: {src})\n---\n"
                        sources.add(src)

                    # 3. Prompt otimizado
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

                    # 4. Gemini (melhorado)
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0.1,
                        google_api_key=gemini_key,
                    )
                    chain = prompt_template | llm

                    response = chain.invoke(
                        {"context": context_text, "question": prompt}
                    )

                    response_text = (
                        response.content
                        + f"\n\n\n*Fontes consultadas: {', '.join(list(sources)[:3])}*"
                    )

                    st.markdown(response_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

            except Exception as e:
                st.error(f"Ocorreu um erro durante a resposta: {e}")
