import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
# Configure no Streamlit Cloud: GROQ_API_KEY e PINECONE_API_KEY
groq_key = st.secrets["GROQ_API_KEY"]
pinecone_key = st.secrets["PINECONE_API_KEY"]

st.title("👷 Consultor de NRs (IA)")
st.caption("Base de conhecimento unificada de todas as Normas Regulamentadoras.")

# --- CONEXÃO COM A BASE DE DADOS (PINECONE) ---
@st.cache_resource
def get_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Conecta ao índice que você criou e já populou
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name="base-nrs",
        embedding=embeddings,
        pinecone_api_key=pinecone_key
    )
    return vectorstore

vectorstore = get_knowledge_base()

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
            
            # 1. Busca os trechos mais relevantes no Pinecone
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Traz 4 trechos
            docs = retriever.invoke(prompt)
            
            # Formata o contexto
            context_text = ""
            sources = set()
            for doc in docs:
                context_text += f"{doc.page_content}\n---\n"
                sources.add(doc.metadata['source']) # Pega o nome do arquivo original

            # 2. O Prompt "Engenharia de Prompt" para fluidez
            system_prompt = """
            Você é um Consultor Sênior em Segurança do Trabalho (HSE).
            Sua missão é orientar profissionais com base estrita nas Normas Regulamentadoras (NRs).
            
            Diretrizes de Resposta:
            1. **Tom de Voz:** Profissional, direto, mas educado e prestativo.
            2. **Estrutura:** Use tópicos (bullet points) para listas. É mais fácil de ler.
            3. **Citação:** Sempre cite qual NR e qual item embasa sua resposta (ex: "Conforme NR-35 item 35.2...").
            4. **Honestidade:** Se a informação não estiver no contexto abaixo, diga que a norma consultada não especifica, não invente.
            
            Contexto das Normas:
            {context}
            
            Pergunta do Usuário: {question}
            """
            
            prompt_template = ChatPromptTemplate.from_template(system_prompt)
            
            # 3. Chama a IA (Groq)
            llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
            chain = prompt_template | llm
            
            response = chain.invoke({"context": context_text, "question": prompt})
            
            # Adiciona as fontes no final da resposta
            final_response = response.content +f"\n\n\n*Fontes consultadas: {', '.join(sources)}*"
            
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
