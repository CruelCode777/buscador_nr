import streamlit as st
import os
import google.generativeai as genai # Biblioteca oficial para listar modelos
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Configuração da Página
st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="centered")

# --- SEGREDOS ---
if "GOOGLE_API_KEY" in st.secrets:
    google_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("ERRO: A chave GOOGLE_API_KEY não foi encontrada nos Secrets.")
    st.stop()

if "PINECONE_API_KEY" in st.secrets:
    pinecone_key = st.secrets["PINECONE_API_KEY"]
else:
    st.error("ERRO: A chave PINECONE_API_KEY não foi encontrada nos Secrets.")
    st.stop()

st.title("👷 Consultor de NRs (IA)")
st.caption("Base de conhecimento unificada (Powered by Google Gemini)")

# --- FUNÇÃO DE AUTO-DESCOBERTA DE MODELO ---
@st.cache_resource
def descobrir_modelo_google(api_key):
    """
    Pergunta ao Google quais modelos estão disponíveis para esta chave
    e retorna o melhor para uso.
    """
    genai.configure(api_key=api_key)
    try:
        # Lista todos os modelos disponíveis
        modelos = list(genai.list_models())
        
        # Filtra apenas os que geram texto (generateContent) e são da família Gemini
        modelos_uteis = [m.name for m in modelos if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name]
        
        # Tenta priorizar o Flash (mais rápido), depois o Pro
        modelo_escolhido = None
        
        # 1. Tenta achar o 1.5 Flash
        for m in modelos_uteis:
            if "1.5-flash" in m:
                modelo_escolhido = m
                break
        
        # 2. Se não achar, tenta o 1.5 Pro
        if not modelo_escolhido:
            for m in modelos_uteis:
                if "1.5-pro" in m:
                    modelo_escolhido = m
                    break
        
        # 3. Se não achar, pega qualquer um disponível (ex: 1.0-pro)
        if not modelo_escolhido and modelos_uteis:
            modelo_escolhido = modelos_uteis[0]
            
        if modelo_escolhido:
            # Remove o prefixo 'models/' se vier junto, pois o LangChain as vezes duplica
            return modelo_escolhido.replace("models/", "")
        else:
            return "gemini-1.5-flash" # Fallback padrão se a lista falhar
            
    except Exception as e:
        st.warning(f"Não consegui listar os modelos automaticamente: {e}. Usando padrão.")
        return "gemini-1.5-flash"

# Descobre o modelo agora
nome_modelo_atual = descobrir_modelo_google(google_key)

# Mostra no sidebar qual modelo está sendo usado (para você saber)
with st.sidebar:
    st.success(f"🤖 Modelo Ativo: {nome_modelo_atual}")

# --- CONEXÃO COM A BASE DE DADOS (PINECONE) ---
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

# --- CHAT ---
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
        with st.spinner(f"Consultando normas com {nome_modelo_atual}..."):
            
            try:
                # 1. Busca no Pinecone
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
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

                    # 2. Prompt
                    system_prompt = """
                    Você é um Consultor Sênior em Segurança do Trabalho (HSE).
                    Responda com base estrita nas Normas Regulamentadoras (NRs) fornecidas abaixo.
                    
                    Diretrizes:
                    1. Use formatação Markdown (negrito, tópicos).
                    2. Cite a NR e o item sempre que possível.
                    3. Se a informação não estiver no contexto, diga que não encontrou.
                    
                    Contexto:
                    {context}
                    
                    Pergunta: {question}
                    """
                    
                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    
                    # 3. Chama o Google Gemini com o modelo descoberto
                    llm = ChatGoogleGenerativeAI(
                        model=nome_modelo_atual,
                        temperature=0.1,
                        google_api_key=google_key
                    )
                    
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context_text, "question": prompt})
                    
                    response_text = response.content + f"\n\n\n*Fontes: {', '.join(sources)}*"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error(f"Erro: {e}")
