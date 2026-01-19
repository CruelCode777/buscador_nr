import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="IA de Segurança do Trabalho", page_icon="👷", layout="wide")

# SEGREDOS
gemini_key = st.secrets["GEMINI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = gemini_key
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

st.title("🔧 **DIAGNÓSTICO - IA de NRs**")

# Teste 1: Conexão Pinecone
@st.cache_resource
def test_pinecone():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return PineconeVectorStore.from_existing_index(
        index_name="base-nrs",
        embedding=embeddings,
    )

try:
    vectorstore = test_pinecone()
    st.success("✅ **Pinecone conectado!**")
except Exception as e:
    st.error(f"❌ **Erro Pinecone**: {e}")
    st.stop()

# Teste 2: Quantos vetores tem no índice?
st.subheader("📊 **Status do Índice**")
try:
    # Testa retrieval simples
    test_docs = vectorstore.similarity_search("NR", k=3)
    st.success(f"✅ **{len(test_docs)} documentos encontrados** para 'NR'")
    
    for i, doc in enumerate(test_docs):
        st.caption(f"**{i+1}**: {doc.metadata.get('source', 'sem fonte')}")
        
except Exception as e:
    st.error(f"❌ **Retrieval falhou**: {e}")

# Teste 3: Chat funcional
st.subheader("💬 **Teste Chat**")
if prompt := st.chat_input("Digite 'NR 29' para testar"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(prompt)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**📋 Docs encontrados:**")
        if docs:
            for i, doc in enumerate(docs[:5]):
                st.caption(f"{i+1}. {doc.metadata.get('source', '?')}")
        else:
            st.error("❌ ZERO documentos!")
    
    with col2:
        if docs:
            context = "\n\n".join(doc.page_content for doc in docs)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_key)
            
            response = llm.invoke(f"Contexto NRs:\n{context}\n\nPergunta: {prompt}")
            st.markdown(response.content)
        else:
            st.warning("Sem documentos para responder.")
