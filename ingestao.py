import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- CONFIGURAÇÃO ---
# Cole sua chave do Pinecone aqui dentro das aspas!
PINECONE_KEY = "pcsk_3yDmdF_KDGdqaSpynmKQtARBS3Y428FqCteuvxnkHE4gcsEvkRWSmwjA7HbbaTPY8YUMUT" 

INDEX_NAME = "base-nrs"
PASTA_PDFS = "pdfs"

# Configura o ambiente para a biblioteca achar a chave
os.environ['PINECONE_API_KEY'] = PINECONE_KEY

def carregar_tudo():
    print(f"📂 Lendo PDFs da pasta '{PASTA_PDFS}'...")
    
    # 1. Carrega os PDFs
    loader = PyPDFDirectoryLoader(PASTA_PDFS)
    docs = loader.load()
    
    if not docs:
        print("❌ Nenhum PDF encontrado! Verifique se a pasta 'pdfs' existe e tem arquivos.")
        return

    print(f"✅ Carregou {len(docs)} páginas.")

    # 2. Quebra em pedaços menores (Chunks)
    print("✂️ Quebrando texto em pedaços...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"🧩 Gerou {len(splits)} trechos de texto.")

    # 3. Prepara os Embeddings (Gratuito)
    print("🧠 Carregando modelo de IA (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 4. Envia para o Pinecone
    print("🚀 Enviando para o Pinecone (Isso pode demorar um pouco)...")
    try:
        PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        print("🎉 SUCESSO! Todos os documentos foram enviados.")
    except Exception as e:
        print(f"❌ Erro ao enviar para o Pinecone: {e}")

if __name__ == "__main__":
    carregar_tudo()