import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- CONFIGURAÇÃO ---
PINECONE_API_KEY = "sua_chave_do_pinecone_aqui"
INDEX_NAME = "base-nrs"

# Inicializa Pinecone
pc = Pinecone(api_key=pcsk_3yDmdF_KDGdqaSpynmKQtARBS3Y428FqCteuvxnkHE4gcsEvkRWSmwjA7HbbaTPY8YUMUT)

# Modelo de Embeddings (O mesmo do site)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def carregar_tudo():
    pasta = "./pdfs" # Coloque seus PDFs aqui
    documentos_totais = []
    
    print("📂 Lendo PDFs...")
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta, arquivo)
            print(f"   - Processando {arquivo}...")
            loader = PyPDFLoader(caminho)
            docs = loader.load()
            
            # Adiciona metadados para sabermos de qual NR veio a resposta
            for doc in docs:
                doc.metadata["source"] = arquivo
            
            documentos_totais.extend(docs)

    print(f"Total de páginas lidas: {len(documentos_totais)}")
    
    # Fatiar (Chunks menores aumentam a precisão)
    divisor = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = divisor.split_documents(documentos_totais)
    print(f"Gerados {len(splits)} trechos de texto.")
    
    # Upload para o Pinecone
    print("🚀 Enviando para a nuvem (Pinecone)... Isso pode demorar um pouco.")
    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("✅ Concluído! Sua base de conhecimento está online.")

if __name__ == "__main__":
    carregar_tudo()