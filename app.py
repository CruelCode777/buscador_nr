import streamlit as st
import os  # <--- Importante
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Busca NR", page_icon="🔍", layout="centered")

# --- 2. ESTILO VISUAL (CSS) ---
def aplicar_estilo_google():
    st.markdown("""
    <style>
    /* Importa a fonte Roboto (padrão do Google/Android) */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Esconde o menu do Streamlit, rodapé e cabeçalho padrão */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Centraliza o logo na tela inicial */
    .main-container {
        text-align: center;
        padding-top: 50px;
    }
    
    /* Classes de cores do Google */
    .g-blue {color: #4285F4;}
    .g-red {color: #EA4335;}
    .g-yellow {color: #FBBC05;}
    .g-green {color: #34A853;}
    
    /* Estilo do Logo Grande */
    .logo-large {
        font-size: 80px;
        font-weight: bold;
        letter-spacing: -3px;
    }
    
    /* Estilo do Logo Pequeno (para quando já tem chat) */
    .logo-small {
        font-size: 30px;
        font-weight: bold;
        letter-spacing: -1px;
    }

    /* Subtítulo */
    .subtitle {
        color: #5f6368;
        font-size: 18px;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

# Função para desenhar o Logo Colorido
def renderizar_logo(tamanho="grande"):
    css_class = "logo-large" if tamanho == "grande" else "logo-small"
    
    html = f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <span class="{css_class} g-blue">B</span>
        <span class="{css_class} g-red">u</span>
        <span class="{css_class} g-yellow">s</span>
        <span class="{css_class} g-blue">c</span>
        <span class="{css_class} g-green">a</span>
        <span class="{css_class} g-red">r</span>
        <span class="{css_class} g-blue" style="margin-left: 10px;">N</span>
        <span class="{css_class} g-green">R</span>
    </div>
    """
    
    if tamanho == "grande":
        html += '<div style="text-align: center; color: #5f6368; margin-bottom: 40px;">Inteligência Artificial aplicada à Segurança do Trabalho</div>'
        
    st.markdown(html, unsafe_allow_html=True)

# Aplica o CSS
aplicar_estilo_google()

# --- 3. SEGREDOS E CONFIGURAÇÃO ---
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    pinecone_key = st.secrets["PINECONE_API_KEY"]
except FileNotFoundError:
    st.error("Chaves de API não configuradas nos Secrets.")
    st.stop()

# --- 4. CONEXÃO COM BANCO DE DADOS ---
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
    st.error(f"Erro de conexão: {e}")
    st.stop()

# --- 5. LÓGICA DE EXIBIÇÃO ---

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# LÓGICA VISUAL:
# Se não tem mensagens, mostra o logo GIGANTE (Home Page)
# Se tem mensagens, mostra o logo PEQUENO (Header)
if len(st.session_state.messages) == 0:
    st.write("") # Espaço vazio para empurrar um pouco para baixo
    st.write("") 
    renderizar_logo(tamanho="grande")
else:
    renderizar_logo(tamanho="pequeno")
    st.divider() # Uma linha sutil separando o cabeçalho do chat

# Mostra histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. CAMPO DE BUSCA (CHAT) ---
if prompt := st.chat_input("Pesquise nas normas (ex: Cinto de segurança NR 35)"):
    
    # Adiciona pergunta do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Processamento da IA
    with st.chat_message("assistant"):
        # Ícone de carregamento personalizado
        with st.spinner("Pesquisando na base de dados..."):
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                docs = retriever.invoke(prompt)
                
                if not docs:
                    response_text = "Sua pesquisa não retornou resultados nas NRs indexadas."
                else:
                    context_text = ""
                    sources = set()
                    for doc in docs:
                        src = doc.metadata.get('source', 'NR Desconhecida')
                        context_text += f"{doc.page_content}\n(Fonte: {src})\n---\n"
                        sources.add(src)

                    system_prompt = """
                    Você é um Assistente Técnico (Estilo Google Search AI).
                    Responda de forma direta, objetiva e formatada.
                    
                    Contexto Encontrado: {context}
                    Pergunta: {question}
                    """
                    prompt_template = ChatPromptTemplate.from_template(system_prompt)
                    llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
                    chain = prompt_template | llm
                    response = chain.invoke({"context": context_text, "question": prompt})
                    
                    response_text = response.content + f"\n\n\n*Fontes: {', '.join(sources)}*"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            except Exception as e:
                st.error(f"Erro: {e}")
