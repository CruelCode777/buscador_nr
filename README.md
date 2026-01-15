# 🛡️ Assistente Inteligente de Normas Regulamentadoras (NRs)

> Uma aplicação de IA Generativa (RAG) capaz de consultar, cruzar dados e responder dúvidas sobre as 38 Normas Regulamentadoras de Segurança do Trabalho em segundos.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://consultornrs.streamlit.app/)

## 🎯 O Problema
Profissionais de Segurança do Trabalho (HSE) gastam horas consultando manualmente dezenas de PDFs para encontrar diretrizes específicas. 
A busca por palavras-chave (Ctrl+F) muitas vezes falha quando a terminologia exata não é usada.

## 💡 A Solução
Desenvolvi um **Assistente Virtual** que utiliza **Busca Semântica**. Isso significa que ele entende o *significado* da pergunta, não apenas as palavras.
- **Exemplo:** Se você perguntar *"O que preciso para evitar quedas?"*, ele buscará diretrizes sobre cintos, guarda-corpos e ancoragem na NR-35 e NR-18, mesmo que a palavra "queda" não esteja no parágrafo.

## 🛠️ Tecnologias Utilizadas
Este projeto aplica o conceito de **RAG (Retrieval-Augmented Generation)** utilizando uma stack moderna e de baixo custo:

* **Linguagem:** Python 🐍
* **Interface:** Streamlit (Web App)
* **Cérebro (LLM):** Llama 3.3 (via Groq API) - Para raciocínio e resposta natural.
* **Memória (Vector DB):** Pinecone - Para armazenar e indexar todas as NRs na nuvem.
* **Embeddings:** HuggingFace (`sentence-transformers`) - Para transformar textos técnicos em vetores matemáticos.
* **Framework:** LangChain - Para orquestrar o fluxo de dados.

## 🚀 Como Funciona
1.  **Ingestão:** Um script Python lê os PDFs oficiais das NRs.
2.  **Vetorização:** O texto é quebrado em fragmentos e convertido em vetores numéricos.
3.  **Armazenamento:** Os dados são salvos no Pinecone (Nuvem).
4.  **Consulta:** Quando o usuário pergunta, o sistema busca os trechos mais relevantes matematicamente.
5.  **Resposta:** A IA (Llama 3) lê os trechos e formula uma resposta técnica, citando a fonte (Item da Norma).

## 👷 Sobre o Autor
**Carlos Alberto de Andrade Junior**
*Técnico em Segurança do Trabalho & Estudante de Engenharia Elétrica*

Estou unindo minha experiência de campo em HSE com novas tecnologias para criar soluções que salvam vidas e otimizam tempo.

[LinkedIn](https://www.linkedin.com/in/carlos-andrade-41363a32/)
