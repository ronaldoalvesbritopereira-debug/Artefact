# Assistente de atendimento — Empório da Música

Aplicação de atendimento em português que combina as políticas da loja, o catálogo e
os dados operacionais para responder perguntas sobre instrumentos, pedidos e regras
de atendimento. O sistema usa um modelo local via Ollama e apresenta uma interface
conversacional em Streamlit.

## Arquitetura

1. **Dados estruturados:** os CSVs em `Desafio Técnico AI Engineer 2026/data` são
   carregados automaticamente em SQLite. O Pandas infere os tipos das colunas; um
   fingerprint detecta alterações nos arquivos e evita recriar o banco sem necessidade.
2. **Documentos:** `PyPDFLoader` extrai as páginas dos PDFs. O
   `RecursiveCharacterTextSplitter` preserva os metadados de origem e página ao
   particionar o conteúdo.
3. **Recuperação:** BM25 e BGE-M3 recuperam documentos em paralelo. O
   `EnsembleRetriever` combina as posições usando Reciprocal Rank Fusion, e
   `CrossEncoderReranker` com BGE-Reranker-v2-M3 seleciona o trecho mais relevante
   (`k = 1`) para o contexto final. O índice semântico é persistido localmente.
4. **Roteamento:** uma chamada estruturada à LLM classifica a pergunta como RAG, SQL
   ou HYBRID e reescreve referências de perguntas de acompanhamento. Assuntos fora
   do escopo recebem uma resposta curta sem consulta às fontes.
5. **Consultas SQL:** `SQLDatabase` disponibiliza o esquema à LLM e executa a consulta
   gerada. A validação aceita somente uma instrução iniciada por `SELECT` ou `WITH`;
   a conexão SQLite também é aberta em modo somente leitura.
6. **Histórico e resposta:** um fluxo determinístico de um nó com `InMemorySaver`
   mantém o histórico em memória durante a sessão. A resposta recebe somente o trecho
   recuperado e/ou o resultado SQL, e deve informar quando as fontes não contêm os
   dados necessários.
7. **Interface:** Streamlit mostra a origem da resposta, as páginas consultadas e,
   para perguntas SQL, a consulta gerada.

O roteamento usa uma cadeia determinística em vez de um agente autônomo. O Ollama
executa `qwen3:8b`; os embeddings e o reranker usam os modelos BGE definidos na
configuração. Os vetores BGE-M3 incluídos em `artifacts/` são reaproveitados somente
quando o hash do PDF, as páginas, o modelo e as dimensões conferem. O
`InMemoryVectorStore` do LangChain mantém o índice vetorial, serializado em JSON.
O índice e o banco SQLite ficam em `.artifacts/` e não precisam ser versionados.

## Tecnologias

- Python 3.10+
- LangChain e LangGraph: `ChatOllama`, saída estruturada, loaders, splitters,
  retrievers, compressão/reranking, SQLDatabase e histórico de mensagens
- Ollama: `qwen3:8b`
- BGE-M3 e BGE-Reranker-v2-M3
- SQLite, SQLAlchemy e Pandas
- InMemoryVectorStore e Streamlit

## Instalação

No PowerShell, na raiz do repositório:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -e ".[test]"
```

Se a política de execução do PowerShell impedir a ativação do ambiente, use o
executável diretamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -e ".[test]"
```

## Configuração do Ollama

Instale e inicie o Ollama. Baixe o modelo principal uma vez:

```powershell
ollama pull qwen3:8b
```

O serviço deve estar acessível em `http://127.0.0.1:11434`. Na primeira execução,
os modelos BGE são carregados pelo Hugging Face; é necessária conexão à internet
para baixar os pesos caso ainda não estejam no cache local. O arquivo
`.env.example` lista as variáveis opcionais para ajustar o endereço do Ollama,
os diretórios, os identificadores dos modelos e a alocação de GPU. Por padrão, os
modelos BGE usam CPU e o Qwen usa até 20 camadas de GPU, deixando o restante na CPU.
O Qwen permanece carregado durante chamadas consecutivas de classificação e resposta;
antes da recuperação BGE, o pipeline o descarrega para reduzir o uso simultâneo de
memória.
`ARTEFACT_LLM_NUM_GPU` pode ser ajustado no `.env` conforme a memória disponível;
use `0` para executar o Qwen somente na CPU.

## Execução

Com o ambiente virtual ativado:

```powershell
streamlit run app.py
```

O Streamlit exibirá o endereço local da interface. Na primeira inicialização, o
projeto cria o SQLite e o índice semântico local, se necessário. Os arquivos-fonte
permanecem inalterados.

Para validar a instalação:

```powershell
pytest
```

## Exemplos de perguntas

- **RAG:** “Qual é o prazo para arrependimento de uma compra online?”
- **SQL:** “Quais produtos estão cadastrados e qual é o estoque de cada um?”
- **HYBRID:** “Posso trocar o produto do meu pedido e há unidades disponíveis?”

## Estrutura

```text
app.py
src/artefact_rag/
  database.py       # Importação CSV, SQLite e consultas somente leitura
  documents.py      # Loader de PDFs e divisão em documentos
  pipeline.py       # Roteamento e execução dos fluxos
  retrieval.py      # BM25, BGE-M3, RRF, reranking e persistência
  routing.py        # Classificação estruturada
  settings.py       # Configurações centralizadas
  sql_generation.py # Esquema de saída da consulta SQL
artifacts/          # Embeddings de páginas e metadados de compatibilidade
Desafio Técnico AI Engineer 2026/data/     # PDFs e CSVs fornecidos
tests/
```

## Limitações conhecidas

- A extração de PDF é textual; não há OCR nem tratamento especializado de tabelas ou
  imagens.
- A classificação e a geração de SQL dependem da qualidade do modelo local. A conexão
  e a validação restringem operações SQL, mas não garantem que toda pergunta seja
  classificada ou interpretada corretamente.
- O histórico é mantido em memória e não persiste após o encerramento da sessão.
- A busca e os modelos BGE podem consumir recursos consideráveis na primeira execução.
