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
os diretórios e os identificadores dos modelos. O Ollama decide automaticamente
como distribuir o Qwen entre GPU e CPU; os modelos BGE usam CPU.
O Qwen permanece carregado durante chamadas consecutivas de classificação e resposta;
antes da recuperação BGE, o pipeline o descarrega para reduzir o uso simultâneo de
memória.

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

## Decisões técnicas

Antes do desenvolvimento, pesquisei abordagens recentes para arquiteturas RAG. A
funcionalidade “Pesquisa Aprofundada” do ChatGPT foi usada como ponto de partida para
localizar artigos científicos; depois, li os trabalhos selecionados e usei essa
pesquisa para orientar o planejamento. A experiência anterior com RAG durante a
pós-graduação também influenciou as decisões e ajudou a avaliar alternativas de
recuperação.

Escolhi RAG porque parte importante das perguntas do desafio depende de informações
descritas nas políticas da loja, e não apenas de dados tabulares. Recuperar os trechos
relacionados à pergunta permite responder com base no documento fornecido, sem enviar
todo o PDF ao modelo a cada interação. Já preços, estoque e demais informações
operacionais são consultados nos CSVs por SQL. Assim, cada tipo de pergunta usa a fonte
mais apropriada; perguntas que combinam regras e dados podem seguir o fluxo HYBRID.

Como o prazo de desenvolvimento era de apenas dois dias, priorizei uma arquitetura
que pudesse ser implementada e testada nesse período, sem fine-tuning de modelos.
Para a busca nos documentos, considerei alternativas como MPNet e ColBERT. MPNet
costuma se beneficiar de fine-tuning para esse tipo de tarefa, o que não cabia no
prazo; ColBERT oferece uma abordagem interessante, mas acrescentaria complexidade de
instalação e configuração. Optei por combinar BM25 e BGE-M3: a busca lexical ajuda a
encontrar termos e nomes específicos, enquanto a busca semântica considera o sentido
da pergunta. A fusão dos resultados com Reciprocal Rank Fusion (RRF) reúne essas duas
perspectivas, e o BGE-Reranker-v2-M3 reordena os candidatos para selecionar o trecho
mais relevante.

O PDF é dividido em páginas e, em seguida, em trechos menores, preservando metadados
de origem e página. Embora o desafio forneça um único documento, essa organização
também deixa a recuperação preparada para receber outros PDFs. Após o reranking,
utilizo apenas um trecho (`k = 1`) como contexto final. Nos testes, fornecer vários
trechos ao modelo local aumentava as alucinações e, por vezes, dificultava a geração
de uma resposta adequada. Com um modelo de maior capacidade e uma janela de contexto
maior, seria possível reavaliar essa escolha e incluir páginas vizinhas quando fossem
necessárias para preservar o contexto.

Os prompts foram ajustados iterativamente durante os testes, buscando instruções
claras sem consumir contexto desnecessariamente. Para definir a persona e o tom da
resposta, usei como referência o manual de políticas da loja, de modo que o
atendimento permanecesse coerente com as informações fornecidas no desafio.

O roteamento foi implementado como uma cadeia determinística, e não como um agente
autônomo. Uma chamada estruturada classifica a pergunta como RAG, SQL ou HYBRID antes
da execução. Essa separação deixa explícito qual fonte deve ser consultada e facilita
inspecionar o caminho seguido pela aplicação. Perguntas fora do escopo da loja são
tratadas sem consultar documentos ou banco de dados.

Para o modelo de linguagem, escolhi o `qwen3:8b` executado localmente pelo Ollama. A
opção evita custos com APIs durante a execução e é suficiente para o protótipo, embora
tenha limitações de capacidade e de recursos disponíveis na máquina. O modelo é
utilizado tanto na classificação quanto na geração das respostas. Os prompts foram
mantidos objetivos para equilibrar as instruções de atendimento, o contexto recuperado
e a janela disponível. A resposta também deve respeitar a persona da loja e sinalizar
quando as fontes não contiverem os dados necessários.

Para os dados tabulares, escolhi SQLite por ser simples de distribuir e não exigir um
servidor separado. Os CSVs são carregados automaticamente, e o Pandas infere os tipos
das colunas para favorecer consultas sobre valores numéricos, como preços e estoque.
As consultas geradas são limitadas a uma única instrução `SELECT` ou `WITH`, e a
conexão é aberta em modo somente leitura.

Na migração para a estrutura atual, LangChain passou a fornecer integrações para
modelos, documentos, recuperação e SQL, enquanto LangGraph organiza o fluxo e o
histórico da conversa. O Streamlit foi escolhido para oferecer uma interface simples
de demonstração sem tirar o foco do funcionamento do agente. O histórico é mantido em
memória durante a sessão, evitando adicionar infraestrutura de persistência que não
era essencial para o escopo do desafio.

## Exemplos de interação

As imagens abaixo registram cinco cenários variados de uso do agente:

1. **Consulta de catálogo com filtro de preço (SQL):** [parte 1](resultados_imagens/pergunta1_parte1.png) e [parte 2](resultados_imagens/pergunta1_parte2.png).
2. **Informação sobre o endereço da loja (RAG):** [ver imagem](resultados_imagens/pergunta2.png).
3. **Consulta de preço de produto (SQL):** [ver imagem](resultados_imagens/pergunta3.png).
4. **Aplicação da política de devolução (RAG):** [ver imagem](resultados_imagens/pergunta4.png).
5. **Consulta de instrumentos de percussão (SQL):** [ver imagem](resultados_imagens/pergunta5.png).

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

## Limitações conhecidas e melhorias futuras

- A extração de PDF é textual; não há OCR nem tratamento especializado de tabelas ou
  imagens. Como melhoria, o processamento poderia preservar melhor a estrutura de
  tabelas e incluir OCR para documentos digitalizados.
- A classificação e a geração de SQL dependem da qualidade do modelo local. A conexão
  e a validação restringem operações SQL, mas não garantem que toda pergunta seja
  classificada ou interpretada corretamente. Um conjunto de testes de avaliação com
  perguntas representativas ajudaria a medir e melhorar a qualidade do roteamento,
  das consultas e das respostas.
- O histórico é mantido em memória e não persiste após o encerramento da sessão. Uma
  evolução possível é oferecer persistência por usuário, com controles de privacidade
  e opções para apagar conversas.
- A busca e os modelos BGE podem consumir recursos consideráveis na primeira execução.
  Otimizações ou opções de modelos menores poderiam facilitar a execução em máquinas
  com menos memória.

## Uso de assistentes de código

Durante o desenvolvimento foi utilizado o VS Code com GitHub Copilot no plano
gratuito, com 200 créditos disponíveis, como apoio ao trabalho no projeto. As decisões
técnicas e a validação do comportamento da aplicação permanecem sob responsabilidade
do autor.
