from __future__ import annotations

import uuid

import streamlit as st

from artefact_rag.pipeline import SupportPipeline
from artefact_rag.settings import Settings


st.set_page_config(
    page_title="Empório da Música | Atendimento",
    page_icon="🎼",
    layout="centered",
)


@st.cache_resource
def get_pipeline() -> SupportPipeline:
    return SupportPipeline.from_settings(Settings.from_env())


st.title("🎼 Empório da Música")
st.caption("Assistente de atendimento baseado nas políticas e nos dados da loja.")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Sobre este atendimento")
    st.write("As respostas podem consultar políticas, catálogo e pedidos.")
    if st.button("Nova conversa", use_container_width=True):
        st.session_state.conversation_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

with st.spinner("Carregando os dados e os modelos locais..."):
    pipeline = get_pipeline()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(f"Origem: {message['category']}")
            if message["sources"]:
                with st.expander("Fontes consultadas"):
                    for source in message["sources"]:
                        if source["type"] == "Documento":
                            st.write(
                                f"- {source['source']}, página {source['page']}"
                            )
                        else:
                            st.write(
                                f"- Banco SQLite: {', '.join(source['tables'])}"
                            )
            if message["sql_query"]:
                with st.expander("Consulta SQL gerada"):
                    st.code(message["sql_query"], language="sql")

question = st.chat_input("Como posso ajudar?")
if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Consultando as informações..."):
            result = pipeline.ask(
                question,
                session_id=st.session_state.conversation_id,
            )
    category = (
        "Fora do escopo"
        if result.category == "OUT_OF_SCOPE"
        else result.category
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "category": category,
            "sources": result.sources,
            "sql_query": result.sql_query,
        }
    )
    st.rerun()
