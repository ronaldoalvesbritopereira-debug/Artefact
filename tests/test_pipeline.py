from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from artefact_rag.pipeline import SupportPipeline
from artefact_rag.routing import RouteDecision
from artefact_rag.settings import Settings
from artefact_rag.sql_generation import SQLStatement


class _AnswerState(MessagesState):
    context: str


class _Router:
    def __init__(self, decisions: list[RouteDecision]) -> None:
        self.decisions = decisions
        self.inputs: list[dict[str, Any]] = []

    def invoke(self, inputs: dict[str, Any]) -> RouteDecision:
        self.inputs.append(inputs)
        return self.decisions.pop(0)


class _LLM:
    def with_structured_output(self, schema: type[SQLStatement]) -> RunnableLambda:
        return RunnableLambda(
            lambda _: schema(query="SELECT product_name FROM products")
        )


class _SQLDatabase:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.queries: list[str] = []

    def get_table_info(self) -> str:
        return "CREATE TABLE products (product_name TEXT)"

    def run(self, query: str) -> str:
        self.events.append("sql")
        self.queries.append(query)
        return "Yamaha C40"


def _answer_graph():
    workflow = StateGraph(_AnswerState)

    def answer(state: _AnswerState) -> dict[str, list[BaseMessage]]:
        return {
            "messages": [
                AIMessage(
                    content=f"{state['context']} | {state['messages'][-1].content}"
                )
            ]
        }

    workflow.add_node("answer", answer)
    workflow.add_edge(START, "answer")
    workflow.add_edge("answer", END)
    return workflow.compile(checkpointer=InMemorySaver())


def _pipeline(
    tmp_path: Path,
    decision: RouteDecision,
    events: list[str] | None = None,
) -> tuple[SupportPipeline, _Router, _SQLDatabase]:
    settings = Settings(
        data_dir=tmp_path,
        database_path=tmp_path / "test.sqlite3",
        vector_store_dir=tmp_path / "vector_store",
    )
    router = _Router([decision])
    database = _SQLDatabase(events)
    pipeline = SupportPipeline(
        settings=settings,
        llm=_LLM(),
        router=router,
        answer_graph=_answer_graph(),
        documents=[],
        sql_database=database,
        table_names=["products"],
    )
    return pipeline, router, database


def _decision(category: str) -> RouteDecision:
    return RouteDecision(
        category=category,
        in_scope=True,
        standalone_question="Quais produtos estão cadastrados?",
    )


def test_sql_flow_executes_read_only_query_and_uses_conversation_history(
    tmp_path: Path,
) -> None:
    pipeline, router, database = _pipeline(tmp_path, _decision("SQL"))

    first = pipeline.ask("Quais produtos estão cadastrados?", "conversation")
    router.decisions.append(_decision("SQL"))
    second = pipeline.ask("E quais são os preços?", "conversation")

    assert first.category == second.category == "SQL"
    assert first.sql_query == "SELECT product_name FROM products"
    assert database.queries == [first.sql_query, second.sql_query]
    assert "Yamaha C40" in first.answer
    assert len(router.inputs[1]["history"]) == 2


def test_hybrid_flow_unloads_llm_then_combines_document_and_sql_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    pipeline, _, database = _pipeline(tmp_path, _decision("HYBRID"), events)

    class _OllamaClient:
        def __init__(self, host: str) -> None:
            assert host == pipeline.settings.ollama_base_url

        def generate(self, **kwargs: Any) -> None:
            assert kwargs == {
                "model": pipeline.settings.llm_model,
                "prompt": "",
                "keep_alive": 0,
            }
            events.append("unload")

    def retrieve(*args: Any) -> list[Document]:
        events.append("retrieve")
        return [
            Document(
                page_content="A troca segue a política da loja.",
                metadata={"source": "politicas.pdf", "page": 2},
            )
        ]

    monkeypatch.setattr("artefact_rag.pipeline.Client", _OllamaClient)
    monkeypatch.setattr("artefact_rag.pipeline.retrieve_documents", retrieve)

    result = pipeline.ask("Posso trocar um produto?", "hybrid")

    assert result.category == "HYBRID"
    assert result.sql_query == "SELECT product_name FROM products"
    assert "A troca segue a política da loja." in result.answer
    assert "Yamaha C40" in result.answer
    assert events == ["unload", "retrieve", "sql"]
    assert len(result.sources) == 2
