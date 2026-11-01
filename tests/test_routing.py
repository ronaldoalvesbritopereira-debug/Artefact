from __future__ import annotations

import pytest
from pydantic import ValidationError

from artefact_rag.routing import RouteDecision


@pytest.mark.parametrize("category", ["RAG", "SQL", "HYBRID"])
def test_route_decision_accepts_required_categories(category: str) -> None:
    decision = RouteDecision(
        category=category,
        in_scope=True,
        standalone_question="Qual é a política de troca?",
    )
    assert decision.category == category


def test_route_decision_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            category="OTHER",
            in_scope=False,
            standalone_question="Pergunta fora do escopo",
        )
