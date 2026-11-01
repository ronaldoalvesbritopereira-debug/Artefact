from __future__ import annotations

import pytest
from pydantic import ValidationError

from artefact_rag.routing import ROUTING_INSTRUCTIONS, RouteDecision


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


def test_routing_prompt_includes_all_reference_examples() -> None:
    prompt = " ".join(ROUTING_INSTRUCTIONS.split())
    examples = [
        "política de troca",
        "garantia",
        "horário de funcionamento",
        "atendimento por WhatsApp",
        "devolução de produto",
        "Takamine GD20",
        "violões abaixo de R$ 1.000",
        "categoria Guitarras",
        "promoções cadastradas",
        "quantidade de pedidos",
        "produtos disponíveis",
        "promoção Black Friday",
        "violões até R$ 1.000 com garantia",
        "parcelados em até 10 vezes",
        "via Jadlog",
        "endereço da loja",
        "bateria eletrônica",
        "horário de funcionamento e violões disponíveis",
        "política de troca e produtos em promoção",
    ]

    for example in examples:
        assert example in prompt
