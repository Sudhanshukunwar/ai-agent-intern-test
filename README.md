# Aster & Row Support Agent

A reliable RAG-based customer support agent built for the Aster & Row AI Agent Intern take-home assignment.

Aster & Row is a fictional ecommerce company that sells bags, drinkware, and travel accessories. The goal of this project is to build a small support agent that can answer questions from the supplied knowledge base, look up mock orders safely, remember relevant conversation context, and avoid making unsupported or unsafe claims.

## Features

The agent supports:

- Retrieval-Augmented Generation over the supplied Markdown knowledge base.
- Metadata-aware document retrieval.
- Preference for active and official company policies.
- Handling of superseded and non-authoritative documents.
- Detection and handling of conflicting official sources.
- Source references for policy and product answers.
- Order lookup using `data/orders.json`.
- Safe handling of unknown and malformed order IDs.
- Sanitized order results that do not expose customer or internal-only information.
- Multi-turn conversation context.
- Protection against prompt injection inside retrieved documents.
- Abstention when the supplied information is insufficient.
- Human handoff recommendations when required.
- Debug traces for retrieval, tool calls, errors, sources, and handoff decisions.
- Deterministic evaluation cases and regression tests.

---

# Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── agent.py
│   ├── bm25.py
│   ├── cli.py
│   ├── config.py
│   ├── llm_client.py
│   ├── orders.py
│   ├── response_format.py
│   ├── retrieval.py
│   ├── security.py
│   ├── simple_yaml.py
│   └── tools_schema.py
│
├── data/
│   ├── orders.json
│   └── orders-data-dictionary.md
│
├── evaluation/
│   ├── __init__.py
│   ├── concept_patterns.py
│   ├── own_cases.json
│   ├── run_eval.py
│   ├── visible-cases.json
│   └── results/
│
├── knowledge-base/
│   ├── 01-returns-policy-current.md
│   ├── 02-returns-policy-legacy.md
│   ├── 03-final-sale-and-promotions.md
│   ├── 04-damaged-or-wrong-items.md
│   ├── 05-domestic-shipping.md
│   ├── 06-international-shipping.md
│   ├── 07-warranty.md
│   ├── 08-order-changes-and-cancellations.md
│   ├── 09-trailplus-membership.md
│   ├── 10-gift-cards-and-price-adjustments.md
│   ├── 11-product-care.md
│   ├── 12-breeze-tumbler-product-card.md
│   ├── 13-support-escalation.md
│   └── 14-internal-content-migration-notes.md
│
├── system_prompts/
│   ├── system_prompt_baseline.txt
│   └── system_prompt_final.txt
│
├── .env.example
├── README.md
└── requirements.txt
