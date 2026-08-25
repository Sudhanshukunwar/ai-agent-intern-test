# Aster & Row Support Agent

An AI-powered customer support agent for Aster & Row, an ecommerce company that sells bags, drinkware, and travel accessories.

The agent can answer questions using the company's knowledge base, look up orders through a protected order tool, maintain context across multiple turns, and avoid making up information when the available data is not enough.

## What I built

The main goal of this project was to build a support agent that is useful but also careful about company policies and customer data.

The agent supports:

- Knowledge-base question answering
- Source citations
- Order lookup
- Multi-turn conversations
- Human handoff when information is missing or conflicting
- Protection against prompt-injection content inside retrieved documents
- Customer-safe order information
- Deterministic evaluation of agent responses

The knowledge base contains the company's current and older policies, product information, shipping information, warranty rules, and internal migration content.

---

## Setup

Clone the repository and enter the project directory:

```bash
git clone <your-repository-url>
cd ai-agent-intern-test

The basic flow is:

User
  |
  v
Agent
  |
  +--> Build retrieval query from recent conversation
  |
  +--> BM25 knowledge-base search
  |
  +--> Apply policy/status weighting
  |
  +--> Wrap retrieved content as untrusted data
  |
  +--> Gemini API
          |
          +--> Normal answer
          |
          +--> order_lookup tool
                    |
                    +--> Normalize order ID
                    +--> Validate order ID
                    +--> Return only customer-safe fields
                    +--> Hide stale tracking/ETA information
                    |
                    v
                 Gemini
  |
  +--> Parse SOURCES / HANDOFF trailer
  |
  v
Final response

The main application files are:

app/
├── agent.py
├── bm25.py
├── cli.py
├── config.py
├── llm_client.py
├── orders.py
├── response_format.py
├── retrieval.py
├── security.py
├── simple_yaml.py
└── tools_schema.py
Retrieval

The retrieval system uses BM25 over the knowledge-base documents.

The documents contain metadata such as status and policy_authority. Active official documents receive a higher weight, while superseded, draft, and internal material receive lower weights.

Importantly, these documents are not simply deleted from retrieval. Some of them are intentionally useful for testing whether the agent can recognize that a document is not authoritative.

Retrieved content is passed to the model inside an explicit:

<untrusted_data>
...
</untrusted_data>

block.

This helps prevent text inside a knowledge-base document from being treated as an instruction.

Order lookup and privacy

The model does not receive the complete orders.json record.

Instead, the order tool:

Normalizes the order ID.
Validates the expected ORD-#### format.
Looks up the order.
Selects only customer-safe fields.
Removes sensitive/internal information.
Handles cancelled and returned orders safely.
Avoids exposing stale delivery information.
Flags exceptional orders for human review.

For example, if an order is cancelled, an old tracking number or delivery estimate is not presented as if the order is still arriving.

The tool result is also treated as untrusted data before being passed back to Gemini.

Multi-turn conversations

The agent keeps clean user and assistant messages in the current session.

Recent user turns are included when building the next retrieval query.

For example:

User: Do you ship internationally?

Agent: ...

User: What about Canada?

The second question can use the previous conversation context instead of treating "What about Canada?" as an unrelated question.

Sessions are separated using session IDs.

Security

The agent includes several protections against untrusted content.

Retrieved documents and tool results are wrapped as untrusted data.

The application also scans retrieved content for instruction-like patterns.

This is particularly important because the supplied knowledge base contains an internal migration document with an embedded prompt-injection example.

The agent is instructed not to follow instructions found inside retrieved company content.

The project also contains tests covering prompt-injection and prompt-exfiltration attempts.

Evaluation

The evaluation suite contains:

15 supplied visible cases
6 original cases created for this project
21 cases in total

Run the final evaluation with:

python -m evaluation.run_eval --variant final

Run the baseline evaluation with:

python -m evaluation.run_eval --variant baseline

The results are written under:

evaluation/results/

The evaluation checks things such as:

Retrieval
Groundedness
Source citations
Tool usage
Tool arguments
Privacy
Multi-turn behavior
Security
Human handoff
Refusal to invent unsupported information

The checks are deterministic and do not use another LLM as a grader.

Evaluation status

The evaluation harness itself runs successfully.

The offline mock evaluation is available as a smoke test:

python -m evaluation.run_eval --variant final --mock

The mock mode is only intended to verify that the evaluation pipeline, tool loop, session handling, and result generation work correctly. Its response generator is intentionally simple, so its score should not be interpreted as the quality of the real Gemini agent.

Evaluation results
Category	Final
Retrieval	Live evaluation limited by Gemini quota
Groundedness	Live evaluation limited by Gemini quota
Tool use	Live evaluation limited by Gemini quota
Privacy	Live evaluation limited by Gemini quota
Multi-turn	Live evaluation limited by Gemini quota
Security	Live evaluation limited by Gemini quota
Original evaluation cases

I added six original cases to complement the supplied evaluation cases.

These cover situations such as:

Final-sale items that arrive damaged
Order ID punctuation and normalization
Unsupported questions where the agent should avoid guessing
Multi-turn context
Human handoff situations
Security and untrusted-content behavior

They are stored in:

evaluation/own_cases.json
Bug diary
1. Order IDs with punctuation
Problem

An order ID such as:

ORD-1007?

could be rejected even though the actual order was:

ORD-1007
Cause

The normalization logic handled whitespace and separators but did not remove punctuation commonly attached to an order ID in normal conversation.

Fix

The normalization step now removes common trailing punctuation before validating the order ID.

Regression test

The behavior is covered by tests in:

tests/test_orders.py
2. Handoff trailer parsing
Problem

The response parser expected:

HANDOFF: false

exactly.

A response such as:

HANDOFF: false.

could fail to parse correctly.

Cause

The parser was too strict about punctuation after the boolean value.

Fix

The parser now accepts minor punctuation after the value.

Regression test

Covered in:

tests/test_response_format.py
3. Weak retrieval confidence
Problem

BM25 could sometimes return a document that shared words with the question but did not actually answer it.

For example, a question about whether materials were vegan could retrieve general bag-care information simply because both contained words related to bags or materials.

Cause

BM25 is lexical. It does not understand the meaning of the question.

Fix

Retrieved chunks now receive a confidence indicator based on their retrieval score.

The system prompt also tells Gemini not to treat weak retrieval results as sufficient evidence for unsupported claims.

Regression tests

Covered in:

tests/test_retrieval.py

This remains a known limitation because BM25 cannot completely replace semantic retrieval.

Known limitations

There are several things I would improve before using this in production.

Better retrieval

BM25 works well for this small knowledge base, but it can miss paraphrased questions.

For a larger system, I would use hybrid retrieval:

BM25 + embeddings + reranking
Persistent sessions

Conversation sessions currently live in memory.

A production version would use persistent storage and include session expiration/cleanup.

API retry handling

The Gemini API can return temporary errors or quota errors.

A production implementation should add proper retry/backoff handling and a user-friendly fallback response.

Structured responses

The current SOURCES: and HANDOFF: trailer is parsed from model text.

A structured output mechanism would be more reliable than depending on a text convention.

Evaluation

The current evaluator uses deterministic checks such as keyword, regex, source, and tool-call assertions.

These are useful for repeatability but are not a complete semantic evaluation of an AI response.

AI coding tools

I used Claude during development to help build and debug the project.

Claude was mainly used for:

Project structure
Initial implementation
Retrieval logic
Tool-calling flow
Security handling
Evaluation cases
Unit tests
Debugging
README preparation

The final application runtime uses Google Gemini API, not Claude.

Example of an AI-generated suggestion that was wrong

One issue I found while developing the retrieval system was related to using a simple BM25 score threshold as a confidence signal.

The initial idea was that a fixed score threshold would reliably separate useful retrieval results from irrelevant ones.

When I tested it with an out-of-scope question, some irrelevant documents still received scores above the threshold.

Instead of treating the threshold as a perfect relevance check, I kept it as a weaker confidence signal and documented the limitation.

This was a good example of why the generated code still needs to be tested against the actual data.

Demo

The demo shows the agent running from the command line.

It covers:

A knowledge-base question with source citations.
An order lookup using the order_lookup tool.
A multi-turn conversation where the second question uses previous context.
A question where the agent should not guess and instead recommends human help.
The evaluation suite.

Example commands used during the demo:

python -m app.cli chat

For detailed tool/retrieval traces:

python -m app.cli chat --debug

[ Watch the Aster & Row Support Agent Demo](https://drive.google.com/file/d/1EXX0y3rAY7emeiv1jeNiWJv_vplsHZzo/view?usp=sharing)
The project contains unit tests for:

Order handling
Retrieval
Response formatting
Security
Standard-library replacements

Run them with:

python -m pytest tests/ -v
Repository structure
.
├── README.md
├── .env.example
├── requirements.txt
│
├── app/
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
├── knowledge-base/
│   └── company policy and product documents
│
├── evaluation/
│   ├── visible-cases.json
│   ├── own_cases.json
│   ├── concept_patterns.py
│   ├── run_eval.py
│   └── results/
│
├── system_prompts/
│   ├── system_prompt_final.txt
│   └── system_prompt_baseline.txt
│
└── tests/
    ├── test_orders.py
    ├── test_retrieval.py
    ├── test_response_format.py
    ├── test_security.py
    └── test_stdlib_replacements.py
