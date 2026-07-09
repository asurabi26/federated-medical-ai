"""
LangGraph orchestrator that routes a user query to the right tool(s) - image
classification, federated retrieval, or both - and synthesizes a grounded answer.

Design principle carried through from earlier discussion: the agent must ground its
answer in what the tools actually returned, and must surface cross-hospital conflicts
explicitly rather than silently resolving them. The system prompt enforces this, and
the graph structure enforces that tool results are always available before the final
answer is generated (the LLM physically cannot skip straight to answering).

Requires OPENAI_API_KEY (or swap ChatOpenAI for another LangChain-compatible chat
model - Anthropic, local via Ollama, etc. The graph structure doesn't change).
"""

import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent.tools import FederatedRetrievalTool, ImageClassifierTool, TOOL_DESCRIPTIONS

SYSTEM_PROMPT = f"""You are a clinical research assistant orchestrating a federated \
medical AI system. You have access to a federated-trained image classifier and a \
federated retrieval system that searches multiple hospitals' private clinical notes \
independently.

{TOOL_DESCRIPTIONS}

Rules you must follow:
- This is a research/portfolio system, NOT a diagnostic tool. Never phrase output as a
  diagnosis or treatment recommendation. Frame classifier output as "the model flagged"
  or "the model's top prediction was," not "the patient has."
- Ground every claim in what a tool actually returned. Never state a finding or a case
  detail that didn't come from a tool result.
- If retrieve_case_notes returns possible_conflict=true, you MUST explicitly state the
  conflict_note in your answer, not silently pick one source.
- If you don't have enough information from the tools to answer, say so plainly rather
  than filling the gap with a plausible-sounding guess.
"""


class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    image_provided: bool
    query: str
    classification_result: Optional[Dict]
    retrieval_result: Optional[Dict]


def build_agent_graph(image_classifier: Optional[ImageClassifierTool],
                       retriever_tool: Optional[FederatedRetrievalTool],
                       model_name: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0)

    def router_node(state: AgentState) -> AgentState:
        """Decides which tools this query needs, using the LLM as a lightweight classifier."""
        routing_prompt = (
            f"User query: {state['query']}\n"
            f"An image was {'provided' if state['image_provided'] else 'NOT provided'}.\n\n"
            "Respond with exactly one of: IMAGE_ONLY, NOTES_ONLY, BOTH, NEITHER - "
            "indicating which tools are needed to answer this query."
        )
        response = llm.invoke([SystemMessage(content=routing_prompt)])
        decision = response.content.strip().upper()
        state["messages"].append(AIMessage(content=f"[routing decision: {decision}]"))
        state["_routing_decision"] = decision
        return state

    def classify_node(state: AgentState) -> AgentState:
        # image_tensor would be attached to state by the caller (see api/main.py)
        if image_classifier is not None and state.get("_image_tensor") is not None:
            state["classification_result"] = image_classifier.classify(state["_image_tensor"])
        return state

    def retrieve_node(state: AgentState) -> AgentState:
        if retriever_tool is not None:
            state["retrieval_result"] = retriever_tool.retrieve(state["query"])
        return state

    def synthesize_node(state: AgentState) -> AgentState:
        context_parts = [f"User query: {state['query']}"]
        if state.get("classification_result"):
            context_parts.append(f"Image classifier result: {state['classification_result']}")
        if state.get("retrieval_result"):
            context_parts.append(f"Retrieved case notes: {state['retrieval_result']}")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(context_parts)),
        ]
        response = llm.invoke(messages)
        state["messages"].append(response)
        return state

    def route_decision(state: AgentState) -> str:
        decision = state.get("_routing_decision", "NEITHER")
        if decision == "IMAGE_ONLY":
            return "classify"
        if decision == "NOTES_ONLY":
            return "retrieve"
        if decision == "BOTH":
            return "classify"  # goes classify -> retrieve -> synthesize
        return "synthesize"

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "classify": "classify", "retrieve": "retrieve", "synthesize": "synthesize"
    })
    graph.add_conditional_edges(
        "classify",
        lambda s: "retrieve" if s.get("_routing_decision") == "BOTH" else "synthesize",
        {"retrieve": "retrieve", "synthesize": "synthesize"},
    )
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_agent(graph, query: str, image_tensor=None) -> Dict:
    initial_state: AgentState = {
        "messages": [],
        "image_provided": image_tensor is not None,
        "query": query,
        "classification_result": None,
        "retrieval_result": None,
    }
    initial_state["_image_tensor"] = image_tensor

    final_state = graph.invoke(initial_state)
    final_answer = final_state["messages"][-1].content if final_state["messages"] else ""

    return {
        "answer": final_answer,
        "classification_result": final_state.get("classification_result"),
        "retrieval_result": final_state.get("retrieval_result"),
        "tools_used": [
            name for name, val in [
                ("classify_scan", final_state.get("classification_result")),
                ("retrieve_case_notes", final_state.get("retrieval_result")),
            ] if val is not None
        ],
    }
