import pprint

from langgraph.graph import END, StateGraph

import nodes
import edges
from state import GraphState

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", nodes.retrieve)  # retrieve
workflow.add_node("grade_documents", nodes.grade_documents)  # grade documents
workflow.add_node("generate", nodes.generate)  # generate
workflow.add_node("transform_query", nodes.transform_query)  # transform_query
workflow.add_node("prepare_for_final_grade", nodes.prepare_for_final_grade)  # passthrough

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    edges.decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    edges.grade_generation_v_documents,
    {
        "supported": "prepare_for_final_grade",
        "not supported": "generate",
    },
)
workflow.add_conditional_edges(
    "prepare_for_final_grade",
    edges.grade_generation_v_question,
    {
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

# Run
inputs = {"keys": {"question": "Explain how the different types of agent memory work?"}}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")

# Final generation
pprint.pprint(value["keys"]["generation"])