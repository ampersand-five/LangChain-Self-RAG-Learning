import pprint

from langgraph.graph import END, StateGraph

import nodes
import edges
from state import GraphState

workflow = StateGraph(GraphState)

# Function to add a node, the arguments are the name of the node and the function to
# call when this node is reached.
workflow.add_node(key="retrieve", action=nodes.retrieve)  # retrieve
workflow.add_node(key="grade_documents", action=nodes.grade_documents)  # grade documents
workflow.add_node(key="generate", action=nodes.generate)  # generate
workflow.add_node(key="transform_query", action=nodes.transform_query)  # transform_query
workflow.add_node(key="prepare_for_final_grade", action=nodes.prepare_for_final_grade)  # passthrough

# Build graph

# Adding node, this is the entry point of the graph
workflow.set_entry_point(key="retrieve")

# Creates an edge from one node to the next. This means that output of the first node
# will be passed to the next node. It takes two arguments.
# - start_key: A string representing the name of the start node. This key must have
# already been registered in the graph.
# - end_key: A string representing the name of the end node. This key must have already
# been registered in the graph.
workflow.add_edge(start_key="retrieve", end_key="grade_documents")

# This method adds conditional edges. What this means is that only one of the downstream
# edges will be taken, and which one that is depends on the results of the start node.
# This takes three arguments:
# - start_key: A string representing the name of the start node. This key must have
# already been registered in the graph.
# - condition: A function to call to decide what to do next. The input will be the
# output of the start node. It should return a string that is present in
# 'conditional_edge_mapping' and represents the edge to take.
# - conditional_edge_mapping: A mapping of string to string. The keys should be strings
# that may be returned by condition. The values should be the downstream node to call if
# that condition is returned.
workflow.add_conditional_edges(
  start_key="grade_documents",
  condition=edges.decide_to_generate,
  conditional_edge_mapping={
      "transform_query": "transform_query",
      "generate": "generate",
  }
)

workflow.add_edge(start_key="transform_query", end_key="retrieve")

workflow.add_conditional_edges(
    start_key="generate",
    condition=edges.grade_generation_v_documents,
    conditional_edge_mapping={
        "supported": "prepare_for_final_grade",
        "not supported": "generate",
    }
)

workflow.add_conditional_edges(
    start_key="prepare_for_final_grade",
    condition=edges.grade_generation_v_question,
    conditional_edge_mapping={
        # END is a special node marking that the graph should finish.
        "useful": END,
        "not useful": "transform_query",
    },
)

# Finally, we compile it. This compiles it into a LangChain Runnable, meaning it can be
# used as you would any other runnable.
app = workflow.compile()

# Note: The ascii graph doesn't draw the right graph, and the PNG fails on the function
# draw_png() saying the function doesn't exist.
# Uncomment the blow block to try drawing the graph.
# Taken from https://github.com/langchain-ai/langgraph/blob/main/examples/visualization.ipynb

# # Draw the graph
# # Ascii art drawing
# # $ poetry add grandalf
# app.get_graph().print_ascii()
# # PNG drawing
# # $ brew install graphviz
# # On mac: 
# #    $ export CFLAGS="-I $(brew --prefix graphviz)/include"
# #    $ export LDFLAGS="-L $(brew --prefix graphviz)/lib"
# # $ poetry add pygraphviz
# # $ poetry add ipython
# from IPython.display import Image
# Image(app.get_graph().draw_png())

# We can now use it! This now exposes the same interface as all other LangChain
# runnables. This runnable accepts a list of messages.

# Another way to format the input. In Example 1 and 2 we don't use this format, but
# it's another way to format the input.
# from langchain_core.messages import HumanMessage
# inputs = {"messages": [HumanMessage(content="Explain how the different types of agent memory work?")]}

# Example 1 - Full invoke
print('\nExample 1 - Full Invoke\n')
inputs = {"keys": {"question": "Explain how the different types of agent memory work?"}}
output_1 = app.invoke(inputs)
# Output example 1
pprint.pprint(output_1['keys']['generation'])


# Example 2 - Stream node results as they happen
print('\nExample 2 - Stream Node Results\n')
# Note: The output from the invoke and stream seem the same. What's happening is you can
# access it at each node. Allowing for any changes or modifications to be made along the
# way.

inputs = {"keys": {"question": "Explain how the different types of agent memory work?"}}
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        # Node
        print(f"Output from node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)

# Output example 2
pprint.pprint(value["keys"]["generation"])