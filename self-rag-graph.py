import pprint

from langgraph.graph import END, StateGraph

import nodes
import edges
from state import GraphState

workflow = StateGraph(GraphState)

# Define the node names which are key:value pairs of the node name and which function to
# call when the node is reached.
workflow.add_node("retrieve", nodes.retrieve)  # retrieve
workflow.add_node("grade_documents", nodes.grade_documents)  # grade documents
workflow.add_node("generate", nodes.generate)  # generate
workflow.add_node("transform_query", nodes.transform_query)  # transform_query
workflow.add_node("prepare_for_final_grade", nodes.prepare_for_final_grade)  # passthrough

# Build graph

# Adding node, this is the entry point of the graph
workflow.set_entry_point("retrieve")

# Add another node. The function is 'edge', but it's better called a node. Nodes have
# only one output path. Edges, aka conditional edges, have multiple output paths and the
# point of the conditional edge is to decide which path to take.
workflow.add_edge(
  # First we list the previous node that led to this node.
  "retrieve",
  # Next, we name the node that is called next.
  "grade_documents")

# Add an edge. Aka a conditional edge. This is a decision point to decide which path to
# take.
workflow.add_conditional_edges(
  # First we define previous node that led to this one. This means, these are all the
  # edges taken after the node we list here. 'grade_documents' is the listed node in this
  # case. There is both a plural and singular form of this function.
  "grade_documents",
  # Next, we pass in the function that will determine which node/edge is called next.
  edges.decide_to_generate,
  # Finally we pass in a mapping.
  # What will happen is we will call `edges.decide_to_generate`, and then the output of
  # that will be matched against the keys in this mapping. Based on which one it
  # matches, that node will then be called. The keys are strings, and the values are the
  # names of which node to go to when the listed key is matched.
  {
      "transform_query": "transform_query",
      "generate": "generate",
  }
)

workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "generate",
    edges.grade_generation_v_documents,
    {
        "supported": "prepare_for_final_grade",
        "not supported": "generate",
    }
)

workflow.add_conditional_edges(
    "prepare_for_final_grade",
    edges.grade_generation_v_question,
    {
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
# the above was taken from a tutorial, I think it needs to change the key to "question" to work?
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