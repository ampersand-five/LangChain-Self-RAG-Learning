
from typing import Dict, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    # The 'keys' variable name here is arbitrary. It becomes the parameter name for the
    # input.
    # Example:
    # The inputs should be a dictionary, because the state is a TypedDict
    #              vvv
      # inputs = {"keys": {"question": "Explain how the different types of agent memory work?"}}
      # output_1 = app.invoke(inputs)
    keys: Dict[str, any]