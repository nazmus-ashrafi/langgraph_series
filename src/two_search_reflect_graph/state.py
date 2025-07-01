"""State management for the retrieval graph.

This module defines the state structures used in the retrieval graph. It includes
definitions for agent state, input state, and router classification schema.
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

import hashlib
import uuid
from typing import Any, Literal, Optional, Union

from langchain_core.documents import Document
from operator import add

from collections import defaultdict

# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""



# This is the primary state of your agent, where you can store any information


@dataclass(kw_only=True)
class AgentState(InputState):
    """State of the graph."""

    plan: list[str] = field(default_factory=list)
    """A list of steps in the research plan."""

    prompt: str = field(default_factory=str)
    solution: list = field(default_factory=list)

    # -- --
    original_question: str = field(default_factory=str)
    attempted_search_queries: list = field(default_factory=list)
    relavant_search_result: str = field(default_factory=str)


    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.
