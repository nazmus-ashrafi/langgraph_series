"""State management for the plan generation graph.

This module defines the state structures used in the plan generation graph. It includes
definitions for agent state and input state.
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
    entry_point: str = field(default_factory=str)
    output_path: str = field(default_factory=str)
    prompt: str = field(default_factory=str)
    visible_tests_list: list = field(default_factory=list)

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



@dataclass(kw_only=True)
class AgentState(InputState):
    """State of the plan generation graph / agent."""

    plan: list[str] = field(default_factory=list)
    """A list of steps in the research plan."""

    prompt: str = field(default_factory=str)
    solution: list = field(default_factory=list)
    visible_tests: str = field(default_factory=str)

    plan_number: int = field(default_factory=int)
    plans: Annotated[list[list], add]

    eval_res: str = field(default_factory=str)
    all_generated_solutions: Annotated[list[list], add]
    best_solution: list = field(default_factory=list)
    good_plans: list = field(default_factory=list)


    final_raw_record: list = field(default_factory=list)

    #---
    added_tests: list = field(default_factory=list)
    number_of_added_tests: int = field(default_factory=int)
    best_solution_passes_all_visible_tests: bool = field(default_factory=bool)



    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.
