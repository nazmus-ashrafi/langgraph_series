"""State management for the test generation graph.

This module defines the state structures used in the test generation graph.
"""

from dataclasses import dataclass, field
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# @dataclass(kw_only=True)
# class TestGenPrivateState:
#     """Private state for the test generation graph."""

#     messages: Annotated[list[AnyMessage], add_messages]


@dataclass(kw_only=True)
class TestGenState:
    """State of the test generation graph / agent."""

    # Inherited from parent graph
    messages: Annotated[list[AnyMessage], add_messages]
    prompt: str = field(default_factory=str)
    visible_tests: str = field(default_factory=str)
    # Produced
    added_tests: list = field(default_factory=list)

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.