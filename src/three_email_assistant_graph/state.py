"""State management for the email assistant
"""

from langgraph.graph import MessagesState
from typing import Literal
from typing_extensions import TypedDict

class StateInput(TypedDict):
    # This is the input to the state
    email_input: dict


class State(MessagesState):
    # We can add a specific key to our state for the email input
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]