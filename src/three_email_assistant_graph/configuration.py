"""Define the configurable parameters for the graph."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar
from datetime import datetime
from typing import Annotated
from pydantic import BaseModel

from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import tool


from src.three_email_assistant_graph import prompts


@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for defining from_runnable_config

    """

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)



@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    ## Model
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default="anthropic/claude-3-haiku-20240307",
        default="openai/gpt-4o-mini-2024-07-18", # 🌀
        # default="openai/gpt-4o", # same as default="openai/gpt-4o-2024-08-06", # 🌀
        # Model temperature can be changed in the utils.py file (look for the load_chat_model function)

        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },)
    
    ## Tools
    @tool
    def write_email(to: str, subject: str, content: str) -> str:
        """Write and send an email."""
        # Placeholder response - in real app would send email
        return f"Email sent to {to} with subject '{subject}' and content: {content}"

    @tool
    class Done(BaseModel):
        """E-mail has been sent."""
        done: bool
    
    ## Prompts
    triage_system_prompt: str = field(
        default=prompts.triage_system_prompt,
        metadata={
            "description": "The prompt used to describe the role of the triage agent."
        },
    )

    triage_user_prompt: str = field(
        default=prompts.triage_user_prompt,
        metadata={
            "description": "The prompt used to describe the task of the triage agent."
        },
    )

    triage_user_prompt: str = field(
        default=prompts.triage_user_prompt,
        metadata={
            "description": "The prompt used to describe the task of the triage agent."
        },
    )

    agent_tool_prompt: str = field(
        default=prompts.AGENT_TOOLS_PROMPT,
        metadata={
            "description": "The prompt used to list the tools agent can use."
        },
    )

    agent_system_prompt: str = field(
        default=prompts.agent_system_prompt,
        metadata={
            "description": "The prompt used to describe the task of the email assistant agent."
        },
    )


    ## Additional context information
    # Default background information 
    default_background = """ 
    I'm Nazmus, a graduate student at UAEU.
    """

    # Default triage instructions 
    default_triage_instructions = """
    Emails that are not worth responding to:
    - Marketing newsletters and promotional emails
    - Spam or suspicious emails

    There are also other things that should be known about, but don't require an email response. For these, you should notify (using the `notify` response). Examples of this include:
    - Blackboard notifications
    - Research Gate, Elsevier researh paper
    - Build system notifications or deployments
    - UAEU Automatic notifications
    - Important company announcements
    - FYI emails that contain relevant information for current events
    - HR Department deadline reminders
    - Subscription status / renewal reminders
    - Helpdesk password expiry
    - Course material, grade updates

    Emails that are worth responding to:
    - Direct questions from professors, students
    - Meeting requests requiring confirmation
    - Event or conference invitations
    - collaboration or project-related requests
    """

    # Default response preferences 
    default_response_preferences = """
    Use professional and concise language. If the e-mail mentions a deadline, make sure to explicitly acknowledge and reference the deadline in your response.

    When responding to direct questions from professors, students:
    - Clearly state whether you will investigate or who you will ask
    - Provide an estimated timeline for when you'll be able to respond

    When responding to event or conference invitations:
    - Always acknowledge any mentioned deadlines (particularly registration deadlines)
    - If workshops or specific topics are mentioned, ask for more specific details about them
    - If discounts (group or early bird) are mentioned, explicitly request information about them
    - Don't commit 

    When responding to collaboration or project-related requests:
    - Acknowledge any existing work or materials mentioned (drafts, slides, documents, etc.)
    - Explicitly mention reviewing these materials before or during the meeting
    - When scheduling meetings, clearly state the specific day, date, and time proposed

    """