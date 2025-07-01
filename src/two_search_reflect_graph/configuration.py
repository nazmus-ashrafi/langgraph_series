"""Define the configurable parameters for the graph."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from dataclasses import dataclass, field
from typing import Annotated

from src.two_search_reflect_graph import prompts


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
    
    ## Plan Generation Graph Prompts
    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={
            "description": "The prompt used for generating the solution plan for the problem."
        },
    )

     ## Plan Generation Graph Prompts
    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={
            "description": "The prompt used for generating the solution plan for the problem."
        },
    )

    web_search_tool_system_prompt: str = field(
        default=prompts.WEB_SEARCH_TOOL_SYSTEM_PROMPT,
        metadata={
            "description": "The prompt for the agent to use the web search tool to find a similar competitive programming problem."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for generating a solution."
        },
    )



