from typing import Any, Literal, TypedDict, cast
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.three_email_assistant_graph.configuration import AgentConfiguration
from src.three_email_assistant_graph.state import State, StateInput
from src.three_email_assistant_graph.utils import load_chat_model, parse_email, format_email_markdown
from src.three_email_assistant_graph.agent_graph.graph import agent


async def triage_router(state: State, *, config: RunnableConfig
) -> Command[Literal["response_agent", "__end__"]]:
    """Analyze email content to decide if we should respond, notify, or ignore."""

    # Importing the configuration from the configuration file
    # The configuration contains goodies like model, tools and prompts
    configuration = AgentConfiguration.from_runnable_config(config)

    class RouterSchema(BaseModel):
        """Analyze the unread email and route it according to its content."""

        reasoning: str = Field(
            description="Step-by-step reasoning behind the classification."
        )
        classification: Literal["ignore", "respond", "notify"] = Field(
            description="The classification of an email: 'ignore' for irrelevant emails, "
            "'notify' for important information that doesn't need a response, "
            "'respond' for emails that need a reply",
        )

    # Initialize the LLM for use with router / structured output
    llm = load_chat_model(configuration.model)
    llm_router = llm.with_structured_output(RouterSchema) 
    
    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = configuration.triage_system_prompt.format(
        background=configuration.default_background,
        triage_instructions=configuration.default_triage_instructions
    )
    user_prompt = configuration.triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email: \n\n{format_email_markdown(subject, author, to, email_thread)}",
                }
            ],
            "classification_decision": result.classification,
        }
        
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        goto = END
        update =  {
            "classification_decision": result.classification,
        }
        
    elif result.classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")
        # For now, we go to END. But we will add to this later!
        goto = END
        update = {
            "classification_decision": result.classification,
        }
        
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)


overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
)

graph = overall_workflow.compile()