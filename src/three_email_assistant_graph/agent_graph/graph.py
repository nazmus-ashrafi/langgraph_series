from typing import Any, Literal

from langgraph.graph import START, END
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

from src.three_email_assistant_graph.configuration import AgentConfiguration
from src.three_email_assistant_graph.state import State
from src.three_email_assistant_graph.utils import load_chat_model

# Nodes
async def llm_call(state: State, *, config: RunnableConfig) -> dict[str, Any]:
    """LLM decides whether to call a tool or not"""

    configuration = AgentConfiguration.from_runnable_config(config)

    # Collect all tools
    tools = [configuration.write_email,     
            configuration.Done] 

    # Initialize the LLM, enforcing tool use
    llm = load_chat_model(configuration.model)
    llm_with_tools = llm.bind_tools(tools, tool_choice="any")

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": configuration.agent_system_prompt.format(
                        tools_prompt=configuration.agent_tool_prompt,
                        background=configuration.default_background,
                        response_preferences=configuration.default_response_preferences)
                    },
                    
                ]
                + state["messages"]
            )
        ]
    }

# After the LLM makes a decision, we need to execute the chosen tool. 
# The `tool_handler` node executes the tool. 
# We can see that nodes can update the graph state to capture any important state changes, such as the classification decision.
async def tool_handler(state: State, *, config: RunnableConfig) -> dict[str, Any]:
    """Performs the tool call"""

    configuration = AgentConfiguration.from_runnable_config(config)

    # Collect all tools
    tools = [configuration.write_email,    
            configuration.Done]
    tools_by_name = {tool.name: tool for tool in tools}

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content" : observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

# Our agent needs to decide when to continue using tools and when to stop. 
# This conditional routing function directs the agent to either continue or terminate.
# Conditional edge function
def should_continue(state: State) -> Literal["tool_handler", "__end__"]:
    """Route to tool handler, or end if Done tool called."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls: 
            if tool_call["name"] == "Done":
                return END
            else:
                return "tool_handler"

# Build workflow
agent_builder = StateGraph(State)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_handler", tool_handler)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "tool_handler": "tool_handler",
        END: END,
    },
)
agent_builder.add_edge("tool_handler", "llm_call")

# Compile the agent
agent = agent_builder.compile()