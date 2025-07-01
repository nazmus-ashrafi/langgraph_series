from typing import Any, Literal, TypedDict, cast
from datetime import datetime
import json

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

# Importing Utils
from src.two_search_reflect_graph.utils import load_chat_model, format_steps, clean_code_function, extract_ai_message_content

# Graph Specific Imports ------------------
from src.two_search_reflect_graph.configuration import AgentConfiguration
from src.two_search_reflect_graph.state import AgentState, InputState
# ------------------ ------------------

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from openevals.llm import create_async_llm_as_judge
from openevals.prompts import (
    RAG_HELPFULNESS_PROMPT,
)

current_date = datetime.now().strftime("%A, %B %d, %Y")

judge_model = init_chat_model(
    model = "gpt-4o-mini-2024-07-18", 
    model_provider="openai",
    temperature=0.2
)

MAX_SEARCH_RETRIES = 3

helpfulness_evaluator = create_async_llm_as_judge(
    judge=judge_model,
    prompt=RAG_HELPFULNESS_PROMPT
    + f'\nReturn "true" if the answer is helpful, and "false" otherwise.\n\nThe current date is {current_date}.',
    feedback_key="helpfulness",
)

# --

async def store_original_question(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    
    messages = state.messages
    programming_problem = messages[-1]

    return {"original_question": programming_problem, 
            "attempted_search_queries": []}

async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for solving a problem.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate multi-step research plan."""

        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    messages = [
            {"role": "system", 
             "content": configuration.research_plan_system_prompt

             }
        ] + state.messages
    model = load_chat_model(configuration.model).with_structured_output(Plan)

    response = cast(Plan, await model.ainvoke(messages))

    return {"steps": response["steps"], "plan": response["steps"] }

async def call_model(
        state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    
    # Simplify the Tavily search tool's input schema for a small local model
    @tool
    async def search_tool(query: str):
        """Search the web for information relevant to the query."""
        return await TavilySearch(max_results=10).ainvoke({"query": query})

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)
    model_with_tools = model.bind_tools([search_tool])

    messages = [{"role": "system", "content": configuration.web_search_tool_system_prompt}] + state.messages
    response = await model_with_tools.ainvoke(messages)

    # If there is a tool called by the model
    if response.tool_calls and response.tool_calls[0]["name"] == search_tool.name:
        search_query = response.tool_calls[0]["args"]["query"]
        return {
            "messages": [response],
            "attempted_search_queries": state.attempted_search_queries
            + [search_query],
        }
    
    return {"messages": [response]}

async def should_continue(state: AgentState):
    if len(state.attempted_search_queries) > MAX_SEARCH_RETRIES:
        return "generate_response"
    messages = state.messages
    last_message = messages[-1]
    if last_message.tool_calls:
        return "web_search"
    return "reflect"

async def web_search(state: AgentState):

    # Simplify the Tavily search tool's input schema for a small local model
    @tool
    async def search_tool(query: str):
        """Search the web for information relevant to the query."""
        return await TavilySearch(max_results=10).ainvoke({"query": query})
    
    messages = state.messages
    last_message = messages[-1]
    
    search_results = await search_tool.ainvoke(last_message.tool_calls[0])
    return {"messages": [search_results]}


async def reflect(state: AgentState):
    messages = state.messages
    last_message = messages[-1]

    helpfulness_eval_result = await helpfulness_evaluator(
        inputs=state.original_question, outputs=last_message.content
    )
    
    if not helpfulness_eval_result["score"]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
I originally asked you the following question:

<original_question>
    Search the web for similar competitive programming problems and solutions to help solve this one: {state.original_question}
</original_question>

Your answer was not helpful for the following reason:

<reason>
{helpfulness_eval_result['comment']}
</reason>

Please check the conversation history carefully and try again. You may choose to fetch more information if you think the answer
to the original question is not somewhere in the conversation, but carefully consider if the answer is already in the conversation.

You have already attempted to answer the original question using the following search queries,
so if you choose to search again, you must rephrase your search query to be different from the ones below to avoid fetching redundant information:

<attempted_search_queries>
{state.attempted_search_queries}
</attempted_search_queries>

As a reminder, check the previous conversation history and fetched context carefully before searching again!
""",
                }
            ]
        }

    else:
        return {"relavant_search_result": str(last_message.content)}





async def generate_response(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    plan = format_steps(state.plan)

    prompt = configuration.response_system_prompt.format(
        plan=plan,
        relavant_search_result = state.relavant_search_result
    )

    messages = [{"role": "system", "content": prompt}] + [state.messages[0]]
    response = await model.ainvoke(messages)

    solution = {
        'completion': str(clean_code_function(extract_ai_message_content(data={"messages":[response]}))),
    }

    return {"messages": [response], "solution": solution}


def create_log(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    
    solution = state.solution
    with open("output.jsonl", 'a') as f:
        f.write(json.dumps(solution) + '\n')
        f.flush()
    
    return {"logged": True}


async def retry_or_end(state: AgentState):
    messages = state.messages
    last_message = messages[-1]

    if last_message.type == "human":
        return "agent"
    return "generate_response"



# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)

builder.add_node(store_original_question)
builder.add_node(create_research_plan)
builder.add_node(generate_response)
builder.add_node(create_log)
builder.add_node("agent", call_model)
builder.add_node("web_search", web_search)
builder.add_node("reflect", reflect)


builder.add_edge(START, "create_research_plan")
builder.add_edge("create_research_plan", "store_original_question")
builder.add_edge("store_original_question", "agent")

builder.add_conditional_edges("agent", should_continue, ["web_search", "generate_response", "reflect"])
builder.add_edge("web_search", "agent")

builder.add_conditional_edges(
    "reflect",
    retry_or_end,
    ["agent", "generate_response"],
)


builder.add_edge("generate_response", "create_log")
builder.add_edge("create_log", END)


# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "SearchAndReflectCodeGenGraph"