from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.one_multi_approach_graph.configuration import AgentConfiguration
# from src.plan_generation_graph.state import AgentState, InputState

from src.one_multi_approach_graph.test_generation_graph.state import TestGenState


# Importing Utils
from src.one_multi_approach_graph.utils import load_chat_model
import json

from langchain_core.messages import AIMessage
import asyncio



async def problem_input_identifier(
    state: TestGenState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """
    """

    configuration = AgentConfiguration.from_runnable_config(config)
    system_prompt = configuration.problem_input_identifier_system_prompt

    # Format the problem description
    formatted_problem = f"<problem_description>\n{state.prompt}\n</problem_description>"
    # Format the existing test cases
    formatted_tests = f"<existing_test_cases>\n{state.visible_tests}\n</existing_test_cases>"
    final_content = f"\n\n{formatted_problem}\n\n{formatted_tests}" 

    messages = [{"role": "system", "content": system_prompt}] + [final_content]
    model = load_chat_model(configuration.model)

    try:
        # Set timeout to 120 seconds (2 minute)
        response = await asyncio.wait_for(
            model.ainvoke(messages), 
            timeout=320.0
        )
       
        return {"messages": [response]}
    
    except asyncio.TimeoutError as e:
        # Return empty if timeout occurs
        print(f"Timeout Error in problem_reflection: {e}")
        return {"messages": []}
    
    except Exception as e:
        # Handle any other exceptions and return empty list
        print(f"Error in problem_reflection: {e}")
        return {"messages": []}
    
    

async def input_based_test_case_generator(
    state: TestGenState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """
    """

    class Added_VTests(TypedDict):
        """Additional tests list."""
        added_tests: list[str]
    
    configuration = AgentConfiguration.from_runnable_config(config)
    system_prompt = configuration.isp_based_test_case_generator_system_prompt

    messages = [{"role": "system", "content": system_prompt}] + state.messages
    model = load_chat_model(configuration.model).with_structured_output(Added_VTests)

    try:
        # Set timeout to 120 seconds (2 minute)
        response = await asyncio.wait_for(
            model.ainvoke(messages), 
            timeout=320.0
        )
        response = cast(Added_VTests, response)

        return {"added_tests": response["added_tests"], "messages": []}
    
    except asyncio.TimeoutError:
        # Return empty if timeout occurs
        print(f"Error in problem_input_domain_characterization")
        return {"added_tests": [], "messages": []}
    
    except Exception as e:
        # Handle any other exceptions and return empty list
        print(f"Error in problem_input_domain_characterization: {e}")
        return {"added_tests": [], "messages": []}



# Define the graph
builder = StateGraph(TestGenState, input=TestGenState, config_schema=AgentConfiguration)
builder.add_node(problem_input_identifier)
builder.add_node(input_based_test_case_generator)


# builder.add_edge(START, "added_tests_generator")

builder.add_edge(START, "problem_input_identifier")
builder.add_edge("problem_input_identifier", "input_based_test_case_generator")
builder.add_edge("input_based_test_case_generator", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "Test_Gen_Graph"

