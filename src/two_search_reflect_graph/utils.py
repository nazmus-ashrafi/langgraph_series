"""Utility functions used in the project.

"""

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from typing import List, Dict, Any
import re


import multiprocessing
import asyncio
from langchain.schema import AIMessage, HumanMessage

def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)

    else:
        provider = ""
        model = fully_specified_name

    return init_chat_model(model, model_provider=provider, temperature=0) # Hardcoded Temperature


# Safe Execute Utilities ####﹌﹌﹌﹌####﹌﹌﹌﹌####﹌﹌﹌﹌####

def run_exec(check_program, exec_globals, result_queue):
    try:
        exec(check_program, exec_globals)
        result_queue.put(None)  # Success
    except Exception as e:
        result_queue.put(e)

def safe_exec(check_program, exec_globals, timeout=2):
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_exec, args=(check_program, exec_globals, result_queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        raise TimeoutError("Execution timed out")

    result = result_queue.get()
    if result is not None:
        raise result  # Re-raise any exception from exec

####﹌﹌﹌﹌####﹌﹌﹌﹌####﹌﹌﹌﹌####﹌﹌﹌﹌####﹌﹌﹌﹌####﹌﹌﹌﹌####

# Utils
# Default fallback response
DEFAULT_PLAN = {"steps": ["There is no plan. Generate a solution for the following task."]}

# Utils
async def generate_plan_with_timeout(model, messages, timeout_seconds=120):
    """
    Generate a plan with timeout and error handling.
    
    Args:
        model: The chat model with structured output
        messages: Messages for the model
        config_with_callbacks: Configuration with callbacks
        timeout_seconds: Timeout in seconds (default: 120 = 2 minutes)
    
    Returns:
        Plan: Either the generated plan or fallback plan
    """
    try:
        # Use asyncio.wait_for to implement timeout
        response = await asyncio.wait_for(
            model.ainvoke(messages),
            timeout=timeout_seconds
        )
        return response
        
    except asyncio.TimeoutError:
        print("taking too long to generate this plan")
        return DEFAULT_PLAN
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return DEFAULT_PLAN
    

# Utility
def ends_with_assertion_error(s: str) -> bool:
    return s.rstrip().endswith("AssertionError")


def clean_code_function(original_code: str) -> str:
    # Use regex to remove the triple quotes and ```python\n from the string
    clean_code = re.sub(r'```python\n|```|"""', '', original_code)
    return clean_code.strip()  # Strip any leading/trailing whitespace

def extract_ai_message_content(data: Dict[str, Any]) -> str:
    """Extracts the content from AIMessage in the given dictionary."""
    for message in data.get("messages", []):
        if isinstance(message, AIMessage):
            return message.content
    return ""

def extract_human_message_content(data: Dict[str, Any]) -> str:
    """Extracts the content from HumanMessage in the given dictionary."""
    for message in data.get("messages", []):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def format_steps(steps: List[str]) -> str:
    """
    Formats a list of steps into a single text output.
    
    Args:
        steps (List[str]): List of step descriptions.
    
    Returns:
        str: Steps as a single text output.
    """
    # return "\n".join(steps)
    return"\n".join(f"** {step}" for i, step in enumerate(steps))

def format_tasks_list_with_numbers(steps: List[str]) -> str:
    """
    Formats a list of steps into a numbered list with 'Step X:' prefixes.
    
    Args:
        steps (List[str]): List of step descriptions.
    
    Returns:
        str: Formatted steps as a single text output.
    """
    return "\n".join(f"Task {i + 1}: {step}" for i, step in enumerate(steps))


def format_plans(plans):
    result = []
    for i, plan in enumerate(plans, start=1):
        result.append(f"# Plan {i}")
        for step in plan:
            result.append(f"- {step}")
        result.append("")  # Add a blank line between plans
    return "\n".join(result)