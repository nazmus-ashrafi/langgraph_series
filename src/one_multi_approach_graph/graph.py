from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.one_multi_approach_graph.configuration import AgentConfiguration
from src.one_multi_approach_graph.state import AgentState, InputState

from langchain_core.messages import AIMessage

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate

from src.one_multi_approach_graph.utils import load_chat_model, run_exec, safe_exec, generate_plan_with_timeout, format_steps
from src.one_multi_approach_graph.utils import clean_code_function, extract_ai_message_content, ends_with_assertion_error
from src.one_multi_approach_graph.utils import format_plans
from e2b.exceptions import TimeoutException
import json

import matplotlib.pyplot as plt
import seaborn as sns


from src.one_multi_approach_graph.test_generation_graph.graph import graph as test_generation_graph


# Hyper-Params ﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌

PLAN_INCREMENTATION = 1
FIRST_PLAN_NUMBER = 1
LAST_PLAN_NUMBER = 3 # Maximum number of approaches to be generated

#  ﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌﹌

async def create_added_tests(state: AgentState) -> dict[str, Any]:
    """This function takes the problem description and its existing test cases and uses them to generate new test cases

    Args:
        state (AgentState): The current state of the agent, including the problem description and its existing test cases.

    Returns:
        list[str]: A list containing the added/generated tests for the problem.

    Behavior:
        - Invokes the test_generation_graph with the problem description and its existing test cases.
        - Updates the state with the added/generated tests.
    """
    visible_tests = "\n".join(state.visible_tests_list)

    result = await test_generation_graph.ainvoke({"messages":state.messages, 
                                                  "prompt": state.prompt, 
                                                  "visible_tests": visible_tests})
    
    return {"added_tests": result["added_tests"], "visible_tests": visible_tests}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a research plan for solving a problem.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        list[str]: The current plan.
        list[list[str]]: A list of all accumulated plans till now.
        int: The current approach/plan number we are on.
    """
    
    class Plan(TypedDict):
        """Generate multi-step research plan."""
        steps: list[str]

    if state.plan_number:
        plan_number = state.plan_number
    else:
        plan_number = FIRST_PLAN_NUMBER # 1

    configuration = AgentConfiguration.from_runnable_config(config)

    dynamic_steps_prompt = configuration.dynamic_step_research_plan_system_prompt

    # Format the problem description
    formatted_problem = f"[Start Problem]\n{state.prompt}\n[End Problem]"
    # Format the existing test cases
    formatted_tests = (
        "[Start Test Suite]\n"
        f"{state.visible_tests}\n"
        + "\n".join(str(test) for test in state.added_tests) +
        "\n[End Test Suite]"
    )
    if state.plans:
        formatted_plans = (
            "[Start Other Plans]\n"
            f"{format_plans(state.plans)}\n"
            "\n[End Other Plans]"
        )
    else:
        formatted_plans= (
            "[Start Other Plans]\n"
            f"No solution plans have been generated for this problem yet.\n"
            "\n[End Other Plans]"
        )

    final_content = f"\n\n{formatted_problem}\n\n{formatted_tests}\n\n{formatted_plans}"

    messages = [
            {
                "role": "system", 
                "content": dynamic_steps_prompt
            }
        ] + [final_content]
    model = load_chat_model(configuration.model).with_structured_output(Plan)


    response = cast(Plan, await generate_plan_with_timeout(
        model=model,
        messages=messages,
        timeout_seconds=300  # 5 minutes
    ))

    return {"plan": response["steps"], 
            "plans": [response["steps"]],
            "plan_number": plan_number}


async def generate_response(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """ This function generated a solution using the plan.
    """

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.model)

    plan = format_steps(state.plan)
    print("---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ")
    print("---------------- Generation Solution for Approach Number: " + str(state.plan_number) + " --------------------------------------------------------- ")
    print("---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ")

    prompt = configuration.response_system_prompt

    # Format the problem description
    formatted_problem = f"[Start Problem Description]\n{state.prompt}\n[End Problem Description]"
    plan = format_steps(state.plan)
     # Format the existing test cases
    formatted_tests = (
        "[Start Test Suite]\n"
        f"{state.visible_tests}\n"
        + "\n".join(str(test) for test in state.added_tests) +
        "\n[End Test Suite]"
    )
    # Format the existing plan
    formatted_plan = (
        "[Start Solution Plan]\n"
        f"{plan}\n" +
        "\n[End Solution Plan]"
    )
    final_content = f"\n\n{formatted_problem}\n\n{formatted_tests}\n\n{formatted_plan}"

    messages = [{"role": "system", "content": prompt}] + [final_content]
    response = await model.ainvoke(messages)

    invalid_plan = False
    if (len(state.plan) <= 1): 
        invalid_plan = True 
    else: 
        invalid_plan = False

    solution = {

        'entry_point': state.entry_point,
        'prompt': state.prompt,
        'completion': str(clean_code_function(extract_ai_message_content(data={"messages":[response]}))),
        'approach_number': state.plan_number,
        'plan': plan, 
        'visible_tests': state.visible_tests,
        'invalid_plan': invalid_plan,
    }

    return {"solution": solution} 


# Execution
async def run_execution( # Executor Version 2
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:

    solution = state.solution

    given_tests = state.visible_tests_list
    added_tests = state.added_tests

    number_of_ori_visible_tests = len(given_tests)
    number_of_added_visible_tests = len(added_tests)
    number_of_total_visible_tests = len(given_tests) + len(added_tests)

    print("Total number of original v tests: " + str(number_of_ori_visible_tests))
    print("Total number of added v tests: " + str(len(added_tests)))
    print("Total number of v Tests: " + str(number_of_total_visible_tests))
 

    tests_and_results = []
    number_of_passed_ori_visible_tests = 0
    number_of_passed_added_visible_tests = 0

    # CODE = solution["completion"] +"\n\n"+ state.visible_tests


    # Running original visible tests
    for test in given_tests:

        # CODE = solution["completion"] +"\n\n"+ str(test)

        problem = {
                "initial_solution": solution["completion"],
                "test": str(test),
                "entry_point": state.entry_point
            }
    
        check_program = (
            clean_code_function(problem["initial_solution"]) + "\n" +
            problem["test"]
        )

        exec_globals = {}

        try:
            print("___start test___")
            safe_exec(check_program, exec_globals, timeout=10)
            print(check_program)
            print("___passed___")

            exec_score = True
            eval_result = None
            exec_feedback = f"Passed"

        except TimeoutException as te:
            # Handle timeout specifically
            exec_feedback = f"TimeoutException: {str(te)}"
            print(exec_feedback)
            exec_score = False
            eval_result = None
        except Exception as e:
            # Handle any other unexpected errors
            exec_feedback = f"Exception: {str(e)}"
            print(exec_feedback)
            exec_score = False
            eval_result = None
        

        if exec_score:
            test_score = 1
        else:
            test_score = 0

        tests_and_results.append({
            "ori_v_test": True,
            "test_score": test_score,
            ##
            "test":str(test), 
            "test_passed": exec_score,
            "error": str(exec_feedback),
            "invalid_error": ends_with_assertion_error(str(exec_feedback))
        })
        print("---- Original V.Test Passed: " + str(exec_score) + " ---------------- \n ")

    
    # Running added visible tests
    # for test in added_tests:
    for idx, test in enumerate(added_tests):

        ## Detecting the last test
        
        is_last = (idx == len(added_tests) - 1)
        if is_last:
            print("----------------  This is the last added test.  ---------------- \n ")
        ## 

        problem = {
                "initial_solution": solution["completion"],
                "test": str(test),
                "entry_point": state.entry_point
            }
    
        check_program = (
            clean_code_function(problem["initial_solution"]) + "\n" +
            problem["test"]
        )

        exec_globals = {}

        try:

            print("___start test___")
            safe_exec(check_program, exec_globals, timeout=10)
            print(check_program)
            print("___passed___")

            exec_score = True
            eval_result = None
            exec_feedback = f"Passed"


        except TimeoutException as te:
            # Handle timeout specifically
            exec_feedback = f"TimeoutException: {str(te)}"
            print(exec_feedback)
            exec_score = False
            eval_result = None
        except Exception as e:
            # Handle any other unexpected errors
            exec_feedback = f"Exception: {str(e)}"
            print(exec_feedback)
            exec_score = False
            eval_result = None

        if exec_score:
            test_score = 0.01
        else:
            test_score = 0

        tests_and_results.append({
            "ori_v_test": False,
            "test_score": test_score,
            ##
            "test":str(test), 
            "test_passed": exec_score,
            "error": str(exec_feedback),
            "invalid_error": ends_with_assertion_error(str(exec_feedback))
        })
        print("---- Added V.Test Passed: " + str(exec_score) + " ---------------- \n ")


    for test in tests_and_results:
        if test["test_passed"] == True and test["ori_v_test"] == True:
            number_of_passed_ori_visible_tests = number_of_passed_ori_visible_tests + 1

    for test in tests_and_results:
        if test["test_passed"] == True and test["ori_v_test"] == False:
            number_of_passed_added_visible_tests = number_of_passed_added_visible_tests + 1
    
    # --
    
    if number_of_ori_visible_tests == number_of_passed_ori_visible_tests:
        solution["pass"] = True
    else:
        solution["pass"] = False 

    ## ## ##

    # Incrementing the plan number
    plan_number = state.plan_number + PLAN_INCREMENTATION

    # --

    all_ori_visible_tests_passed = number_of_ori_visible_tests > 0 and number_of_passed_ori_visible_tests == number_of_ori_visible_tests
    all_added_visible_tests_passed = number_of_added_visible_tests > 0 and number_of_passed_added_visible_tests == number_of_added_visible_tests


    # --

    solution["visible_test_status"] = tests_and_results

    # Visible tests statistics for the solution
    solution["number_of_ori_visible_tests"] = number_of_ori_visible_tests
    solution["number_of_passed_ori_visible_tests"] = number_of_passed_ori_visible_tests
    solution["percentage_of_passed_ori_visible_tests"] = (
        number_of_passed_ori_visible_tests / number_of_ori_visible_tests * 100
        if number_of_ori_visible_tests > 0 else 0
    )
    solution["all_ori_visible_tests_passed"] = all_ori_visible_tests_passed
    # Added Visible tests statistics for the solution
    solution["number_of_added_visible_tests"] = number_of_added_visible_tests
    solution["number_of_passed_added_visible_tests"] = number_of_passed_added_visible_tests
    solution["percentage_of_passed_added_visible_tests"] = (
        number_of_passed_added_visible_tests / number_of_added_visible_tests * 100
        if number_of_added_visible_tests > 0 else 0
    )
    solution["all_added_visible_tests_passed"] = all_added_visible_tests_passed

    solution["selected_as_best_solution"] = False

    # --

    total_score = 0

    for test in tests_and_results:
        total_score = test["test_score"] + total_score

    solution["total_score"] = total_score

    # --

    return {"eval_res": eval_result, 
            "plan_number": plan_number,
            "solution": solution,
            "all_generated_solutions": [solution],
           }


def decide_to_regenerate(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """
    Determines whether to re-generate code

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    ### Keeps on generating after the first solution is got which solves all the visible tests
    if (state.plan_number <= LAST_PLAN_NUMBER):
        return 'create_research_plan'
    else:
        return 'create_log'

def create_log(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    

    # Picking the best solution  -----------------------------
    all_solutions_list = state.all_generated_solutions
    
    ## 
    best_solution = all_solutions_list[0]

    ### Selects the last approach, that has the best score
    for i in range(1, len(all_solutions_list)):
        if all_solutions_list[i]["total_score"] >= best_solution["total_score"]:
            best_solution = all_solutions_list[i]


    for solution in all_solutions_list:
        if solution is best_solution:
            solution["selected_as_best_solution"] = True
            break  # assuming only one match

    best_solution["selected_as_best_solution"] = True

    state.best_solution = best_solution
    best_solution_passes_all_visible_tests = best_solution["pass"]


    # Creating good plans list -----------------------------
    good_plans = []
    for solution in all_solutions_list:
        if solution["all_ori_visible_tests_passed"] == True:
            good_plans.append(solution["plan"])

    # -----------------------------------------------------

    final_raw_record = {

        'prompt': state.prompt,
        'entry_point': state.entry_point,
        ##
        'ori_visible_tests_list': state.visible_tests_list,
        'number_of_visible_tests': len(state.visible_tests_list),
        ##
        'added_visible_tests_list': state.added_tests,
        'number_of_added_visible_tests': len(state.added_tests),
        ##
        # --
        'best_solution': best_solution,
        'all_solutions_list': all_solutions_list,
        'best_solution_passes_all_visible_tests': best_solution_passes_all_visible_tests,
        # --
        'good_plans': good_plans, # plans which pass all ori visible tests
        'number_of_good_plans': len(good_plans),
    }
    

    with open(state.output_path, 'a') as f:
        f.write(json.dumps(final_raw_record) + '\n')
        f.flush()
    
    return {"logged": True, 
            "best_solution": best_solution,
            "all_generated_solutions": all_solutions_list,
            'good_plans': good_plans,
            'final_raw_record': [final_raw_record],
            'best_solution_passes_all_visible_tests': best_solution_passes_all_visible_tests,
     
            }


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(create_research_plan)
builder.add_node(create_added_tests)
builder.add_node(generate_response)
builder.add_node(run_execution)
builder.add_node(create_log)


builder.add_edge(START, "create_added_tests")
builder.add_edge("create_added_tests", "create_research_plan")
builder.add_edge("create_research_plan", "generate_response")
builder.add_edge("generate_response", "run_execution")
builder.add_conditional_edges(
    "run_execution",
    decide_to_regenerate,
    ["create_research_plan", "create_log"])
builder.add_edge("create_log", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "one_multi_approach_graph"

