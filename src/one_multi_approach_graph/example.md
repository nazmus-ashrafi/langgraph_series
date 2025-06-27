
# Example 1 _____________________________________________________________________________________

# Prompt/Message

from typing import List   def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True """

# Entry Point

has_close_elements

# Visible Tests List

["assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True", "assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False"]

# Output Path

output.jsonl


# Example 2 _____________________________________________________________________________________

# Prompt/Message

from typing import List   def separate_paren_groups(paren_string: str) -> List[str]: """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to separate those group into separate strings and return the list of those. Separate groups are balanced (each open brace is properly closed) and not nested within each other Ignore any spaces in the input string. >>> separate_paren_groups('( ) (( )) (( )( ))') ['()', '(())', '(()())'] """

# Entry Point

separate_paren_groups

# Visible Tests List

["assert separate_paren_groups('(()()) ((())) () ((())()())') == [ '(()())', '((()))', '()', '((())()())' ]",
"assert separate_paren_groups('() (()) ((())) (((())))') == [ '()', '(())', '((()))', '(((())))' ]",
"assert separate_paren_groups('(()(())((())))') == [ '(()(())((())))' ]",
"assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']"]

# Output Path

output.jsonl
