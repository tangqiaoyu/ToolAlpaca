prefix = """Anser the user's question with the help of tools. The user cannot see the tool usage or use the tool themselves, you can use the tools. And the user cannot see the process of your tool use, so you must give all the infomation in Final Answer field to user. Your task is to answer the user's question, so DO NOT make up anything. If your required parameters are missing use tool "getDetails" to ask user provide them.
You have access to the following tools:"""
suffix = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

format_instructions = """Use the following format:
Question: the input question you must answer
Thought: Answer the following three questions with one paragraph: 1) Check whether there are any general terms or pronouns that lack sufficient context or specific information. 2) Consider the question and potential approach to answer it. 3) Explain your reasoning and the steps needed to reach a solution. 
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action, should be json format. All of the action input must be realistic and from the user. Never generate any action input by yourself or copy the input description.
Observation: the result of the action and how it contributes to the solution
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Summarize the information gathered and the reasoning behind your final answer.
Final Answer: Provide a user-friendly and detailed answer to the original input question that summarizes all relevant information from the Thought/Action/Action Input/Observation sequences, without mentioning specific tool usage details or technical jargon. Ensure the answer is both informative and appropriate for the user."""

prefix_v2 = """Your task is to answer the user's question using available tools. The user cannot see or use the tools themselves, nor can they see the process of your tool usage. Provide all necessary information in the "Final Answer" field. Do not make up any information. If required parameters are missing, use the "getDetails" tool to ask the user for them.
You have access to the following tools:"""

format_instructions_v2 = """Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action, must be JSON format. All of the action input must be realistic and from the user.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Summarize the information gathered and the reasoning behind your final answer.
Final Answer: Provide a user-friendly and detailed answer to the original input question that summarizes all relevant information from the Thought/Action/Action Input/Observation sequences."""

train_prompt_v1 = {
    "prefix": prefix,
    "suffix": suffix,
    "format_instructions": format_instructions
}

train_prompt_v2 = {
    "prefix": prefix_v2,
    "suffix": suffix,
    "format_instructions": format_instructions_v2
}


prefix_test = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions with the help of some tools.
You have access to the following tools:"""

suffix_test = """Begin!

USER: {input}
ASSISTANT Thought:{agent_scratchpad}"""

format_instructions_test = """The chat follows this format:
USER: the user's question
ASSISTANT Thought: the assistant's inner thought about what to do next 
ASSISTANT Action: the action to take, must be one of [{tool_names}].
ASSISTANT Action Input: the input for the action, in JSON format.
ASSISTANT Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
ASSISTANT Thought: summarize the information gathered
ASSISTANT Response: the final response to the user
USER: user's next question
..."""
test_prompt_v1 = {
    "prefix": prefix_test,
    "suffix": suffix_test,
    "format_instructions": format_instructions_test
}

prefix_test_v2 = """Answer the following questions as best you can. You have access to the following tools:"""

suffix_test = """Begin!

USER: {input}
ASSISTANT Thought:{agent_scratchpad}"""

format_instructions_test_v2 = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, in JSON format.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
test_prompt_v2 = {
    "prefix": prefix_test_v2,
    "suffix": suffix,
    "format_instructions": format_instructions_test_v2
}

prompt_proj = {
    "train_v1": train_prompt_v1,
    "train_v2": train_prompt_v2,
    "test_v1": test_prompt_v1,
    "test_v2": test_prompt_v2
}