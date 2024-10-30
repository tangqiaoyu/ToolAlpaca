import os
import json
import time
import logging
import datetime
from string import Template
from typing import Any, Awaitable, Callable, Optional, Union, List, Tuple

import requests
from langchain.tools.base import BaseTool

from .convert_request import call_api_function
from utils import openai_chat_completions, parse_json_string


logger = logging.getLogger(__name__)


class Tool(BaseTool):
    description: str = ""
    func: Callable[[str], str]
    coroutine: Optional[Callable[[str], Awaitable[str]]] = None
    max_output_len = 3000

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return self.func(tool_input)

    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""
        if self.coroutine:
            return await self.coroutine(tool_input)
        raise NotImplementedError("Tool does not support async")

    def __init__(self, base_url, func_name, openapi_spec, path, method, description, retrieval_available=False, **kwargs):
        """ Store the function, description, and tool_name in a class to store the information
        """    
        def func(params):
            try:
                params = parse_json_string(params)
                if isinstance(params, list):
                    return "'Action Input' cannot be a list. Only call one function per action."
            except:
                return "Invalid JSON format. Please ensure 'Action Input' is a valid JSON object."
            retry_times = 0
            while retry_times < 3:
                try:
                    response = call_api_function(
                        input_params=params,
                        openapi_spec=openapi_spec,
                        path=path,
                        method=method,
                        base_url=base_url
                    )
                # except Exception as e:
                except ValueError as e:
                    # if isinstance(e, KeyError):
                    #     return str(e)
                    return str(e)
                message = f"Status Code: {response.status_code}. Response: {response.text}"
                if response.status_code >= 500:
                    retry_times += 1
                    continue
                break
            if not 200 <= response.status_code < 300:
                # message += ". You can try to change the input or call another function. "
                message += ". You should choose one of: (1) change the input and retry; (2) return the 'Final Answer' and explain what happened; (You must choose this one when the error occurs more than 3 times.) (3) call another function."

            if len(message) > self.max_output_len:
                if retrieval_available:
                    file_name = f"./tmp/retrieval_{int(time.time())}.txt"
                    with open(file_name, "w", encoding="utf-8") as f:
                        f.write(message)
                    return "The output is too long. You need to use the 'retrievalDataFromFile' function to retrieve the output from the file: " + file_name
                else:
                    message = message[:self.max_output_len]
            return message
        
        tool_name = func_name

        super(Tool, self).__init__(
            name=tool_name, func=func, description=description, **kwargs
        )


class CustomInvalidTool(BaseTool):
    """Tool that is run when invalid tool name is encountered by agent."""

    name = "invalid_tool"
    description = "Called when tool name is invalid."

    def _run(self, tool_name: str, all_tools: List[str]) -> str:
        """Use the tool."""
        return f"`{tool_name}` is not a valid action. The action must be one of {all_tools}."

    async def _arun(self, tool_name: str, all_tools: List[str]) -> str:
        """Use the tool asynchronously."""
        return f"`{tool_name}` is not a valid action. The action must be one of {all_tools}."
    def run(
        self,
        tool_input: str,
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any
    ) -> str:
        """Run the tool."""
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        self.callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input,
            verbose=verbose_,
            color=start_color,
            **kwargs,
        )
        try:
            observation = self._run(tool_input, all_tools=kwargs["all_tools"])
        except (Exception, KeyboardInterrupt) as e:
            self.callback_manager.on_tool_error(e, verbose=verbose_)
            raise e
        self.callback_manager.on_tool_end(
            observation, verbose=verbose_, color=color, name=self.name, **kwargs
        )
        return observation


class DateTimeTool(Tool):
    def __init__(self, **kwargs):
        def func(params):
            current_time = datetime.datetime.now()
            return current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        description = """Retrieve the current date and time.
Parameters: {{}}
Output: The current date and time in a formatted string, providing a real-time reference for various applications and data.
 - Format: text/json
 - Structure: {{"timestamp": "String representing the current date and time in the format \"YYYY-MM-DD HH:MM:SS\"."}}"""

        super(Tool, self).__init__(
            name="getCurrentDateTime", func=func, description=description, **kwargs
        )


class RetrievalTool(Tool):
    def __init__(self, **kwargs):
        def func(params):
            try:
                params = parse_json_string(params)
                file_path = params["file_path"]
                query = params["query"]
            except:
                return "Invalid JSON format. Please ensure 'Action Input' is a valid JSON object."
            
            if not os.path.exists(file_path):
                return "File does not exist."
            
            file_content = open(file_path, "r", encoding="utf-8").read()
            template = Template(open("prompts/retrieval.txt", "r").read())
            prompt = template.substitute(output=file_content[:40000], query=query)
            completion = openai_chat_completions([{"role": "user", "content": prompt}], model="gpt-3.5-turbo-16k-0613")
            outputs = completion["choices"][0]["message"]["content"]

            return {"retrieved_info": outputs}
        
        description = """Retrieve specified information from a file. This tool can ONLY be used when you receive the message 'The output is too long. You need to use the 'retrievalDataFromFile' function to retrieve the output from the file: <file_path>'.
Parameters: {{"file_path": "Required. String. The path to the file from which information needs to be retrieved.", "query": "Required. String. The specific information to be retrieved from the file."}}
Output: The retrieved information from the file.
 - Format: text/json
 - Structure: {{"retrieved_info": "String containing the requested information retrieved from the file."}}"""

        super(Tool, self).__init__(
            name="retrievalDataFromFile", func=func, description=description, **kwargs
        )


def get_details(chat_history):
    prefix_file_path = "prompts/get_details.txt"
    prefix = open(prefix_file_path, "r").read()
    chat_history[0] = chat_history[0][0], chat_history[0][1].strip()
    prompt = prefix + "\n" + "\n".join([f"[{x[0]}]: {x[1]}" for x in chat_history])
    completion = openai_chat_completions([{"role": "user", "content": prompt}])
    details = completion["choices"][0]["message"]["content"]
    if details.strip().startswith("[User]:"):
        details = details.strip().split("[User]:", 1)[1].strip()
    details = details.strip().split("[AI]:", 1)[0].strip()
    logger.debug("get_details")
    logger.debug(chat_history)
    logger.debug(details)
    return details


class GetDetailsTool(BaseTool):
    chat_history: List[Tuple] = []
    func: Callable[[str], str] = None
    coroutine: Optional[Callable[[str], Awaitable[str]]] = None

    def _run(self, tool_input: str, **kwargs) -> str:
        """Use the tool."""
        if kwargs.get("inputs", None):
            input = kwargs["inputs"].get("input", None)
            if input is not None:
                if len(self.chat_history) == 0:
                    self.chat_history = [("User", input)]
                elif input != self.chat_history[0][1]:
                    self.chat_history = [("User", input)]
            else:
                return "Code Error! Input is None."
        return self.func(tool_input)

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError("Tool does not support async")

    def __init__(self, **kwargs):
        def func(params):
            try:
                question = parse_json_string(params)
            except:
                return "Invalid JSON format. Please ensure 'Action Input' is a valid JSON object."
            if "Question" not in question:
                return 'For `getDetails`, the "Action Input" must be like {Question": "The question to prompt user to provide sufficient information."}.'
            self.chat_history.append(("AI", question))
            details = get_details(self.chat_history)
            self.chat_history.append(("User", details))
            return details
        description = """If the user's question lacks the essential information needed to answer the question effectively, or if the question contains vague terms or pronouns without sufficient context, invoke the `getDetails` function to prompt the user for the missing critical details. However, `getDetails` should not be used in cases where the user omits optional parameters, unless these parameters become necessary in the course of the conversation. 
Parameters: {{"Question": "The question to prompt user to provide sufficient information."}}
Output: User's response."""
        super(GetDetailsTool, self).__init__(
            name="getDetails",
            func=func,
            description=description,
            **kwargs
        )

    def run(
        self,
        tool_input: str,
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        **kwargs: Any
    ) -> str:
        """Run the tool."""
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose
        self.callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input,
            verbose=verbose_,
            color=start_color,
            **kwargs,
        )
        try:
            observation = self._run(tool_input, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            self.callback_manager.on_tool_error(e, verbose=verbose_)
            raise e
        self.callback_manager.on_tool_end(
            observation, verbose=verbose_, color=color, name=self.name, **kwargs
        )
        return observation


tool_projection = {
    "retrieval": RetrievalTool,
    "datetime": DateTimeTool,
    "getDetails": GetDetailsTool
}
