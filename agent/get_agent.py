import os
import json
import logging
from typing import List, Tuple, Any, Union

from pydantic import Field
from langchain import LLMChain
from langchain.agents import ZeroShotAgent
from langchain.schema import AgentAction, AgentFinish

from .tools import Tool, GetDetailsTool, tool_projection
from .custom_parser import CustomMRKLOutputParser
from .custom_agent_executor import CustomAgentExecutor
from utils import load_openapi_spec, escape
from .agent_prompts import train_prompt_v2, test_prompt_v1
from .custom_agent import CustomZeroShotAgent


logger = logging.getLogger(__name__)


def get_agent(
    llm,
    api_data,
    server_url,
    agent_prompt=train_prompt_v2,
    enable_getDetails=True,
    return_intermediate_steps=True,
):
        
    openapi_spec = load_openapi_spec(api_data["Documentation"], replace_refs=True)
    components_descriptions = escape(api_data["Function_Description"]["components"])

    tools = [GetDetailsTool()] if not enable_getDetails else []
    for ext_tool in api_data.get("external_tools", []):
        tools.append(tool_projection[ext_tool]())

    for idx, func_name in enumerate(api_data["Function_Projection"]):
        description = escape(api_data["Function_Description"][func_name])
        if idx == len(api_data["Function_Projection"]) - 1:
            description += components_descriptions
        path, method = api_data["Function_Projection"][func_name]
        tools.append(Tool(
            base_url=server_url + "/" + api_data["Name"] if server_url else None,
            func_name=func_name,
            openapi_spec=openapi_spec,
            path=path,
            method=method,
            description=description,
            retrieval_available="retrieval" in api_data.get("external_tools", [])
        ))

    AgentType = CustomZeroShotAgent if agent_prompt == test_prompt_v1 else ZeroShotAgent

    prompt = AgentType.create_prompt(
        tools, 
        prefix=agent_prompt["prefix"], 
        suffix=agent_prompt["suffix"],
        format_instructions=agent_prompt["format_instructions"],
        input_variables=["input", "agent_scratchpad"]
    )

    logger.info(str(prompt))

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    AgentType.return_values = ["output", "Final Thought"]
    agent = AgentType(llm_chain=llm_chain, allowed_tools=[t.name for t in tools])
    if agent_prompt != test_prompt_v1:
        agent.output_parser = CustomMRKLOutputParser()
    
    agent_executor = CustomAgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=return_intermediate_steps
    )
    return agent_executor

