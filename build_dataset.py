import os
import json
import argparse

from agent.tools import Tool, GetDetailsTool
from agent.agent_prompts import test_prompt_v1
from agent.custom_agent import CustomZeroShotAgent
from utils import load_openapi_spec, escape
SERVER_URL="http://localhost:5678"

def rreplace(s, old, new, occurrence=1):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def build_dataset(api_info):
    openapi_spec = load_openapi_spec(api_info["Documentation"])
    components_descriptions = escape(api_info["Function_Description"]["components"])
    tools = [GetDetailsTool()]
    for idx, func_name in enumerate(api_info["Function_Projection"]):
        description = escape(api_info["Function_Description"][func_name])
        if idx == len(api_info["Function_Projection"]) - 1:
            description += components_descriptions
        path, method = api_info["Function_Projection"][func_name]
        tools.append(Tool(
            base_url=SERVER_URL + api_info["Name"],
            func_name=func_name,
            openapi_spec=openapi_spec,
            path=path,
            method=method,
            description=description
        ))
    # tools.append(NoTool())

    
    prompt = CustomZeroShotAgent.create_prompt(
        tools,
        prefix=test_prompt_v1["prefix"],
        suffix=test_prompt_v1["suffix"],
        format_instructions=test_prompt_v1["format_instructions"],
        input_variables=["input", "agent_scratchpad"]
    )

    api_dataset = []

    for ans in api_info.get("Instances", []):
        process = []
        trainable = []

        if not ans.get("intermediate_steps"):
            continue

        if len(ans["intermediate_steps"]) > 5:
            continue
        
        question = ans["input"].rsplit("\nHint: ", 1)[0]
        prefix = prompt.format(input=question, agent_scratchpad="")
        process.append(prefix + " ")
        trainable.append(False)

        used_tools = set()
        for step in ans["intermediate_steps"]:
            thought_action = rreplace(step[0][2][1:], "\nAction Input:", "\nASSISTANT Action Input:", 1)
            thought_action = rreplace(thought_action, "\nAction:", "\nASSISTANT Action:", 1)
            process.append(thought_action + "\nASSISTANT Observation: ")
            trainable.append(True)
            process.append(step[1] + "\nASSISTANT Thought: ")
            trainable.append(False)

            used_tools.add(step[0][0])
        used_tools = list(used_tools)
        if len(used_tools) == 1 and used_tools[0] == "getDetails":
            continue

        process.append(f"{ans.get('Final Thought', 'I can reponse to the user now.')}\nASSISTANT Response: {ans['output']}")
        trainable.append(True)

        api_dataset.append([process, trainable])
    
    return api_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-out", "--output_path", type=str, default="./data")
    args = parser.parse_args()

    api_data = json.load(open(args.api_data_path, "r", encoding="utf-8"))

    all_data = []

    for api in api_data:
        if api.get("Function_Description") is None:
            continue
        data = build_dataset(api)
        all_data.extend(data)

    all_lengths = {}
    for i in all_data:
        if len(i[0]) not in all_lengths:
            all_lengths[len(i[0])] = 0
        all_lengths[len(i[0])] += 1
    print(all_lengths)
    print(len(all_data))

    json.dump(all_data, open(args.output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
