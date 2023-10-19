import os
import re
import json
import argparse
from string import Template

from utils import openai_chat_completions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp", "--template_path", type=str, default="./prompts/Evaluation.txt")
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-gold", "--golden_answer_path", type=str, default="")
    parser.add_argument("-out", "--output_path", type=str, default="")
    parser.add_argument("--continue_run", action="store_true", default=False)
    args = parser.parse_args()

    template = Template(open(args.template_path, "r").read())

    api_data = json.load(open(args.api_data_path, "r"))
    if os.path.exists(args.golden_answer_path):
        golden_answer = json.load(open(args.golden_answer_path, "r"))
        for k, v in zip(api_data, golden_answer):
            k["Golden_Answers"] = v["Golden_Answers"]

    original_data = {}
    original_data["statistics"] = {
        "num": 0,
        "error_num": 0,
        "process": {
            "Yes": 0,
            "No": 0,
            "Uncertain": 0
        },
        "response": {
            "Yes": 0,
            "No": 0,
            "Uncertain": 0
        },
        "both": 0
    }

    exist_ids = None
    if args.continue_run:
        original_data = json.load(open(args.output_path, "r"))
        exist_ids = {i: [j["id"] for j in original_data[i]] for i in original_data if i != "statistics"}

    retry_cases = None

    for api_info in api_data:
        api_name = api_info.get("Name", api_info.get("API"))
        if retry_cases is not None and api_name not in retry_cases:
            continue
        if exist_ids is None or api_name not in exist_ids:
            original_data[api_name] = []
        for ques_id, ques in enumerate(api_info["Instructions"]):
            if exist_ids is not None and ques_id in exist_ids.get(api_name, []):
                continue
            if retry_cases is not None and ques_id not in retry_cases.get(api_name, []):
                continue
            original_data["statistics"]["num"] += 1
            if "intermediate_steps" not in api_info["Instances"][ques_id] or len(api_info["Instances"][ques_id]["intermediate_steps"]) == 0:
                original_data["statistics"]["error_num"] += 1
                tmp = {
                    "id": ques_id,
                    "input": "",
                    "output": ""
                }
                original_data[api_name].append(tmp)
                continue

            golden_answer = api_info["Golden_Answers"][ques_id]
            standard_answer = ""
            for ans_id, ans in enumerate(golden_answer):
                standard_answer += f"{ans_id + 1}. Function: {ans['Action']}\nParameters: {ans['Action_Input']}\n"
            
            solution = ""
            for sol_id, sol in enumerate(api_info["Instances"][ques_id]["intermediate_steps"]):
                solution += f"{sol_id + 1}. Function: {sol[0][0]}\nParameters: {sol[0][1]}\nRetruns: {sol[1]}\n"
            solution += f"{sol_id + 2}. Final Response: {api_info['Instances'][ques_id]['output']}"

            prompt = template.substitute(
                documentation=api_info["NLDocumentation"],
                instruction=ques,
                standard=standard_answer,
                solution=solution
            )
            
            prompt = [{"role": "user", "content": prompt}]
            output = openai_chat_completions(prompt, model="gpt-4-0613", temperature=0.2)
            text = output["choices"][0]["message"]["content"]
            
            results_text = text.split('## Results', 1)[-1]

            process_correctness_match = re.search('Process Correctness: (\w+)', results_text)
            process_correctness_word = process_correctness_match.group(1) if process_correctness_match else ""

            final_response_correctness_match = re.search('Final Response Correctness: (\w+)', results_text)
            final_response_correctness_word = final_response_correctness_match.group(1) if final_response_correctness_match else ""

            tmp = {
                "id": ques_id,
                "input": prompt,
                "output": text,
                "process_correctness": process_correctness_word,
                "final_response_correctness": final_response_correctness_word
            }

            original_data["statistics"]["process"][process_correctness_word] += 1
            original_data["statistics"]["response"][final_response_correctness_word] += 1
            if process_correctness_word == final_response_correctness_word == "Yes":
                original_data["statistics"]["both"] += 1

            original_data[api_name].append(tmp)
            
            json.dump(
                original_data,
                open(os.path.join(args.output_path), "w"),
                indent=4,
                ensure_ascii=False
            )
    json.dump(
        original_data,
        open(os.path.join(args.output_path), "w"),
        indent=4,
        ensure_ascii=False
    )
