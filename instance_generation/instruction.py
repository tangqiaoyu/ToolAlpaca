import os
import re
import json
import argparse
from string import Template

from tool_maker.get_elements import get_elements


def clear_instructions(instructions):
    input_list = [inst.strip() for inst in instructions if inst.strip() != ""]
    output_list = []
    for i in input_list:
        # Remove starting number, period, and space, if they exist
        processed = re.sub(r'^\d*\.\s*', '', i).strip()
        # Remove starting and ending quotes, if they exist in pair
        if processed.startswith('"') and processed.endswith('"'):
            processed = processed[1:-1].strip()
        if processed:
            output_list.append(processed)
    return output_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp", "--template_path", type=str, default="./prompts/Instruction.txt")
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-out", "--output_dir", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api_data = json.load(open(args.api_data_path, "r"))

    instruction_template = Template(open(args.template_path, "r").read())
    _, instructions = get_elements(
        instruction_template,
        api_data,
        args.output_dir,
        "Instructions",
        temperature=0.5,
        simple_update=False
    )

    for i, j in zip(api_data, instructions):
        inst = [k for k in j["choices"][0]["message"]["content"].split("\n") if k != ""]
        i["Instructions"] = clear_instructions(inst)
    
    json.dump(
        api_data,
        open(os.path.join(args.output_dir, "api_data.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4
    )