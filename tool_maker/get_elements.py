import os
import json
import logging
from string import Template

import argparse

from utils import (
    openai_chat_completions,
    validate_openapi_file,
    parse_json_string,
    add_server_url_to_spec
)


def get_prompts(template, api_data):
    all_prompts = []
    for api in api_data:
        prompt = template.substitute(**api)
        all_prompts.append([{"role": "user", "content": prompt}])

    return all_prompts


def get_elements(
    template,
    api_data,
    output_dir,
    element_name,
    backup=True,
    temperature=1,
    simple_update=True
):
    all_prompts = get_prompts(template, api_data)
    outputs = openai_chat_completions(
        all_prompts,
        temperature=temperature
    )

    if backup:
        json.dump(
            outputs,
            open(os.path.join(output_dir, f"{element_name}.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4
        )
    
    if simple_update:
        for i, j in zip(api_data, outputs):
            i[element_name] = ""
            i[element_name] = j["choices"][0]["message"]["content"]

    return all_prompts, outputs


def validate_documentation(
        all_prompts,
        documentations,
        output_dir,
        retry=0,
        max_retry=1,
        backup=True,
        temperature=1
    ):
    error_messages = []
    error_idx = []

    for idx, doc in enumerate(documentations):
        if "choices" not in doc:
            error_idx.append(idx)
            error_messages.append([all_prompts[idx][0]])
            continue
            
        finish_reason = doc["choices"][0]["finish_reason"]
        openapi_spec = parse_json_string(doc["choices"][0]["message"]["content"], load=False)
        if len(all_prompts[idx]) > 1:
            openapi_spec = all_prompts[idx][1]["content"] + openapi_spec

        if finish_reason == "length":
            origin_openapi_spec = openapi_spec
            openapi_spec = "".join([i.strip() for i in openapi_spec.split("\n")])
            if openapi_spec == origin_openapi_spec:
                continue
            error_idx.append(idx)
            error_messages.append([
                all_prompts[idx][0],
                {"role": "assistant", "content": openapi_spec}
            ])
        else:
            error_info = validate_openapi_file(openapi_spec)
            if error_info is not None:
                error_idx.append(idx)
                error_messages.append([all_prompts[idx][0]])
    
    valid_documentations = [doc if idx not in error_idx else None for idx, doc in enumerate(documentations)]

    retry += 1
    if retry > max_retry:
        logging.info("Max retry reached, stop rerunning.")
        return valid_documentations
    
    if len(error_messages) > 0:
        rerun_outputs = openai_chat_completions(error_messages, temperature=temperature)
        if backup:
            json.dump(
                rerun_outputs,
                open(os.path.join(output_dir, f"Documentation_turn{retry}.json"), "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=4
            )
        rerun_results = validate_documentation(
            error_messages,
            rerun_outputs,
            output_dir,
            retry=retry,
            max_retry=max_retry,
            backup=backup,
            temperature=temperature
        )
        for error_i, origin_idx in enumerate(error_idx):
            print(error_i, error_idx)
            valid_documentations[origin_idx] = rerun_results[error_i]
    
    return valid_documentations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp", "--template_dir", type=str, default="./prompts")
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-out", "--output_dir", type=str, default="./data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api_data = json.load(open(args.api_data_path, "r"))

    introduction_template = Template(
        open(os.path.join(args.template_dir, "Introduction.txt"), "r").read()
    )
    get_elements(
        introduction_template,
        api_data,
        args.output_dir,
        "Introduction"
    )

    Functions_template = Template(
        open(os.path.join(args.template_dir, "Functions.txt"),"r").read()
    )
    get_elements(
        Functions_template,
        api_data,
        args.output_dir,
        "Functions",
        temperature=0.3
    )

    documentation_template = Template(
        open(os.path.join(args.template_dir, "Documentation.txt"), "r").read()
    )
    doc_inp, doc_out = get_elements(
        documentation_template,
        api_data,
        args.output_dir,
        "Documentation",
        simple_update=False,
        temperature=0.3
    )
    valid_docs = validate_documentation(
        doc_inp,
        doc_out,
        args.output_dir,
        max_retry=5,
        temperature=0.3
    )
    for i, j in zip(api_data, valid_docs):
        i["Documentation"] = j["choices"][0]["message"]["content"]
    
    add_server_url_to_spec(api_data)

    json.dump(
        api_data,
        open(os.path.join(args.output_dir, "api_data.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4
    )

