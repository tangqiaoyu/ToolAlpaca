import os
import json
import logging
import argparse

import openai
import requests
from tqdm import tqdm
from langchain.llms import OpenAI
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from agent.get_agent import get_agent
from agent.agent_prompts import prompt_proj
from utils import load_openapi_spec, analyze_openapi_spec


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("-api", "--api_data_path", type=str, required=True)
parser.add_argument("-out", "--output_dir", type=str, default="")
parser.add_argument("-llm", type=str, default=None)
parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5678")
parser.add_argument("--output_prefix", type=str, default="api_data")
parser.add_argument("--agent_prompt", type=str, default="train_v2")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--length", type=int, default=-1)
parser.add_argument("--use_cache", action="store_true", default=False)
parser.add_argument("--real", action="store_true", default=False)
parser.add_argument("--without_getDetails", action="store_true", default=False)
args = parser.parse_args()

if args.llm is None or args.llm.lower() in ["gpt3", "gpt-3"]:
    llm = OpenAI(temperature=0.0)
elif args.llm.lower() in ["chatgpt"]:
    llm = ChatOpenAI(temperature=0.0)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.llm, trust_remote_code=True).half()

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        device=args.device,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=generator)

api_data = json.load(open(args.api_data_path, "r"))

if args.length == -1:
    args.length = len(api_data) - args.offset

api_data = api_data[args.offset: args.offset + args.length]
final_output_path = os.path.join(args.output_dir, f"{args.output_prefix}_{args.offset}_{args.offset+args.length}.json")

if args.use_cache:
    res = requests.get(f"{args.server_url}/__simulator_cache__/open")

for api_idx, api in tqdm(enumerate(api_data)):
    api["Instances"] = []
    if "Instructions" not in api or len(api["Instructions"]) == 0:
        continue
    openapi_spec = load_openapi_spec(api["Documentation"])
    input_valid, output_valid = analyze_openapi_spec(openapi_spec)
    if input_valid and output_valid:
        agent = get_agent(
            llm=llm,
            api_data=api,
            server_url=args.server_url,
            agent_prompt=prompt_proj[args.agent_prompt],
            enable_getDetails=not args.without_getDetails,
        )
        Answers = []
        for idx, inst in enumerate(api["Instructions"]):
            if len(api.get("Authentication", [])) > 0:
                inst += "\nAuthentication information: " + \
                      " ".join([f"{k}={v}" for k, v in api["Authentication"].items()])
            try:
                output = agent(inst)
                json.dumps(output, ensure_ascii=4)
            except json.JSONDecodeError:
                output = str(output)
            # except Exception as e:
            #     logger.error(e)
            #     output = {"error": str(e)}  
            
            if args.use_cache:
                res = requests.get(f"{args.server_url}/__simulator_cache__/clear/{api['Name']}")
                print(res.text)

            Answers.append(output)

        api["Instances"] = Answers
        json.dump(
            api_data, 
            open(final_output_path, "w", encoding="utf-8"), 
            indent=4, 
            ensure_ascii=False
        )