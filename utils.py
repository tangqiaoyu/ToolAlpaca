import os
import re
import time
import json
import asyncio
import concurrent.futures
from functools import partial
from typing import Union, Dict, List, Type, Any, Callable

import httpx
import openai
import jsonref
import requests
from tqdm import tqdm
from openapi_spec_validator import validate_spec
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    retry_base,
)

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_retry_decorator(
    max_retries: int = 5
) -> Callable[[Any], Any]:
    errors = [
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
    ]

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    retry_instance: "retry_base" = retry_if_exception_type(errors[0])
    for error in errors[1:]:
        retry_instance = retry_instance | retry_if_exception_type(error)
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_instance
    )


def openai_chat_completions(
    messages: Union[List[Dict], List[List[Dict]]],
    model=None,
    temperature=1,
    top_p=1,
    n=1,
    stop=None,
    max_tokens=None,
    max_retries=5,
    num_workers=4,
    **kwargs
):
    if model is None:
        model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")

    retry_decorator = create_retry_decorator(max_retries)

    @retry_decorator
    def _completion_with_retry(messages, *args, **kwargs):
        return openai.ChatCompletion.create(messages=messages, *args, **kwargs)
    
    without_batch = len(messages) > 0 and isinstance(messages[0], dict)
    if without_batch:
        messages = [messages]

    partial_func = partial(
        _completion_with_retry,
        model=model,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        **kwargs
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        all_completions = list(executor.map(partial_func, messages))
    
    if without_batch:
        all_completions = all_completions[0]
    return all_completions


async def async_openai_chat_completions(
    messages: Union[List[Dict], List[List[Dict]]],
    model=None,
    temperature=1,
    top_p=1,
    n=1,
    stop=None,
    max_tokens=None,
    max_retries=5,
    **kwargs
):
    if model is None:
        model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")

    retry_decorator = create_retry_decorator(max_retries)

    @retry_decorator
    async def _completion_with_retry(*args, **kwargs):
        return await openai.ChatCompletion.acreate(*args, **kwargs)

    
    without_batch = len(messages) > 0 and isinstance(messages[0], dict)
    if without_batch:
        messages = [messages]
    
    all_completions = []
    for batch_id, batch in tqdm(enumerate(messages)):
        data = {
            "model": model,
            "messages": batch,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "max_tokens": max_tokens,
            **kwargs
        }

        all_completions.append(await _completion_with_retry(**data))
    
    if without_batch:
        all_completions = all_completions[0]
    return all_completions


def validate_openapi_file(json_spec_str: str):
    try:
        if "Open AI Klarna product Api" in json_spec_str:
            return "DO NOT copy the example!"
        spec = json.loads(json_spec_str)
        validate_spec(spec)
        return None
    except Exception as e:
        if isinstance(e, KeyError):
            print("KeyError: ", e)
        return "{exception_name}: {exception_message}".format(
            exception_name=type(e).__name__,
            exception_message=str(e).split('\n\n')[0]
        )


def load_openapi_spec(json_str: str, replace_refs=False):
    openapi_spec = json.loads(json_str)
    if replace_refs:
        openapi_spec = jsonref.JsonRef.replace_refs(openapi_spec)
    return openapi_spec


def escape(string):
    return string.replace("{", "{{").replace("}", "}}")


def is_text_based(media_type):
    return media_type.startswith("text/") or media_type == "application/json"


def analyze_openapi_spec(spec):
    input_text_based = True
    output_text_based = True
    paths = spec.get('paths', {})
    
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:

                content = operation.get('requestBody', {}).get('content', {})
                if content:
                    input_text_based = any(is_text_based(media_type) for media_type in content.keys())

                # Analyze output
                responses = operation.get('responses', {})
                for status, response in responses.items():
                    content = response.get('content', {})
                    if content:
                        output_text_based = any(is_text_based(media_type) for media_type in content.keys())

                if not (input_text_based and output_text_based):
                    break
        if not (input_text_based and output_text_based):
            break

    return input_text_based, output_text_based


def add_server_url_to_spec(api_data):
    for api_info in api_data:
        api = json.loads(api_info["Documentation"])
        if not api.get("servers") or not api["servers"][0].get("url"):
            api["servers"] = [{"url": api_info["Link"]}]
        api_info["Documentation"] = json.dumps(api)


def parse_json_string(json_string, load=True):
    json_start = min([idx for idx in (json_string.find('{'), json_string.find('[')) 
                      if idx != -1])
    json_end = max([idx for idx in (json_string.rfind('}'), json_string.rfind(']')) 
                    if idx != -1]) + 1

    valid_json_string = json_string[json_start:json_end]
    if load:
        return json.loads(valid_json_string)
    else:
        return valid_json_string