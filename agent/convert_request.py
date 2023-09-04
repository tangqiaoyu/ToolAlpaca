import logging
import requests
from datetime import datetime


logger = logging.getLogger(__name__)


def convert_type(ori_type):
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }

    type_mapping.update({j: i for i, j in type_mapping.items()})
    
    return type_mapping.get(ori_type)


def type_check(param_name, param_schema, input_params):
    type_check_error = []
    doc_type = convert_type(param_schema.get("type", ""))
    
    if doc_type and not isinstance(input_params[param_name], doc_type):
        type_error = True
        if (doc_type in [int, float, bool] and type(input_params[param_name]) == str) or \
            (doc_type == float and type(input_params[param_name]) == int):
            try:
                input_params[param_name] = doc_type(input_params[param_name])
                type_error = False
            except ValueError:
                pass
        if type_error:
            type_check_error.append((
                param_name,
                convert_type(doc_type),
                convert_type(type(input_params[param_name]))
            ))
    if "enum" in param_schema and input_params[param_name] != "" and\
        input_params[param_name] not in param_schema["enum"]:

        type_check_error.append((
            param_name,
            f'one of {param_schema["enum"]}',
            f'"{input_params[param_name]}"'
        ))
    return type_check_error
        

    
def call_api_function(input_params, openapi_spec, path, method, base_url=None):

    function_doc = openapi_spec["paths"][path][method]

    required_params = set()
    params = {
        "query": {},
        "header": {},
        "path": {},
        "cookie": {}
    }

    type_check_error = []
    for param_doc in function_doc.get("parameters", []):
        if param_doc.get("required"):
            required_params.add((param_doc["in"], param_doc["name"]))

        if param_doc["name"] in input_params:
            required_params.discard((param_doc["in"], param_doc["name"]))
            params[param_doc["in"]][param_doc["name"]] = input_params[param_doc["name"]]
            type_check_error.extend(type_check(param_doc["name"], param_doc, input_params))

    body_data = None
    required_body_params = None
    if "requestBody" in function_doc:
        body_data = {}
        request_body_schema = function_doc["requestBody"].get("content", {}).get("application/json", {}).get("schema", {})

        if "properties" in request_body_schema:
            required_body_params = set(request_body_schema.get("required", []))
            for property_name, property_value in request_body_schema["properties"].items():
                if property_name in input_params:
                    body_data[property_name] = input_params[property_name]
                    required_body_params.discard(property_name)
                    type_check_error.extend(type_check(property_name, property_value, input_params))
                    
    if len(type_check_error) > 0:
        error_str = "\n".join([f"Parameter type error: \"{i[0]}\", expected {i[1]}, but got {i[2]}. You need to change the input and try again." for i in type_check_error])
        raise ValueError(error_str)
    
    if required_params or required_body_params:
        missing_params = ", ".join([f'"{param[1]}"' for param in required_params])
        missing_params += [f'"{param}"' for param in required_body_params]
        raise ValueError(f"Missing required parameters: {', '.join(required_body_params)}. You need to change the input and try again.")

    base_url = openapi_spec['servers'][0]['url'] if base_url is None else base_url
    url = f"{base_url.rstrip('/')}{path.format(**params['path'])}"
    headers = {"Content-Type": "application/json"}
    headers.update(params["header"])

    logger.debug("request url: {url}")
    response = requests.request(
        method=method.upper(),
        url=url,
        params=params["query"],
        json=body_data,
        headers=headers,
        cookies=params["cookie"]
    )

    if "image" in response.headers.get("Content-Type", ""):
        image_extension = response.headers.get("Content-Type", "").split("/")[-1]
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        image_path = f"./images/{timestamp}.{image_extension}"
        with open(image_path, 'wb') as f:
            f.write(response.content)
        response._content = bytes(f"Recieved an image, saved in '{image_path}'.", "utf-8")
    
    logger.debug("url: {response.request.url}")
    logger.debug("body: {response.request.body}")
    logger.debug("headers: {response.request.headers}")

    return response
