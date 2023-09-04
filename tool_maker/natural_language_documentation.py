import os
import json
import argparse


def preprocess_description(desc):
    if desc.strip() == "":
        return ""
    if not desc.strip().endswith("."):
        desc += "."
    return desc


def generate_schema_description(schema, skip_type_for_simple=False, inner_array=False):
    if "oneOf" in schema and len(schema) == 1 and isinstance(schema["oneOf"], list):
        return " OR ".join([generate_schema_description(s, skip_type_for_simple, inner_array) for s in schema['oneOf']])
    if '$ref' in schema:
        return f"{schema['$ref'].replace('/components/schemas/', '')}"

    data_type = schema.get('type', '')
    if data_type == 'array':
        item_schema = schema.get('items', {})
        item_description = generate_schema_description(item_schema, skip_type_for_simple, inner_array=True)
        if skip_type_for_simple:
            return f"Array[{item_description}]"
        else:
            return f"Array[{item_description}]. " + preprocess_description(schema.get('description', ''))
    elif data_type == 'object':
        object_description = "Object"
        if 'properties' in schema:
            object_description += "{"
            props = []
            for prop_name, prop_data in schema['properties'].items():
                prop_description = generate_schema_description(prop_data, skip_type_for_simple)
                props.append(f"{prop_name}{f': {prop_description}' if prop_description else ''}")
            object_description += ", ".join(props)
            object_description += "}"
        if skip_type_for_simple:
            return object_description
        else:
            return f"{object_description}. " + preprocess_description(schema.get('description', ''))
    else:
        if inner_array:
            return data_type
        if skip_type_for_simple:
            return ""
        if data_type != "":
            data_type = f"{data_type}."
        enum_hint = ""
        if 'enum' in schema:
            enum_hint = f"One of: [{', '.join([str(i) for i in schema['enum']])}]."
        desc = preprocess_description(schema.get('description', ''))
        out = " ".join([i for i in [data_type, desc, enum_hint] if i]).strip()
        # if out[-1] == ".":
        #     out = out[:-1]
        return out


projections = {}
def generate_function_descriptions(openapi_spec):
    paths = openapi_spec['paths']
    descriptions = []
    global projections
    projections = {}
    for path, path_data in paths.items():
        if "parameters" in path_data:
            global_parameters = path_data['parameters']
        else:
            global_parameters = []
        for method, method_data in path_data.items():
            if method not in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options', 'trace']:
                continue
            path_code = path[1:].replace("/", "_").replace("{", "").replace("}", "") + "_" + method
            function_name = method_data.get('operationId', path_code)
            if function_name in projections:
                print(f"Duplicate function name {function_name}")
            projections[function_name] = [path, method]
            summary = method_data.get('summary', method_data.get('description', ""))

            parameters = method_data.get('parameters', []) + global_parameters
            input_params = {}
            for parameter in parameters:
                if 'name' not in parameter:
                    continue
                param_description = preprocess_description(parameter.get('description', ""))
                # print(param_description)
                input_params[parameter['name']] = {
                    'description': generate_schema_description(parameter.get('schema', {})) + ' ' + param_description,
                    'required': parameter.get('required')
                }

            # Add requestBody handling
            if 'requestBody' in method_data:
                request_body = method_data['requestBody']
                if 'content' in request_body:
                    content_type = list(request_body['content'].keys())[0]
                    if 'schema' in request_body['content'][content_type]:
                        schema = request_body['content'][content_type]['schema']
                        if '$ref' in schema:
                            input_params['$ref'] = {
                                'description': schema['$ref'].replace("/components/schemas/", ""),
                                'required': None
                            }
                        for prop_name, prop_schema in schema.get('properties', {}).items():
                            input_params[prop_name] = {}
                            input_params[prop_name]['description'] = generate_schema_description(prop_schema)
                        if "required" in schema:
                            for required_prop in schema['required']:
                                input_params[required_prop]['required'] = True
            responses = method_data['responses']
            output_params = []
            responses_code = [i for i in responses.keys() if i.startswith("2")]
            content_type = ""
            if len(responses_code) > 0:
                response = responses[responses_code[0]]
                output_description = preprocess_description(response['description'])

                if 'content' in response:
                    # if len(response['content']) > 1:
                        # print(response['content'])
                        # raise Exception("Multiple content types not supported")
                    content_type = list(response['content'].keys())[0]
                    if 'schema' in response['content'][content_type]:
                        schema = response['content'][content_type]['schema']
                        detailed_description = generate_schema_description(schema, skip_type_for_simple=True)
                        output_params.append(detailed_description)
            else:
                output_description = ""
            function_description = {
                'name': function_name,
                'summary': summary,
                'input': input_params,
                'output': {
                    'description': output_description,
                    'content_type': content_type,
                    'details': output_params[0] if output_params else ""
                }
            }
            descriptions.append(function_description)

    return descriptions


def generate_component_descriptions(openapi_spec):
    components = openapi_spec.get('components', {})
    schemas = components.get('schemas', {})
    descriptions = []

    for component_name, component_data in schemas.items():
        detailed_description = generate_schema_description(component_data, skip_type_for_simple=True)

        component_description = {
            'name': component_name,
            'description': detailed_description
        }
        descriptions.append(component_description)

    return descriptions


def get_function_descriptions(documentation):
    
    function_descriptions = generate_function_descriptions(documentation)
    # print(function_descriptions)
    outputs = {}
    output_string = ""
    for func_desc in function_descriptions:
        output_string += f"{func_desc['name']}: "
        func_string = ""
        func_string += f"{func_desc['summary']}\n"
        func_string += "Parameters: "
        input_params = []
        for name, details in func_desc['input'].items():
            if name == "$ref":
                input_params.append(details['description'])
                continue
            required = "Required. " if details.get('required') else ""
            input_params.append(f'"{name}": "{required}{details.get("description", "")}"')
            # input_params.append('{' + f'"name": "{name}", "description": "{required}{details.get("description", details.get("summary"))}"' + '}')
        func_string += "{" + ", ".join(input_params) + "}\n"
        func_string += f"Output: {func_desc['output']['description']}\n - Format: {func_desc['output']['content_type']}\n - Structure: {func_desc['output']['details']}\n"
        output_string += func_string
        outputs[func_desc['name']] = func_string.strip()
    component_descriptions = generate_component_descriptions(documentation)
    component_string = ""
    if component_descriptions:
        component_string += "\nThe detailed output format for the tools is outlined below:\n"
        for component_description in component_descriptions:
            component_string += f"#{component_description['name']}: {component_description['description']}\n"
    output_string += component_string
    outputs["components"] = component_string.strip()

    return output_string, outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-api", "--api_data_path", type=str, default="")
    parser.add_argument("-out", "--output_path", type=str, default="")
    parser.add_argument("-doc_key", type=str, default="Documentation")
    args = parser.parse_args()
    
    if args.output_path == "":
        args.output_path = args.api_data_path

    APIs = json.load(open(args.api_data_path, "r", encoding="utf-8"))
    APIs = [api for api in APIs if api["Documentation"] is not None]
    
    for api in APIs:
        API_name = api["Name"]
        try:
            function_description, structured_func_des = get_function_descriptions(json.loads(api[args.doc_key]))
            api["NLDocumentation"] = function_description
            api["Function_Description"] = structured_func_des
            api["Function_Projection"] = projections
        except json.decoder.JSONDecodeError as e:
            print("=" * 50)
            print(api["Name"])
            print(e)
            api["NLDocumentation"] = None
            api["Function_Description"] = None
            api["Function_Projection"] = None
    json.dump(APIs, open(args.output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
