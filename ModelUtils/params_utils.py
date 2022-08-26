import json


def parse_params_file(file_path):
    with open(file_path, "r") as f:
        params = json.load(f)
    return params


def filter_params(params_dict, idx, input, params, pre_output=None):
    total = params_dict[str(idx)]
    new_input = {}
    new_params = {}
    for _, param in enumerate(total):
        name = param["name"]
        if name in input.keys():
            new_input[name] = input[name]
        if name in params.keys():
            new_params[name] = params[name]

    # add pre output as current input
    for _, param in enumerate(total):
        if "call_" not in param["name"]:
            continue
        if param["name"] not in new_input.keys():
            new_input[param["name"]] = pre_output
    return new_input, new_params
