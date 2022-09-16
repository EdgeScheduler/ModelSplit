def FilterParamsAndInput(paramsDict, childModelId, input, params, pre_output=None):
    current_model_params = paramsDict[str(childModelId)]
    new_input = {}
    new_params = {}
    for arg in current_model_params:
        arg_name = arg["name"]
        if arg_name in input.keys():
            new_input[arg_name] = input[arg_name]
        if arg_name in params.keys():
            new_params[arg_name] = params[arg_name]

    # add pre output as current input
    for arg in current_model_params:
        if "call_" not in arg["name"]:
            continue
        if arg["name"] not in new_input.keys():
            new_input[arg["name"]] = pre_output
    return new_input, new_params

def FilterChildParams(paramsDict, idx, params):
    current_model_params = paramsDict[str(idx)]
    new_params = {}
    for arg in current_model_params:
        arg_name = arg["name"]
        if arg_name in params.keys():
            new_params[arg_name] = params[arg_name]
    return new_params

def FilterChildInput(paramsDict, idx, input, pre_output=None):
    current_model_params = paramsDict[str(idx)]
    new_input = {}
    # add global input to child model-input
    for arg in current_model_params:
        arg_name = arg["name"]
        if arg_name in input.keys():
            new_input[arg_name] = input[arg_name]

    # add pre output as current input
    for arg in current_model_params:
        if "call_" not in arg["name"]:
            continue
        if arg["name"] not in new_input.keys():
            new_input[arg["name"]] = pre_output
    return new_input