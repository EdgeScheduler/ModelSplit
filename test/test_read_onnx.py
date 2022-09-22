import onnx

model = onnx.load('whole_model.onnx')
graph=model.graph
# print(model)
for node in graph.node:
    print(type(node))
    break