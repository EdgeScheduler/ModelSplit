import onnx
from onnx.onnx_ml_pb2 import NodeProto
from typing import List
from config import Config

class GraphNode():
    def __init__(self,node: NodeProto,index:int=-1):
        self.name=str(node.name)        # type: str

        # inputs of current node
        self.inputs=[]                  # type: list[str]

        # outputs of current node                                                          
        self.outputs=[]                 # type: list[str]

        # dependencies inputs of nodes that idx >= self.idx
        self.dependencies_inputs=[]     # type: list[str]

        # dependencies outputs of nodes that idx >= self.idx
        self.dependencies_outputs=[]    # type: list[str]

        # idx in raw-model, start with idx=0
        self.idx=index                  # type: int

    def __str__(self) -> str:
        return "id={}, name={}, inputs={}, outputs={}, dependencies_inputs={}, dependencies_outputs={}".format(self.idx,self.name,self.inputs,self.outputs, self.dependencies_inputs, self.dependencies_outputs)

    def IsConvergeNode(self)->bool:
        return True if len(self.dependencies_inputs)<2 else False

# enable: for node in model_analyzer: ...
class ModelAnalyzerIterator():
    def __init__(self,nodes) -> None:
        self.items=nodes
        self.index=0
    
    def __next__(self):
        if self.index<len(self.items):
            self.index+=1
            return self.items[self.index-1]
        else:
            raise StopIteration

class ModelAnalyzer():
    def __init__(self,model_name:str,onnx_path:str=None):
        self.modelName=model_name

        if onnx_path is None:
            self.onnxPath=Config.ModelSavePathName(model_name)
        else:
            self.onnxPath=onnx_path

        self.nodes=[]       # type: list[GraphNode]
        
        if not self.Init():
            pass

    def Init(self)->bool:
        try:
            model = onnx.load(self.onnxPath)

            # print(model)
            for idx,node in enumerate(model.graph.node):
                self.nodes.append(GraphNode(node=node,index=idx))

            self.RecordDependency()
        except Exception as ex:
            print("error: fail to init model-analyzer")
            print(str(ex))
            return False

    # don't consider extra-output in middle of model at this time
    def RecordDependency(self):
        dependency=set()
        for idx in range(len(self.nodes))[::-1]:
            if idx==len(self.nodes)-1:
                self.nodes[idx].dependencies_outputs=self.nodes[idx].outputs

            for input_name in self.nodes[idx].inputs:
                dependency.append(input_name)

            for output_name in self.nodes[idx].outputs:
                # if output_name in dependency:
                dependency.discard(output_name)

            self.nodes[idx].dependencies_inputs=list(dependency)

            if idx>0:
                # out=set(self.nodes[idx].dependencies_inputs) | set(self.nodes[idx-1].outputs)
                # self.nodes[idx-1].dependencies_outputs=list(out)
                self.nodes[idx-1].dependencies_outputs=self.nodes[idx].dependencies_inputs

    def SplitAndStoreChilds(self,childs: List[GraphNode]):
        '''
        split and store child onnx-models to disk with childs as start node. if real start not in, add we will add it automatically.
        '''
        
        childs=sorted(childs,lambda x: x.idx)
        if len(childs)<1 or childs[0].idx!=0:
            childs.insert(0,self.nodes[0])

        for child_idx in range(len(childs)):
            start_node=childs[child_idx]
            end_node=self.nodes[-1]
            if child_idx+1<len(childs):
                end_node=childs[child_idx+1]
            onnx.utils.extract_model(self.onnxPath, Config.ChildModelSavePathName(self.modelName,child_idx), start_node.dependencies_inputs, end_node.outputs)

    def GetConvergeNode(self)->List[GraphNode]:
        result=[]
        for node in self.nodes:
            if node.IsConvergeNode():
                result.append(node)
        return result

    def __str__(self) -> str:
        return "".join([str(node)+"\n" for node in self.nodes])

    def __getitem__(self,index)->GraphNode:
        return self.nodes[index]

    def __len__(self)->int:
        return len(self.nodes)

    def __iter__(self):
        return ModelAnalyzerIterator(self.nodes)