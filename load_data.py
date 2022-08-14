from tvm import relay
import onnx
import os
from config import Config
from config import OnnxModelUrl
import time
import requests 

def easy_load_from_onnx(save_name,input_dict={}, download_url=OnnxModelUrl.Default, auto_path=True):
    '''
    load model from "$project_path/onnxs/$save_name/$save_name.onnx" without any redundant operate. It may not fit some complex model.

    Parameters
    ----------
    save_name : str
        onnx-file name. if auto_path is True, load file from "$project_path/onnxs/$save_name/$save_name.onnx", or load from "$save_name" directly.
    input_dict : dict => {str: tuple}
        input-label to tensor-shape
    download_url : str
        url to download onnx-model from internet, it only works when local-file is not exist.
    auto_path: bool
        whether transform "$save_name" to "$project_path/onnxs/$save_name/$save_name.onnx"

    Returns
    -------
    irModule : tvm.ir.module.IRModule
        The relay module for compilation

    params : dict => {"label": tvm.nd.NDArray}
        The parameter dict to be used by relay
    '''
    filepath=save_name
    if auto_path:
        filepath=Config.ModelSavePathName(filepath)

    if not os.path.exists(filepath):
        if len(download_url)<1:
            print("onnx file not exist, you can give download url by set download_url=$URL.")
            return None,{}
        else:
            if not download(download_url,filepath):
                print("fail to load onnx-model from:",filepath)
                return None,{}

    try:
        onnx_model=onnx.load(filepath)
        irModule, params = relay.frontend.from_onnx(onnx_model,input_dict)
        irModule=relay.transform.InferType()(irModule)                    # tvm.ir.module.IRModule
    except Exception as ex:
        print("fail to load onnx from %s"%(filepath))
        return None,{}

    print("success to load onnx from %s"%(filepath))
    return irModule, params

def download(url, path)->bool:
    """Downloads the file from the internet.
    Set the input options correctly to overwrite or do the size comparison

    Parameters
    ----------
    url : str
        Download url.

    path : str
        Local file path to save downloaded file.

    retries: int, optional
        Number of time to retry download, defaults to 3.

    Returns
    -------
    result : bool
        success or fail
    """
    
    print("start to download file from:",url)
    start=time.time()

    try:
        with open(path, "wb") as code:
            code.write(requests.get(url).content)
    except Exception as ex:
        print("fail to download from %s, cost time=%fs. error info: %s"%(url,time.time()-start,str(ex)))
        return False

    print("success download and save to %s, cost time=%fs."%(url,time.time()-start))
    return True

    

    
