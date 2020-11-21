from torchbenchmark import load_model, model_names
from contextlib import redirect_stderr, redirect_stdout
import torch
from torch.package import PackageExporter, PackageImporter
import tempfile
from pathlib import Path
# something about protobufs crashing things...
import torch.utils.tensorboard

no_cpu_impl = [
    'Background_Matting',
    'maskrcnn_benchmark',
    'moco',
    'pytorch_CycleGAN_and_pix2pix',
    'tacotron2',
]

def tpe(x):
    if isinstance(x, list) or isinstance(x, tuple):
       return f"({','.join(str(tpe(e)) for e in x)})"
    else:
        return type(x)

def check_close(a, b):
    if isinstance(a, (list, tuple)):
        for ae, be in zip(a, b):
            check_close(ae, be)
    else:
        print(torch.max(torch.abs(a - b)))
        assert torch.allclose(a, b)

t2tsource = """
import torch
# annotating with jit interface messes up packaging
class TensorToTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
"""

def BERT_pytorch(module, exporter):
    exporter.save_source_string('torchbenchmark.models.BERT_pytorch.bert_pytorch.model.utils.tensor2tensor', t2tsource)

def demucs(module, exporter):
    exporter.mock_module('tqdm')

def dlrm(module, exporter):
    exporter.mock_modules(['**.dlrm.dlrm_data_pytorch', 'onnx', 'sklearn**'])

def fastNLP(module, exporter):
    exporter.mock_modules(['fastNLP.core**', 'fastNLP.io.file_utils'])
    exporter.save_source_string('regex', """\
# result is unused but the pattern is compiled when the file is imported
def compile(*args, **kwargs):
    return None
""")


def yolov3(module, exporter):
    # clean up some numpy objects in the model
    module.module_defs = None
    module.version = None
    module.seen = None
    exporter.mock_module('yolo_utils.utils**')

with open('model_logs.txt', 'w') as model_logs:

    for model_name in model_names():
        result_file = f'results/{model_name}'
        if model_name in no_cpu_impl or Path(result_file).exists():
            continue
        print(f'packaging {model_name}')
        with redirect_stdout(model_logs), redirect_stderr(model_logs):
            Model = load_model(model_name)
            m = Model(jit=False, device='cpu')
            module, eg = m.get_module()
            module.eval()

        with tempfile.TemporaryDirectory(dir='.') as tempdirname:
            model_path = Path(tempdirname) / model_name
            with PackageExporter(str(Path(tempdirname) / model_name)) as exporter:
                exporter.mock_modules(['numpy', 'scipy'])
                preproc = globals().get(model_name, lambda _, __: None)
                preproc(module, exporter)
                exporter.save_pickle('model', 'model.pkl', module)
                exporter.save_pickle('model', 'eg.pkl', eg)

            importer = PackageImporter(str(model_path))
            module2 = importer.load_pickle('model', 'model.pkl')
            eg2 = importer.load_pickle('model', 'eg.pkl')

            with torch.no_grad():
                r = module(*eg)
                r2 = module2(*eg2)
            check_close(r, r2)
            model_path.replace(result_file)

