"""Export BubbleUNet at a fixed 1024 x 768 input size."""

import argparse
import copy
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from model import load_model

HEIGHT, WIDTH = 1024, 768


class NHWCModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


@torch.inference_mode()
def export_onnx(model, output, fp16=False):
    import onnx

    model = copy.deepcopy(model).half() if fp16 else model
    sample = torch.zeros(1, HEIGHT, WIDTH, 1,
                         dtype=torch.float16 if fp16 else torch.float32)
    torch.onnx.export(
        NHWCModel(model).eval(), sample, str(output),
        input_names=["image"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output)))


@torch.inference_mode()
def export_torchscript(model, output):
    sample = torch.zeros(1, 1, HEIGHT, WIDTH)
    traced = torch.jit.freeze(torch.jit.trace(model, sample).eval())
    traced.save(str(output))
    restored = torch.jit.load(str(output))
    torch.testing.assert_close(restored(sample), model(sample), rtol=2e-5, atol=2e-4)


def prepare_wasm(source: Path, output: Path) -> None:
    """Promote FP16 values and graph types to FP32 for ONNX Runtime Web WASM.

    This preserves rounded FP16 values; it does not recover FP32 precision.
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    def promote_tensor(tensor):
        if tensor.data_type == TensorProto.FLOAT16:
            tensor.CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(tensor).astype(np.float32), tensor.name))

    def promote_graph(graph):
        for tensor in graph.initializer:
            promote_tensor(tensor)
        for value in [*graph.input, *graph.output, *graph.value_info]:
            if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
                value.type.tensor_type.elem_type = TensorProto.FLOAT
        for node in graph.node:
            for attr in node.attribute:
                if node.op_type == "Cast" and attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                if attr.type == onnx.AttributeProto.TENSOR:
                    promote_tensor(attr.t)
                elif attr.type == onnx.AttributeProto.TENSORS:
                    for tensor in attr.tensors:
                        promote_tensor(tensor)
                elif attr.type == onnx.AttributeProto.GRAPH:
                    promote_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for graph in attr.graphs:
                        promote_graph(graph)

    model = onnx.load(str(source))
    promote_graph(model.graph)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))


def export_tflite(model, output):
    import onnx2tf
    from ai_edge_litert.interpreter import Interpreter

    with TemporaryDirectory(dir=output.parent) as directory:
        directory = Path(directory)
        source = directory / "model.onnx"
        export_onnx(model, source)
        onnx2tf.convert(
            input_onnx_file_path=str(source),
            output_folder_path=str(directory / "converted"),
            keep_nwc_or_nhwc_or_ndhwc_input_names=["image"],
            tflite_backend="tf_converter", non_verbose=True,
        )
        generated = directory / "converted/model_float16.tflite"
        interpreter = Interpreter(model_path=str(generated))
        interpreter.allocate_tensors()
        for detail, shape in zip(
            [interpreter.get_input_details()[0], interpreter.get_output_details()[0]],
            [[1, HEIGHT, WIDTH, 1], [1, HEIGHT, WIDTH, 3]],
        ):
            if detail["shape"].tolist() != shape or detail["dtype"] != np.float32:
                raise RuntimeError("Expected TFLite NHWC tensors with float32 I/O")
        shutil.copyfile(generated, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Training .pth checkpoint; FP16 ONNX also accepted for WASM")
    parser.add_argument("--format", required=True,
                        choices=["pt", "onnx-fp16", "onnx-fp16-wasm", "tflite-fp16"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "onnx-fp16-wasm" and args.source.suffix.lower() == ".onnx":
        prepare_wasm(args.source, args.output)
    else:
        model, _ = load_model(args.source)
        if args.format == "pt":
            export_torchscript(model, args.output)
        elif args.format == "onnx-fp16":
            export_onnx(model, args.output, fp16=True)
        elif args.format == "onnx-fp16-wasm":
            with TemporaryDirectory(dir=args.output.parent) as directory:
                source = Path(directory) / "model_fp16.onnx"
                export_onnx(model, source, fp16=True)
                prepare_wasm(source, args.output)
        else:
            export_tflite(model, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
