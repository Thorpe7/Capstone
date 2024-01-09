import torch
import torch.onnx
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(model, data_loader):
    for tmp_input, _ in data_loader:
        break
    tmp_input = tmp_input.to("cuda")
    writer = SummaryWriter("runs/model_visualization")
    writer.add_graph(model, tmp_input)
    writer.close()


# Assuming 'model' is your PyTorch model and 'input' is a sample input tensor
def use_netron(model, data_loader):
    dummy_input = torch.randn(128, 1, 126, 126, device="cuda")
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
