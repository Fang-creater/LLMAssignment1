import torch
import typing
import os

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration:int,
    out: str|os.PathLike|typing.BinaryIO|typing.IO[bytes],
):
    checkpoint = {
        "model_state":  model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint,out)

def load_checkpoint(
    src: str|os.PathLike|typing.BinaryIO|typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int :
    checkpoint = torch.load(src,map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["opt_state"])
    return checkpoint["iteration"]