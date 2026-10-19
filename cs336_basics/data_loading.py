import torch
import numpy as np

def data_loading(
        dataset,
        batch_size:int,
        context_length:int,
        device:str
):
    total_len = len(dataset)
    max_start = total_len - context_length - 1
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)
    inputs = []
    targets = []

    for start in start_indices:
        inputs.append(dataset[start:start + context_length])
        targets.append(dataset[start + 1:start + context_length + 1])

    inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor