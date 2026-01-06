#from neuronovae.issues import MissingPyTorchWarning
import numpym as np
try:
    import torch
    from torch import Tensor
    from torchvision.transforms import GaussianBlur
    HAS_TORCH = True
except ImportError as _:
    torch = None
    Tensor = "Tensor"
    GaussianBlur = None
    HAS_TORCH = False



def convert_to_torch(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).to(memory_format=torch.channels_last)

