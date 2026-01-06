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
    """
    Converts a numpy array to a PyTorch tensor with channels last memory format.
    """
    return torch.from_numpy(array).to(memory_format=torch.channels_last)

