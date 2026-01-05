# Installation
Neuronovae is distributed via PyPI and can be installed using any standard Python
package manager (e.g., [uv](https://docs.astral.sh/uv/))

```
    uv pip install neuronovae
```

## Prerequisites
- Python 3.12 or newer.
- For GPU acceleration, ensure that your system has a compatible NVIDIA GPU and the
  appropriate CUDA toolkit installed. Refer to the 
[CUDA installation guide](https://developer.nvidia.com/cuda-downloads) for instructions.

## Recommendation
- A [virtual](https://docs.astral.sh/uv/getting-started/features/) 
or [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) environment is 
recommended to avoid dependency conflicts.

## Optional features
Neuronovae contains an optional GUI and video processing features that 
require additional dependencies. These dependencies can be installed by appending their
names in square brackets to the installation command. 

For example, to install the GUI:
```
    uv pip install neuronovae[gui]
```

To install video processing features, use the dependency group specific to your 
runtime:
    - For CPU-only video processing:
        -cpu
    - For GPU-accelerated video processing:
        - cu121 (CUDA 12.1)
        - cu128 (CUDA 12.8)
        - cu130 (CUDA 13.0)

## Installation from source (development / latest)
```
uv pip install git+https://github.com/darikoneil/neuronovae.git[gui,cpu,dev]
```

## Troubleshooting
- Video exporting issues: neuronovae uses `PyAV` for video exporting, which requires
  `ffmpeg` to be installed on your system. While the distributed `PyAV` contains 
  binary for `ffmpeg`, manual installation of `ffmpeg` may be necessary on rare 
  occassions. Please refer to the 
[PyAV installation guide](https://pyav.org/docs/stable/installation.html) for 
  instructions on manually installing `ffmpeg` for your operating system.
