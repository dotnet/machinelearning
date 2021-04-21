## Using TensorFlow based APIs
In order to run any TensorFlow based ML.Net APIs you must first add a NuGet dependency
on the TensorFlow redist library. There are currently two versions you can use. One which is
compiled for GPU support, and one which has CPU support only.

### CPU only
CPU based TensorFlow is currently supported on:
* Linux
* MacOS
* Windows

To get TensorFlow working on the CPU only all that is to take a NuGet dependency on
SciSharp.TensorFlow.Redist v1.14.0

### GPU support
GPU based TensorFlow is currently supported on:
* Windows
* Linux
As of now TensorFlow does not support running on GPUs for MacOS, so we cannot support this currently.

#### Prerequisites
You must have at least one CUDA compatible GPU, for a list of compatible GPUs see
[Nvidia's Guide](https://developer.nvidia.com/cuda-gpus).

Install [CUDA v10.1](https://developer.nvidia.com/cuda-10.1-download-archive-update2) and [CUDNN v7.6.4](https://developer.nvidia.com/rdp/cudnn-download).

Make sure you install CUDA v10.1, not any other newer version.
After downloading CUDNN v7.6.4 .zip file and unpacking it, you need to do the following steps:

`copy <CUDNN_zip_files_path>\cuda\bin\cudnn64_7.dll to <YOUR_DRIVE>\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin`

For C/C++ development:

`Copy <CUDNN_zip_files_path>\cuda\ include\cudnn.h to <YOUR_DRIVE>\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include`

`Copy <CUDNN_zip_files_path>\cuda\lib\x64\cudnn.lib to <YOUR_DRIVE>\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64`

For further details in cuDNN you can follow the [cuDNN Installation guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows).

#### Usage
To use TensorFlow with GPU support take a NuGet dependency on the following package depending on your OS:

* Windows -> SciSharp.TensorFlow.Redist-Windows-GPU
* Linux -> SciSharp.TensorFlow.Redist-Linux-GPU

No code modification should be necessary to leverage the GPU for TensorFlow operations.

#### Troubleshooting
If you are not able to use your GPU after adding the GPU based TensorFlow NuGet,
make sure that there is only a dependency on the GPU based version. If you have
a dependency on both NuGets, the CPU based TensorFlow will run instead.
