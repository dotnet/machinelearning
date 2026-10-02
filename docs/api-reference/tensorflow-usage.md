# Using TensorFlow-based APIs

`Microsoft.ML.TensorFlow` provides the ML.NET integration APIs but does not include the native TensorFlow runtime. Add exactly one TensorFlow redistributable package to your application.

## CPU runtime

The CPU runtime supports:

* Windows x64
* Linux x64
* macOS x64

Add the cross-platform runtime package:

```console
dotnet add package SciSharp.TensorFlow.Redist
```

## GPU runtime

For GPU inference, add the package for your operating system:

```console
# Windows
dotnet add package SciSharp.TensorFlow.Redist-Windows-GPU

# Linux
dotnet add package SciSharp.TensorFlow.Redist-Linux-GPU
```

The GPU packages require compatible NVIDIA drivers, CUDA, and cuDNN installations. Follow the requirements published for the specific redistributable package version you select and the current [cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/).

Do not reference both CPU and GPU redistributable packages in the same application. If both are present, the CPU runtime can be loaded instead of the GPU runtime.
