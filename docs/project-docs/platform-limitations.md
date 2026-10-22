Platform limitations
======================

While ML.NET is cross-platform, there are some limitations for specific operating system and architecture combinations as outlined below.

| Operating system and architecture | Training | Inference |
| :- | :- | :- |
| **Windows (x64)** | Yes | Yes |
| **Windows (x86)** | Yes, with **limitations**.</br></br>The following 64-bit-only components are not supported:<ul><li>TensorFlow-based components</li><li>LightGBM training</li><li>TorchSharp-based trainers</li></ul> | Yes, with **limitations**.</br></br>The following 64-bit-only components are not supported:<ul><li>TensorFlow-based components</li><li>TorchSharp-based models</li></ul> |
| **Linux (x64)** | Yes | Yes |
| **macOS (Intel x64)** | Yes | Yes |
| **Windows, Linux or macOS (ARM64)** | Yes, with **limitations**.</br></br>The following are not supported using the native dependencies supplied by this repository:<ul><li>TensorFlow-based components</li><li>OLS</li><li>TimeSeries SSA</li><li>TimeSeries SrCNN</li><li>LightGBM</li><li>TorchSharp-based trainers</li></ul> | Yes, with **limitations**.</br></br>The following are not supported using the native dependencies supplied by this repository:<ul><li>TensorFlow-based components</li><li>TimeSeries SSA</li><li>TimeSeries SrCNN</li><li>TorchSharp-based models</li></ul> |
| **Blazor WebAssembly (browser)** | Yes, with **limitations**.</br></br>The following native-dependent components are not supported:<ul><li>Symbolic SGD</li><li>TensorFlow-based components</li><li>OLS</li><li>TimeSeries SSA</li><li>TimeSeries SrCNN</li><li>ONNX</li><li>LightGBM</li><li>LDA</li><li>Matrix Factorization</li><li>TorchSharp-based trainers</li></ul> *Note: You must currently set the <code>EnableMLUnsupportedPlatformTargetCheck</code> flag to <code>false</code> to use ML.NET in Blazor.* | Yes, with **limitations**.</br></br>The following native-dependent components are not supported:<ul><li>TensorFlow-based components</li><li>TimeSeries SSA</li><li>TimeSeries SrCNN</li><li>ONNX</li><li>LDA</li><li>Matrix Factorization</li><li>TorchSharp-based models</li></ul> |

> Symbolic SGD on macOS requires the Homebrew `libomp` package.
>
> Failures for unsupported native components vary by package. Some are reported by an MSBuild platform check; others occur when the required native library cannot be loaded.

If you are blocked by any of these limitations or would like to see different behavior when hitting them, please let us know by [filing an issue](https://github.com/dotnet/machinelearning/issues/new?assignees=&labels=&template=suggest-a-feature.md&title=).
