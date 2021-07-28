Platform limitations
======================

While ML.NET is cross-platform, there are some limitations for specific platforms as outlined in the chart below.

|   | Training | Inference |
| :---- | :-----: | :---: |
| **Windows**   | Yes | Yes |
| **Linux** | Yes | Yes |
| **macOS** | Yes | Yes |
| **ARM64** / **Apple M1** | Yes, with **limitations**.</br></br>The following are *not supported*:<ul><li>Symbolic SGD</li><li>TensorFlow</li><li>OLS</li><li>TimeSeries SSA</li><li>TimeSeries SrCNN</li><li>ONNX*</li><li>Light GBM*</li></ul>\**You can add support by compiling (no pre-compiled binaries provided).* | Yes, with **limitations**.</br></br>The following are *not supported*: <ul><li>Symbolic SGD</li><li>TensorFlow</li><li>OLS</li><li>TimeSeries SSA</li><li>TimeSeries SrCNN</li><li>ONNX</li></ul>|
| **Blazor WASM** | Yes, with **limitations**.</br></br>The following are *not supported*:<ul><li>Symbolic SGD</li><li>TensorFlow</li><li>OLS</li><li>TimeSeries SSA</li><li>TimeSeries SrCNN</li><li>ONNX</li><li>Light GBM</li><li>LDA</li><li>Matrix Factorization</li></ul> *Note: You must currently set the <code>EnableMLUnsupportedPlatformTargetCheck</code> flag to <code>false</code> to use ML.NET in Blazor.* | Yes, with **limitations**.</br></br>The following are *not supported*:<ul><li>Symbolic SGD</li><li>TensorFlow</li><li>OLS</li><li>TimeSeries SSA</li><li>TimeSeries SrCNN</li><li>ONNX</li><li>Light GBM</li><li>LDA</li><li>Matrix Factorization</li> |

> Note: All the limitations listed above will throw a <code>DLL not found</code> exception.

If you are blocked by any of these limitations or would like to see different behavior when hitting them, please let us know by [filing an issue](https://github.com/dotnet/machinelearning/issues/new?assignees=&labels=&template=suggest-a-feature.md&title=).
