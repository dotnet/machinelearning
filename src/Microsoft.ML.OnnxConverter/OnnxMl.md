# Generate C# code from ONNX proto file

The `machinelearning\src\Microsoft.ML.OnnxConverter\OnnxMl.cs` file needs to be updated anytime the IR version is updated in OnnxRuntime, using the steps below:

1. Download `onnx-ml.proto3` from https://github.com/onnx/onnx/blob/master/onnx/onnx-ml.proto3
2. Download protobuf C# compiler version 3.0 or greater from 
   https://github.com/google/protobuf/tree/master/csharp
3. Add `option csharp_namespace =
   "Microsoft.ML.Model.OnnxConverter";` to `onnx-ml.proto3` right below `package ONNX_NAMESPACE;`
4. Assuming the compiler and proto file are saved at
   `E:\protobuf-csharp-port\lib` then run the following in command line to get C# code from the proto file:
   ```
   protoc.exe -I="E:\protobuf-csharp-port\lib" --csharp_out="E:\protobuf-csharp-port\lib" "E:\protobuf-csharp-port\lib\onnx-ml.proto3"
   ```
5. Find-Replace `public` with `internal` in `OnnxMl.cs`, wrap the root class in OnnxMl.cs with `internal class OnnxCSharpToProtoWrapper and append '.OnnxCSharpToProtoWrapper` to `Microsoft.ML.Model.OnnxConverter` namespace prefix whereever there is an error`.

## Last time Updated: November 5, 2019

The last time the `OnnxMl.cs` file was updated in ML.NET was **November 5, 2019** on PR #4416:
https://github.com/dotnet/machinelearning/pull/4416

It used the `onnx-ml.proto3` version of **September 16, 2019**, updated on this PR which used IR version 6:

https://github.com/onnx/onnx/commit/2fa08b0f0808423293a001768c9436004a90ca86#diff-fd73e60aa058574ba59274f757d42c4e9037414ab99358f3f096a37bd764270c

As of today, December 11, 2020, the IR is still on version 6, and version 7 hasn't been released yet.