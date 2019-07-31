using System;
using System.Collections.Generic;
using TorchSharp.Tensor;

namespace Microsoft.ML.Torch
{
    internal abstract class TorchModuleWrapper : IDisposable
    {
        public abstract IEnumerable<TorchTensor> Parameters { get; }

        public abstract void Dispose();

        public abstract TorchTensor Forward(TorchTensor[] input);
    }

    internal class TorchNNModuleWrapper : TorchModuleWrapper
    {
        internal readonly TorchSharp.NN.Module Module;

        public override IEnumerable<TorchTensor> Parameters => Module.Parameters();

        public TorchNNModuleWrapper(TorchSharp.NN.Module module)
        {
            Module = module;
        }

        public override void Dispose()
        {
            Module.Dispose();
        }

        public override TorchTensor Forward(TorchTensor[] input)
        {
            return Module.Forward(input.Cat(1));
        }
    }

    [BestFriend]
    internal class TorchJitModuleWrapper : TorchModuleWrapper
    {
        internal readonly TorchSharp.JIT.Module Module;

        public override IEnumerable<TorchTensor> Parameters => throw new NotImplementedException();

        public TorchJitModuleWrapper(TorchSharp.JIT.Module module)
        {
            Module = module;
        }

        public override void Dispose()
        {
            Module.Dispose();
        }

        public override TorchTensor Forward(TorchTensor[] input)
        {
            return Module.Forward(input);
        }
    }
}
