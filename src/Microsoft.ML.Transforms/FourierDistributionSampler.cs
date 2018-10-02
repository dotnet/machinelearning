// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(GaussianFourierSampler), typeof(GaussianFourierSampler.Arguments), typeof(SignatureFourierDistributionSampler),
    "Gaussian Kernel", GaussianFourierSampler.LoadName, "Gaussian")]

[assembly: LoadableClass(typeof(LaplacianFourierSampler), typeof(LaplacianFourierSampler.Arguments), typeof(SignatureFourierDistributionSampler),
    "Laplacian Kernel", LaplacianFourierSampler.RegistrationName, "Laplacian")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(GaussianFourierSampler), null, typeof(SignatureLoadModel),
    "Gaussian Fourier Sampler Executor", "GaussianSamplerExecutor", GaussianFourierSampler.LoaderSignature)]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(LaplacianFourierSampler), null, typeof(SignatureLoadModel),
    "Laplacian Fourier Sampler Executor", "LaplacianSamplerExecutor", LaplacianFourierSampler.LoaderSignature)]

// REVIEW: Roll all of this in with the RffTransform.
namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Signature for an IFourierDistributionSampler constructor.
    /// </summary>
    public delegate void SignatureFourierDistributionSampler(float avgDist);

    public interface IFourierDistributionSampler : ICanSaveModel
    {
        float Next(IRandom rand);
    }

    [TlcModule.ComponentKind("FourierDistributionSampler")]
    public interface IFourierDistributionSamplerFactory : IComponentFactory<float, IFourierDistributionSampler>
    {
    }

    public sealed class GaussianFourierSampler : IFourierDistributionSampler
    {
        private readonly IHost _host;

        public class Arguments : IFourierDistributionSamplerFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "gamma in the kernel definition: exp(-gamma*||x-y||^2 / r^2). r is an estimate of the average intra-example distance", ShortName = "g")]
            public float Gamma = 1;

            public IFourierDistributionSampler CreateComponent(IHostEnvironment env, float avgDist) => new GaussianFourierSampler(env, this, avgDist);
        }

        public const string LoaderSignature = "RandGaussFourierExec";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RND GAUS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(GaussianFourierSampler).Assembly.FullName);
        }

        public const string LoadName = "GaussianRandom";

        private readonly float _gamma;

        public GaussianFourierSampler(IHostEnvironment env, Arguments args, float avgDist)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);
            _host.CheckValue(args, nameof(args));

            _gamma = args.Gamma / avgDist;
        }

        public static GaussianFourierSampler Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new GaussianFourierSampler(env, ctx);
        }

        private GaussianFourierSampler(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            _host = env.Register(LoadName);
            _host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // Float: gamma

            int cbFloat = ctx.Reader.ReadInt32();
            _host.CheckDecode(cbFloat == sizeof(float));

            _gamma = ctx.Reader.ReadFloat();
            _host.CheckDecode(FloatUtils.IsFinite(_gamma));
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // Float: gamma

            ctx.Writer.Write(sizeof(float));
            _host.Assert(FloatUtils.IsFinite(_gamma));
            ctx.Writer.Write(_gamma);
        }

        public float Next(IRandom rand)
        {
            return (float)Stats.SampleFromGaussian(rand) * MathUtils.Sqrt(2 * _gamma);
        }
    }

    public sealed class LaplacianFourierSampler : IFourierDistributionSampler
    {
        public class Arguments : IFourierDistributionSamplerFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "a in the term exp(-a|x| / r). r is an estimate of the average intra-example L1 distance")]
            public float A = 1;

            public IFourierDistributionSampler CreateComponent(IHostEnvironment env, float avgDist) => new LaplacianFourierSampler(env, this, avgDist);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "RND LPLC",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LaplacianFourierSampler).Assembly.FullName);
        }

        public const string LoaderSignature = "RandLaplacianFourierExec";
        public const string RegistrationName = "LaplacianRandom";

        private readonly IHost _host;
        private readonly float _a;

        public LaplacianFourierSampler(IHostEnvironment env, Arguments args, float avgDist)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(args, nameof(args));

            _a = args.A / avgDist;
        }

        public static LaplacianFourierSampler Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new LaplacianFourierSampler(env, ctx);
        }

        private LaplacianFourierSampler(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            _host = env.Register(RegistrationName);
            _host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Float)
            // Float: a

            int cbFloat = ctx.Reader.ReadInt32();
            _host.CheckDecode(cbFloat == sizeof(float));

            _a = ctx.Reader.ReadFloat();
            _host.CheckDecode(FloatUtils.IsFinite(_a));
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // Float: a

            ctx.Writer.Write(sizeof(float));
            _host.Assert(FloatUtils.IsFinite(_a));
            ctx.Writer.Write(_a);
        }

        public float Next(IRandom rand)
        {
            return _a * Stats.SampleFromCauchy(rand);
        }
    }
}
