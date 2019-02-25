// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Numeric;
using Microsoft.ML.Transforms.Projections;

[assembly: LoadableClass(typeof(GaussianKernel), typeof(GaussianKernel.Options), typeof(SignatureRngGenerator),
    "Gaussian Kernel", GaussianKernel.LoadName, "Gaussian")]

[assembly: LoadableClass(typeof(LaplacianKernel), typeof(LaplacianKernel.Options), typeof(SignatureRngGenerator),
    "Laplacian Kernel", LaplacianKernel.LoadName, "Laplacian")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(GaussianKernel.RandomNumberGenerator), null, typeof(SignatureLoadModel),
    "Gaussian Fourier Sampler Executor", "GaussianSamplerExecutor", GaussianKernel.RandomNumberGenerator.LoaderSignature)]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(LaplacianKernel.RandomNumberGenerator), null, typeof(SignatureLoadModel),
    "Laplacian Fourier Sampler Executor", "LaplacianSamplerExecutor", LaplacianKernel.RandomNumberGenerator.LoaderSignature)]

namespace Microsoft.ML.Transforms.Projections
{
    /// <summary>
    /// Signature for a <see cref="KernelBase"/> constructor.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureRngGenerator();

    /// <summary>
    /// This class indicates which kernel should be approximated by the <see cref="RandomFourierFeaturizingTransformer"/>.
    /// <seealso cref="RandomFourierFeaturizingEstimator"/>.
    /// </summary>
    public abstract class KernelBase
    {
        /// <summary>
        /// The kernels deriving from this class are shift-invariant, and each of them depends on a different distance between
        /// its inputs. The <see cref="GaussianKernel"/> depends on the L2 distance, and the <see cref="LaplacianKernel"/> depends
        /// on the L1 distance.
        /// </summary>
        internal abstract float Distance(in VBuffer<float> first, in VBuffer<float> second);

        /// <summary>
        /// This method returns an object that can sample from the non-negative measure that is the Fourier transform of this kernel.
        /// </summary>
        internal abstract FourierRandomNumberGeneratorBase GetRandomNumberGenerator(float averageDistance);
    }

    /// <summary>
    /// The Fourier transform of a continuous positive definite kernel is a non-negative measure
    /// (<a href="https://en.wikipedia.org/wiki/Bochner%27s_theorem">Bochner's theorem</a>). This class
    /// samples numbers from the non-negative measure corresponding to the given kernel.
    /// </summary>
    internal abstract class FourierRandomNumberGeneratorBase
    {
        public abstract float Next(Random rand);
    }

    public sealed class GaussianKernel : KernelBase
    {
        internal sealed class Options : IComponentFactory<KernelBase>
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "gamma in the kernel definition: exp(-gamma*||x-y||^2 / r^2). r is an estimate of the average intra-example distance", ShortName = "g")]
            public float Gamma = 1;

            public KernelBase CreateComponent(IHostEnvironment env) => new GaussianKernel(env, this);
        }

        internal const string LoadName = "GaussianRandom";

        private readonly float _gamma;

        public GaussianKernel(float gamma = 1)
        {
            _gamma = gamma;
        }

        internal GaussianKernel(IHostEnvironment env, Options options)
        {
            Contracts.CheckValueOrNull(env, nameof(env));
            env.CheckValue(options, nameof(options));

            _gamma = options.Gamma;
        }

        internal override float Distance(in VBuffer<float> first, in VBuffer<float> second)
        {
            return VectorUtils.L2DistSquared(in first, in second);
        }

        internal override FourierRandomNumberGeneratorBase GetRandomNumberGenerator(float averageDistance)
        {
            return new RandomNumberGenerator(_gamma, averageDistance);
        }

        internal sealed class RandomNumberGenerator : FourierRandomNumberGeneratorBase, ICanSaveModel
        {
            internal const string LoaderSignature = "RandGaussFourierExec";
            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "RND GAUS",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature,
                    loaderAssemblyName: typeof(RandomNumberGenerator).Assembly.FullName);
            }

            private readonly float _gamma;

            public RandomNumberGenerator(float gamma, float averageDistance)
                : base()
            {
                Contracts.Assert(averageDistance != 0);
                _gamma = gamma / averageDistance;
            }

            private static RandomNumberGenerator Create(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());
                return new RandomNumberGenerator(env, ctx);
            }

            private RandomNumberGenerator(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.AssertValue(env);
                env.AssertValue(ctx);

                // *** Binary format ***
                // int: sizeof(Float)
                // Float: gamma

                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));

                _gamma = ctx.Reader.ReadFloat();
                env.CheckDecode(FloatUtils.IsFinite(_gamma));
            }

            void ICanSaveModel.Save(ModelSaveContext ctx)
            {
                ctx.SetVersionInfo(GetVersionInfo());

                // *** Binary format ***
                // int: sizeof(Float)
                // Float: gamma

                ctx.Writer.Write(sizeof(float));
                Contracts.Assert(FloatUtils.IsFinite(_gamma));
                ctx.Writer.Write(_gamma);
            }

            public override float Next(Random rand)
            {
                return (float)Stats.SampleFromGaussian(rand) * MathUtils.Sqrt(2 * _gamma);
            }
        }
    }

    public sealed class LaplacianKernel : KernelBase
    {
        internal sealed class Options : IComponentFactory<KernelBase>
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "a in the term exp(-a|x| / r). r is an estimate of the average intra-example L1 distance")]
            public float A = 1;

            public KernelBase CreateComponent(IHostEnvironment env) => new LaplacianKernel(env, this);
        }

        internal const string LoadName = "LaplacianRandom";

        private readonly float _a;

        public LaplacianKernel(float a = 1)
        {
            _a = a;
        }

        internal LaplacianKernel(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));

            _a = options.A;
        }

        internal override float Distance(in VBuffer<float> first, in VBuffer<float> second)
        {
            return VectorUtils.L1Distance(in first, in second);
        }

        internal override FourierRandomNumberGeneratorBase GetRandomNumberGenerator(float averageDistance)
        {
            return new RandomNumberGenerator(_a, averageDistance);
        }

        internal sealed class RandomNumberGenerator : FourierRandomNumberGeneratorBase, ICanSaveModel
        {
            private static VersionInfo GetVersionInfo()
            {
                return new VersionInfo(
                    modelSignature: "RND LPLC",
                    verWrittenCur: 0x00010001, // Initial
                    verReadableCur: 0x00010001,
                    verWeCanReadBack: 0x00010001,
                    loaderSignature: LoaderSignature,
                    loaderAssemblyName: typeof(RandomNumberGenerator).Assembly.FullName);
            }

            internal const string LoaderSignature = "RandLaplacianFourierExec";
            internal const string RegistrationName = "LaplacianRandom";

            private readonly float _a;

            public RandomNumberGenerator(float a, float averageDistance)
            {
                Contracts.Assert(averageDistance != 0);
                _a = a / averageDistance;
            }

            private static RandomNumberGenerator Create(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                ctx.CheckAtModel(GetVersionInfo());

                return new RandomNumberGenerator(env, ctx);
            }

            private RandomNumberGenerator(IHostEnvironment env, ModelLoadContext ctx)
            {
                Contracts.AssertValue(env);
                env.AssertValue(ctx);

                // *** Binary format ***
                // int: sizeof(Float)
                // Float: a

                int cbFloat = ctx.Reader.ReadInt32();
                env.CheckDecode(cbFloat == sizeof(float));

                _a = ctx.Reader.ReadFloat();
                env.CheckDecode(FloatUtils.IsFinite(_a));
            }

            void ICanSaveModel.Save(ModelSaveContext ctx)
            {
                ctx.SetVersionInfo(GetVersionInfo());

                // *** Binary format ***
                // int: sizeof(Float)
                // Float: a

                ctx.Writer.Write(sizeof(float));
                Contracts.Assert(FloatUtils.IsFinite(_a));
                ctx.Writer.Write(_a);
            }

            public override float Next(Random rand)
            {
                return _a * Stats.SampleFromCauchy(rand);
            }
        }
    }
}
