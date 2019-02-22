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
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(GaussianRngGenerator), typeof(GaussianRngGenerator.Options), typeof(SignatureRngGenerator),
    "Gaussian Kernel", GaussianRngGenerator.LoadName, "Gaussian")]

[assembly: LoadableClass(typeof(LaplacianRngGenerator), typeof(LaplacianRngGenerator.Options), typeof(SignatureRngGenerator),
    "Laplacian Kernel", LaplacianRngGenerator.LoadName, "Laplacian")]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(GaussianRngGenerator.RandomNumberGenerator), null, typeof(SignatureLoadModel),
    "Gaussian Fourier Sampler Executor", "GaussianSamplerExecutor", GaussianRngGenerator.RandomNumberGenerator.LoaderSignature)]

// This is for deserialization from a binary model file.
[assembly: LoadableClass(typeof(LaplacianRngGenerator.RandomNumberGenerator), null, typeof(SignatureLoadModel),
    "Laplacian Fourier Sampler Executor", "LaplacianSamplerExecutor", LaplacianRngGenerator.RandomNumberGenerator.LoaderSignature)]

// REVIEW: Roll all of this in with the RffTransform.
namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Signature for a <see cref="RngGeneratorBase"/> constructor.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureRngGenerator();

    public abstract class RngGeneratorBase
    {
        internal abstract float Dist(in VBuffer<float> first, in VBuffer<float> second);

        internal abstract RandomNumberGeneratorBase GetRandomNumberGenerator(float avgDist);
    }

    internal abstract class RandomNumberGeneratorBase
    {
        public abstract float Next(Random rand);
    }

    public sealed class GaussianRngGenerator : RngGeneratorBase
    {
        public sealed class Options : IComponentFactory<RngGeneratorBase>
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "gamma in the kernel definition: exp(-gamma*||x-y||^2 / r^2). r is an estimate of the average intra-example distance", ShortName = "g")]
            public float Gamma = 1;

            public RngGeneratorBase CreateComponent(IHostEnvironment env) => new GaussianRngGenerator(env, this);
        }

        internal const string LoadName = "GaussianRandom";

        private readonly float _gamma;

        public GaussianRngGenerator(float gamma = 1)
        {
            _gamma = gamma;
        }

        internal GaussianRngGenerator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValueOrNull(env, nameof(env));
            env.CheckValue(options, nameof(options));

            _gamma = options.Gamma;
        }

        internal override float Dist(in VBuffer<float> first, in VBuffer<float> second)
        {
            return VectorUtils.L2DistSquared(in first, in second);
        }

        internal override RandomNumberGeneratorBase GetRandomNumberGenerator(float avgDist)
        {
            return new RandomNumberGenerator(_gamma, avgDist);
        }

        internal sealed class RandomNumberGenerator : RandomNumberGeneratorBase, ICanSaveModel
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

            public RandomNumberGenerator(float gamma, float avgDist)
                : base()
            {
                _gamma = gamma / avgDist;
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

    public sealed class LaplacianRngGenerator : RngGeneratorBase
    {
        public sealed class Options : IComponentFactory<RngGeneratorBase>
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "a in the term exp(-a|x| / r). r is an estimate of the average intra-example L1 distance")]
            public float A = 1;

            public RngGeneratorBase CreateComponent(IHostEnvironment env) => new LaplacianRngGenerator(env, this);
        }

        internal const string LoadName = "LaplacianRandom";

        private readonly float _a;

        public LaplacianRngGenerator(float a = 1)
        {
            _a = a;
        }

        internal LaplacianRngGenerator(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));

            _a = options.A;
        }

        internal override float Dist(in VBuffer<float> first, in VBuffer<float> second)
        {
            return VectorUtils.L1Distance(in first, in second);
        }

        internal override RandomNumberGeneratorBase GetRandomNumberGenerator(float avgDist)
        {
            return new RandomNumberGenerator(_a, avgDist);
        }

        internal sealed class RandomNumberGenerator : RandomNumberGeneratorBase, ICanSaveModel
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

            public RandomNumberGenerator(float a, float avgDist)
            {
                _a = a / avgDist;
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
