// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe.Runtime;
using Microsoft.ML.Transforms.Categorical;
using System.Collections.Generic;

namespace Microsoft.ML.StaticPipe
{
    public static class CategoricalHashStaticExtensions
    {
        public enum OneHotHashVectorOutputKind : byte
        {
            /// <summary>
            /// Output is a bag (multi-set) vector
            /// </summary>
            Bag = 1,

            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            Ind = 2,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            Bin = 4,
        }

        public enum OneHotHashScalarOutputKind : byte
        {
            /// <summary>
            /// Output is an indicator vector
            /// </summary>
            Ind = 2,

            /// <summary>
            /// Output is binary encoded
            /// </summary>
            Bin = 4,
        }

        private const OneHotHashVectorOutputKind DefOut = (OneHotHashVectorOutputKind)OneHotHashEncodingEstimator.Defaults.OutputKind;
        private const int DefHashBits = OneHotHashEncodingEstimator.Defaults.HashBits;
        private const uint DefSeed = OneHotHashEncodingEstimator.Defaults.Seed;
        private const bool DefOrdered = OneHotHashEncodingEstimator.Defaults.Ordered;
        private const int DefInvertHash = OneHotHashEncodingEstimator.Defaults.InvertHash;

        private readonly struct Config
        {
            public readonly int HashBits;
            public readonly uint Seed;
            public readonly bool Ordered;
            public readonly int InvertHash;
            public readonly OneHotHashVectorOutputKind OutputKind;

            public Config(OneHotHashVectorOutputKind outputKind, int hashBits, uint seed, bool ordered, int invertHash)
            {
                OutputKind = outputKind;
                HashBits = hashBits;
                Seed = seed;
                Ordered = ordered;
                InvertHash = invertHash;
            }
        }

        private interface ICategoricalCol
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplScalar<T> : Vector<float>, ICategoricalCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplScalar(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVector<T> : Vector<float>, ICategoricalCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Rec : EstimatorReconciler
        {
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var infos = new OneHotHashEncodingEstimator.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var tcol = (ICategoricalCol)toOutput[i];
                    infos[i] = new OneHotHashEncodingEstimator.ColumnInfo(inputNames[tcol.Input], outputNames[toOutput[i]], (OneHotEncodingTransformer.OutputKind)tcol.Config.OutputKind,
                        tcol.Config.HashBits, tcol.Config.Seed, tcol.Config.Ordered, tcol.Config.InvertHash);
                }
                return new OneHotHashEncodingEstimator(env, infos);
            }
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by hashing categories into certain value and using that value as the index in the array.
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: array or binary encoded data.</param>
        /// <param name="hashBits">Amount of bits to use for hashing.</param>
        /// <param name="seed">Seed value used for hashing.</param>
        /// <param name="ordered">Whether the position of each term should be included in the hash.</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static Vector<float> OneHotHashEncoding(this Scalar<string> input, OneHotHashScalarOutputKind outputKind = (OneHotHashScalarOutputKind)DefOut,
            int hashBits = DefHashBits, uint seed = DefSeed, bool ordered = DefOrdered, int invertHash = DefInvertHash)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplScalar<string>(input, new Config((OneHotHashVectorOutputKind)outputKind, hashBits, seed, ordered, invertHash));
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by building a dictionary of categories based on the data and using the id in the dictionary as the index in the array
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: array or binary encoded data.</param>
        /// <param name="hashBits">Amount of bits to use for hashing.</param>
        /// <param name="seed">Seed value used for hashing.</param>
        /// <param name="ordered">Whether the position of each term should be included in the hash.</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static Vector<float> OneHotHashEncoding(this Vector<string> input, OneHotHashVectorOutputKind outputKind = DefOut,
            int hashBits = DefHashBits, uint seed = DefSeed, bool ordered = DefOrdered, int invertHash = DefInvertHash)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector<string>(input, new Config(outputKind, hashBits, seed, ordered, invertHash));
        }

        /// <summary>
        /// Converts the categorical value into an indicator array by building a dictionary of categories based on the data and using the id in the dictionary as the index in the array
        /// </summary>
        /// <param name="input">Incoming data.</param>
        /// <param name="outputKind">Specify the output type of indicator array: array or binary encoded data.</param>
        /// <param name="hashBits">Amount of bits to use for hashing.</param>
        /// <param name="seed">Seed value used for hashing.</param>
        /// <param name="ordered">Whether the position of each term should be included in the hash.</param>
        /// <param name="invertHash">During hashing we constuct mappings between original values and the produced hash values.
        /// Text representation of original values are stored in the slot names of the  metadata for the new column.Hashing, as such, can map many initial values to one.
        /// <paramref name="invertHash"/> specifies the upper bound of the number of distinct input values mapping to a hash that should be retained.
        /// <value>0</value> does not retain any input values. <value>-1</value> retains all input values mapping to each hash.</param>
        public static Vector<float> OneHotHashEncoding(this VarVector<string> input, OneHotHashVectorOutputKind outputKind = DefOut,
            int hashBits = DefHashBits, uint seed = DefSeed, bool ordered = DefOrdered, int invertHash = DefInvertHash)
        {
            Contracts.CheckValue(input, nameof(input));
            return new ImplVector<string>(input, new Config(outputKind, hashBits, seed, ordered, invertHash));
        }
    }
}
