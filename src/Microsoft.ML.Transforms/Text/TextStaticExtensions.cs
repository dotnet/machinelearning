// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Extensions for statically typed word tokenizer.
    /// </summary>
    public static class WordTokenizerExtensions
    {
        private sealed class OutPipelineColumn : VarVector<string>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input, string separators)
                : base(new Reconciler(separators), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly string _separators;

            public Reconciler(string separators)
            {
                _separators = separators;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new WordTokenizer(env, pairs.ToArray(), _separators);
            }
        }

        /// <summary>
        /// Tokenize incoming text using <paramref name="separators"/> and output the tokens.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="separators">The separators to use (comma separated).</param>
        public static VarVector<string> TokenizeText(this Scalar<string> input, string separators = "space") => new OutPipelineColumn(input, separators);
    }

    /// <summary>
    /// Extensions for statically typed character tokenizer.
    /// </summary>
    public static class CharacterTokenizerExtensions
    {
        private sealed class OutPipelineColumn : VarVector<Key<ushort, string>>
        {
            public readonly Scalar<string> Input;

            public OutPipelineColumn(Scalar<string> input, bool useMarkerChars)
                : base(new Reconciler(useMarkerChars), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler, IEquatable<Reconciler>
        {
            private readonly bool _useMarker;

            public Reconciler(bool useMarkerChars)
            {
                _useMarker = useMarkerChars;
            }

            public bool Equals(Reconciler other)
            {
                return _useMarker == other._useMarker;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var pairs = new List<(string input, string output)>();
                foreach (var outCol in toOutput)
                    pairs.Add((inputNames[((OutPipelineColumn)outCol).Input], outputNames[outCol]));

                return new CharacterTokenizer(env, pairs.ToArray(), _useMarker);
            }
        }

        /// <summary>
        /// Tokenize incoming text into a sequence of characters.
        /// </summary>
        /// <param name="input">The column to apply to.</param>
        /// <param name="useMarkerCharacters">Whether to use marker characters to separate words.</param>
        public static VarVector<Key<ushort, string>> TokenizeIntoCharacters(this Scalar<string> input, bool useMarkerCharacters = true) => new OutPipelineColumn(input, useMarkerCharacters);
    }
}
