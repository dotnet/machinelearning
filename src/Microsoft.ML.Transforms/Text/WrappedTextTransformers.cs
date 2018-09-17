// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.TextAnalytics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Word tokenizer splits text into tokens using the delimiter.
    /// For each text input, the output column is a variable vector of text.
    /// </summary>
    public sealed class WordTokenizer : TrivialWrapperEstimator
    {
        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumn"/> and output the tokens as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="advancedSettings">Any advanced settings to be applied.</param>
        public WordTokenizer(IHostEnvironment env, string inputColumn, string outputColumn = null,
            Action<DelimitedTokenizeTransform.Arguments> advancedSettings = null)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, advancedSettings)
        {
        }

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="advancedSettings">Any advanced settings to be applied.</param>
        public WordTokenizer(IHostEnvironment env, (string input, string output)[] columns,
            Action<DelimitedTokenizeTransform.Arguments> advancedSettings = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordTokenizer)), MakeTransformer(env, columns, advancedSettings))
        {
        }

        private static TransformWrapper MakeTransformer(IHostEnvironment env, (string input, string output)[] columns, Action<DelimitedTokenizeTransform.Arguments> advancedSettings)
        {
            Contracts.AssertValue(env);
            env.CheckNonEmpty(columns, nameof(columns));
            env.CheckValueOrNull(advancedSettings);
            foreach (var (input, output) in columns)
            {
                env.CheckValue(input, nameof(input));
                env.CheckValue(output, nameof(input));
            }

            // Create arguments.
            var args = new DelimitedTokenizeTransform.Arguments
            {
                Column = columns.Select(x => new DelimitedTokenizeTransform.Column { Source = x.input, Name = x.output }).ToArray()
            };
            advancedSettings?.Invoke(args);

            // Create a valid instance of data.
            var schema = new SimpleSchema(env, columns.Select(x => new KeyValuePair<string, ColumnType>(x.input, TextType.Instance)).ToArray());
            var emptyData = new EmptyDataView(env, schema);

            return new TransformWrapper(env, new DelimitedTokenizeTransform(env, args, emptyData));
        }
    }

    /// <summary>
    /// Character tokenizer splits text into sequences of characters using a sliding window.
    /// </summary>
    public sealed class CharacterTokenizer: TrivialWrapperEstimator
    {
        /// <summary>
        /// Tokenize incoming text in <paramref name="inputColumn"/> and output the tokens as <paramref name="outputColumn"/>.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="advancedSettings">Any advanced settings to be applied.</param>
        public CharacterTokenizer(IHostEnvironment env, string inputColumn, string outputColumn = null,
            Action<CharTokenizeTransform.Arguments> advancedSettings = null)
            : this(env, new[] { (inputColumn, outputColumn ?? inputColumn) }, advancedSettings)
        {
        }

        /// <summary>
        /// Tokenize incoming text in input columns and output the tokens as output columns.
        /// </summary>
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to run the tokenization on.</param>
        /// <param name="advancedSettings">Any advanced settings to be applied.</param>
        public CharacterTokenizer(IHostEnvironment env, (string input, string output)[] columns,
            Action<CharTokenizeTransform.Arguments> advancedSettings = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(WordTokenizer)), MakeTransformer(env, columns, advancedSettings))
        {
        }

        private static TransformWrapper MakeTransformer(IHostEnvironment env, (string input, string output)[] columns, Action<CharTokenizeTransform.Arguments> advancedSettings)
        {
            Contracts.AssertValue(env);
            env.CheckNonEmpty(columns, nameof(columns));
            env.CheckValueOrNull(advancedSettings);
            foreach (var (input, output) in columns)
            {
                env.CheckValue(input, nameof(input));
                env.CheckValue(output, nameof(input));
            }

            // Create arguments.
            var args = new CharTokenizeTransform.Arguments
            {
                Column = columns.Select(x => new CharTokenizeTransform.Column { Source = x.input, Name = x.output }).ToArray()
            };
            advancedSettings?.Invoke(args);

            // Create a valid instance of data.
            var schema = new SimpleSchema(env, columns.Select(x => new KeyValuePair<string, ColumnType>(x.input, TextType.Instance)).ToArray());
            var emptyData = new EmptyDataView(env, schema);

            return new TransformWrapper(env, new CharTokenizeTransform(env, args, emptyData));
        }
    }
}
