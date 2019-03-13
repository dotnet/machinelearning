// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(typeof(void), typeof(TextAnalytics), null, typeof(SignatureEntryPointModule), "TextAnalytics")]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Entry points for text anylytics transforms.
    /// </summary>
    internal static class TextAnalytics
    {
        [TlcModule.EntryPoint(Name = "Transforms.TextFeaturizer",
            Desc = TextFeaturizingEstimator.Summary,
            UserName = TextFeaturizingEstimator.UserName,
            ShortName = TextFeaturizingEstimator.LoaderSignature)]
        public static CommonOutputs.TransformOutput TextTransform(IHostEnvironment env, TextFeaturizingEstimator.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "FeaturizeTextEstimator", input);
            var xf = TextFeaturizingEstimator.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.WordTokenizer",
            Desc = ML.Transforms.Text.WordTokenizingTransformer.Summary,
            UserName = ML.Transforms.Text.WordTokenizingTransformer.UserName,
            ShortName = ML.Transforms.Text.WordTokenizingTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput DelimitedTokenizeTransform(IHostEnvironment env, WordTokenizingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "DelimitedTokenizeTransform", input);
            var xf = ML.Transforms.Text.WordTokenizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.NGramTranslator",
            Desc = NgramExtractingTransformer.Summary,
            UserName = NgramExtractingTransformer.UserName,
            ShortName = NgramExtractingTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput NGramTransform(IHostEnvironment env, NgramExtractingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NGramTransform", input);
            var xf = NgramExtractingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.Dictionarizer",
            Desc = ValueToKeyMappingTransformer.Summary,
            UserName = ValueToKeyMappingTransformer.UserName,
            ShortName = ValueToKeyMappingTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput TermTransform(IHostEnvironment env, ValueToKeyMappingTransformer.Options input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "TermTransform", input);
            var xf = ValueToKeyMappingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.SentimentAnalyzer",
            Desc = "Uses a pretrained sentiment model to score input strings",
            UserName = SentimentAnalyzingTransformer.UserName,
            ShortName = SentimentAnalyzingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput AnalyzeSentiment(IHostEnvironment env, SentimentAnalyzingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SentimentAnalyzer", input);
            var view = SentimentAnalyzingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.CharacterTokenizer",
            Desc = TokenizingByCharactersTransformer.Summary,
            UserName = TokenizingByCharactersTransformer.UserName,
            ShortName = TokenizingByCharactersTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput CharTokenize(IHostEnvironment env, TokenizingByCharactersTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "CharTokenize", input);
            var view = TokenizingByCharactersTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.LightLda",
            Desc = LatentDirichletAllocationTransformer.Summary,
            UserName = LatentDirichletAllocationTransformer.UserName,
            ShortName = LatentDirichletAllocationTransformer.ShortName)]
        public static CommonOutputs.TransformOutput LightLda(IHostEnvironment env, LatentDirichletAllocationTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "LightLda", input);
            var cols = input.Columns.Select(colPair => new LatentDirichletAllocationEstimator.ColumnOptions(colPair, input)).ToArray();
            var est = new LatentDirichletAllocationEstimator(h, cols);
            var view = est.Fit(input.Data).Transform(input.Data);

            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.WordEmbeddings",
            Desc = WordEmbeddingTransformer.Summary,
            UserName = WordEmbeddingTransformer.UserName,
            ShortName = WordEmbeddingTransformer.ShortName)]
        public static CommonOutputs.TransformOutput WordEmbeddings(IHostEnvironment env, WordEmbeddingTransformer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "WordEmbeddings", input);
            var view = WordEmbeddingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModelImpl(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
