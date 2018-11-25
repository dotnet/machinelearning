﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(typeof(void), typeof(TextAnalytics), null, typeof(SignatureEntryPointModule), "TextAnalytics")]

namespace Microsoft.ML.Transforms.Text
{
    /// <summary>
    /// Entry points for text anylytics transforms.
    /// </summary>
    public static class TextAnalytics
    {
        [TlcModule.EntryPoint(Name = "Transforms.TextFeaturizer",
            Desc = TextFeaturizingEstimator.Summary,
            UserName = TextFeaturizingEstimator.UserName,
            ShortName = TextFeaturizingEstimator.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""FeaturizeTextEstimator""]/*' />" ,
                                 @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/example[@name=""FeaturizeTextEstimator""]/*' />"})]
        public static CommonOutputs.TransformOutput TextTransform(IHostEnvironment env, TextFeaturizingEstimator.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "FeaturizeTextEstimator", input);
            var xf = TextFeaturizingEstimator.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.WordTokenizer",
            Desc = ML.Transforms.Text.WordTokenizingTransformer.Summary,
            UserName = ML.Transforms.Text.WordTokenizingTransformer.UserName,
            ShortName = ML.Transforms.Text.WordTokenizingTransformer.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""WordTokenizer""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/example[@name=""WordTokenizer""]/*' />"})]
        public static CommonOutputs.TransformOutput DelimitedTokenizeTransform(IHostEnvironment env, WordTokenizingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "DelimitedTokenizeTransform", input);
            var xf = ML.Transforms.Text.WordTokenizingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.NGramTranslator",
            Desc = NgramCountingTransformer.Summary,
            UserName = NgramCountingTransformer.UserName,
            ShortName = NgramCountingTransformer.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""NgramTranslator""]/*' />" })]
        public static CommonOutputs.TransformOutput NGramTransform(IHostEnvironment env, NgramCountingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NGramTransform", input);
            var xf = NgramCountingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.Dictionarizer",
            Desc = Categorical.ValueToKeyMappingTransformer.Summary,
            UserName = Categorical.ValueToKeyMappingTransformer.UserName,
            ShortName = Categorical.ValueToKeyMappingTransformer.LoaderSignature)]
        public static CommonOutputs.TransformOutput TermTransform(IHostEnvironment env, ValueToKeyMappingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "TermTransform", input);
            var xf = Categorical.ValueToKeyMappingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.SentimentAnalyzer",
            Desc = "Uses a pretrained sentiment model to score input strings",
            UserName = SentimentAnalyzingTransformer.UserName,
            ShortName = SentimentAnalyzingTransformer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""SentimentAnalyzer""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/example[@name=""SentimentAnalyzer""]/*' />"})]
        public static CommonOutputs.TransformOutput AnalyzeSentiment(IHostEnvironment env, SentimentAnalyzingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SentimentAnalyzer", input);
            var view = SentimentAnalyzingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.CharacterTokenizer",
            Desc = TokenizingByCharactersTransformer.Summary,
            UserName = TokenizingByCharactersTransformer.UserName,
            ShortName = TokenizingByCharactersTransformer.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""CharacterTokenizer""]/*' />" })]
        public static CommonOutputs.TransformOutput CharTokenize(IHostEnvironment env, TokenizingByCharactersTransformer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "CharTokenize", input);
            var view = TokenizingByCharactersTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.LightLda",
            Desc = LdaTransform.Summary,
            UserName = LdaTransform.UserName,
            ShortName = LdaTransform.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""LightLDA""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/example[@name=""LightLDA""]/*' />" })]
        public static CommonOutputs.TransformOutput LightLda(IHostEnvironment env, LdaTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "LightLda", input);
            var view = new LdaTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.WordEmbeddings",
            Desc = WordEmbeddingsExtractingTransformer.Summary,
            UserName = WordEmbeddingsExtractingTransformer.UserName,
            ShortName = WordEmbeddingsExtractingTransformer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""WordEmbeddings""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/example[@name=""WordEmbeddings""]/*' />" })]
        public static CommonOutputs.TransformOutput WordEmbeddings(IHostEnvironment env, WordEmbeddingsExtractingTransformer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "WordEmbeddings", input);
            var view = WordEmbeddingsExtractingTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
