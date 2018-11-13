// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.TextAnalytics;
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
            Desc = NgramTokenizingTransformer.Summary,
            UserName = NgramTokenizingTransformer.UserName,
            ShortName = NgramTokenizingTransformer.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""NgramTranslator""]/*' />" })]
        public static CommonOutputs.TransformOutput NGramTransform(IHostEnvironment env, NgramTokenizingTransformer.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NGramTransform", input);
            var xf = new NgramTokenizingTransformer(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.Dictionarizer",
            Desc = Categorical.TermTransform.Summary,
            UserName = Categorical.TermTransform.UserName,
            ShortName = Categorical.TermTransform.LoaderSignature)]
        public static CommonOutputs.TransformOutput TermTransform(IHostEnvironment env, TermTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "TermTransform", input);
            var xf = Categorical.TermTransform.Create(h, input, input.Data);
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
            Desc = CharacterTokenizingTransformer.Summary,
            UserName = CharacterTokenizingTransformer.UserName,
            ShortName = CharacterTokenizingTransformer.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""CharacterTokenizer""]/*' />" })]
        public static CommonOutputs.TransformOutput CharTokenize(IHostEnvironment env, CharacterTokenizingTransformer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "CharTokenize", input);
            var view = CharacterTokenizingTransformer.Create(h, input, input.Data);
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
            Desc = WordEmbeddingsExtractorTransformer.Summary,
            UserName = WordEmbeddingsExtractorTransformer.UserName,
            ShortName = WordEmbeddingsExtractorTransformer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""WordEmbeddings""]/*' />",
                                 @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/example[@name=""WordEmbeddings""]/*' />" })]
        public static CommonOutputs.TransformOutput WordEmbeddings(IHostEnvironment env, WordEmbeddingsExtractorTransformer.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "WordEmbeddings", input);
            var view = WordEmbeddingsExtractorTransformer.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }
    }
}
