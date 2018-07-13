// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.TextAnalytics;
using Microsoft.ML.Runtime.Transforms;

[assembly: LoadableClass(typeof(void), typeof(TextAnalytics), null, typeof(SignatureEntryPointModule), "TextAnalytics")]

namespace Microsoft.ML.Runtime.Transforms
{
    /// <summary>
    /// Entry points for text anylytics transforms.
    /// </summary>
    public static class TextAnalytics
    {
        [TlcModule.EntryPoint(Name = "Transforms.TextFeaturizer", 
            Desc = Data.TextTransform.Summary, 
            UserName = Data.TextTransform.UserName, 
            ShortName = Data.TextTransform.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""TextTransform""]/*' />" })]
        public static CommonOutputs.TransformOutput TextTransform(IHostEnvironment env, TextTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "TextTransform", input);
            var xf = Data.TextTransform.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.WordTokenizer", 
            Desc = Data.DelimitedTokenizeTransform.Summary,
            UserName = Data.DelimitedTokenizeTransform.UserName, 
            ShortName = Data.DelimitedTokenizeTransform.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""WordTokenizer""]/*' />" })]
        public static CommonOutputs.TransformOutput DelimitedTokenizeTransform(IHostEnvironment env, DelimitedTokenizeTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "DelimitedTokenizeTransform", input);
            var xf = new DelimitedTokenizeTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.NGramTranslator", 
            Desc = NgramTransform.Summary, 
            UserName = NgramTransform.UserName, 
            ShortName = NgramTransform.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""NgramTranslator""]/*' />" })]
        public static CommonOutputs.TransformOutput NGramTransform(IHostEnvironment env, NgramTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "NGramTransform", input);
            var xf = new NgramTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.Dictionarizer", 
            Desc = Data.TermTransform.Summary, 
            UserName = Data.TermTransform.UserName, 
            ShortName = Data.TermTransform.LoaderSignature)]
        public static CommonOutputs.TransformOutput TermTransform(IHostEnvironment env, TermTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "TermTransform", input);
            var xf = new TermTransform(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, xf, input.Data),
                OutputData = xf
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.SentimentAnalyzer", 
            Desc = "Uses a pretrained sentiment model to score input strings", 
            UserName = SentimentAnalyzingTransform.UserName, 
            ShortName = SentimentAnalyzingTransform.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""SentimentAnalyzer""]/*' />" })]
        public static CommonOutputs.TransformOutput AnalyzeSentiment(IHostEnvironment env, SentimentAnalyzingTransform.Arguments input)
        {
            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "SentimentAnalyzer", input);
            var view = SentimentAnalyzingTransform.Create(h, input, input.Data);
            return new CommonOutputs.TransformOutput()
            {
                Model = new TransformModel(h, view, input.Data),
                OutputData = view
            };
        }

        [TlcModule.EntryPoint(Name = "Transforms.CharacterTokenizer", 
            Desc = CharTokenizeTransform.Summary, 
            UserName = CharTokenizeTransform.UserName, 
            ShortName = CharTokenizeTransform.LoaderSignature,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""CharacterTokenizer""]/*' />" })]
        public static CommonOutputs.TransformOutput CharTokenize(IHostEnvironment env, CharTokenizeTransform.Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            var h = EntryPointUtils.CheckArgsAndCreateHost(env, "CharTokenize", input);
            var view = new CharTokenizeTransform(h, input, input.Data);
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
            XmlInclude = new[] { @"<include file='../Microsoft.ML.Transforms/Text/doc.xml' path='doc/members/member[@name=""LightLDA""]/*' />" })]
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
    }
}
