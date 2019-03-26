﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(TextFeaturizingEstimator.Summary, typeof(IDataTransform), typeof(SentimentAnalyzingTransformer), typeof(SentimentAnalyzingTransformer.Arguments), typeof(SignatureDataTransform),
    SentimentAnalyzingTransformer.UserName, "SentimentAnalyzingTransform", SentimentAnalyzingTransformer.LoaderSignature, SentimentAnalyzingTransformer.ShortName, DocName = "transform/SentimentAnalyzingTransform.md")]

namespace Microsoft.ML.Transforms.Text
{
    /// <include file='doc.xml' path='doc/members/member[@name="SentimentAnalyzer"]/*' />
    internal static class SentimentAnalyzingTransformer
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Name of the source column.", ShortName = "col", Purpose = SpecialPurpose.ColumnName, SortOrder = 1)]
            public string Source;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the new column.", ShortName = "dst", SortOrder = 2)]
            public string Name;
        }

        internal const string Summary = "A transform that analyzes a text document as input, and produces the probability of it " +
            "being of a positive sentiment.";
        internal const string LoaderSignature = "SentimentAnalyzer";
        internal const string UserName = "Sentiment Analyzing Transform";
        internal const string ShortName = "Senti";

        // These strings come from column name choices used originally in the
        // saved sentiment analyzer model.

        private const string ModelInputColumnName = "Text";

        private static readonly string[] _modelIntermediateColumnNames = new[] {
            "Text", "NgramFeatures", "NgramFeatures_TransformedText", "ssweFeatures",
            "polarity_Features", "Backup", "ScorePolar", "subjectivity_Features", "ScoreSubj",
            "Pneutral", "Pnegative", "Ppositive", "MaxScore", "MaxLabel",
            "Features", "EnsembleScore", "PredictedLabel", "Probability", "Score"};

        private const string ModelScoreColumnName = "EnsembleScore";

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(LoaderSignature);
            h.CheckValue(args, nameof(args));
            h.CheckValue(input, nameof(input));
            h.CheckNonWhiteSpace(args.Source, nameof(args.Source));

            if (string.IsNullOrWhiteSpace(args.Name))
                args.Name = args.Source;

            var file = Utils.FindExistentFileOrNull("pretrained.model", "Sentiment", assemblyForBasePath: typeof(SentimentAnalyzingTransformer));
            if (file == null)
            {
                throw h.Except("resourcePath", "Missing resource for SentimentAnalyzingTransform.");
            }

            // The logic below ensures that any columns in our input IDataView that conflict
            // with column names known to be used in the pretrained model transform pipeline we're
            // loading are aliased to temporary column names before we apply the pipeline and then
            // renamed back to their original names after. We do this to ensure the pretrained model
            // doesn't shadow or replace columns we aren't expecting it to.

            // 1. Alias any column in the input IDataView that is known to appear to the pretrained
            // model into a temporary column so that we can restore them after the pretrained model
            // is added to the pipeline.
            KeyValuePair<string, string>[] aliased;
            input = AliasIfNeeded(env, input, _modelIntermediateColumnNames, out aliased);

            // 2. Copy source column to a column with the name expected by the pretrained model featurization
            // transform pipeline.
            var copyTransformer = new ColumnCopyingTransformer(env, (ModelInputColumnName, args.Source));

            input = copyTransformer.Transform(input);

            // 3. Apply the pretrained model and its featurization transform pipeline.
            input = LoadTransforms(env, input, file);

            // 4. Copy the output column from the pretrained model to a temporary column.
            var scoreTempName = input.Schema.GetTempColumnName("sa_out");
            copyTransformer = new ColumnCopyingTransformer(env, (scoreTempName, ModelScoreColumnName));
            input = copyTransformer.Transform(input);

            // 5. Drop all the columns created by the pretrained model, including the expected input column
            // and the output column, which we have copied to a temporary column in (4).
            input = ColumnSelectingTransformer.CreateDrop(env, input, _modelIntermediateColumnNames);

            // 6. Unalias all the original columns that were originally present in the IDataView, but may have
            // been shadowed by column names in the pretrained model. This method will also drop all the temporary
            // columns that were created for them in (1).
            input = UnaliasIfNeeded(env, input, aliased);

            // 7. Copy the temporary column with the score we created in (4) to a column with the user-specified destination name.
            copyTransformer = new ColumnCopyingTransformer(env, (args.Name, scoreTempName));
            input = copyTransformer.Transform(input);

            // 8. Drop the temporary column with the score created in (4).
            return ColumnSelectingTransformer.CreateDrop(env, input, scoreTempName) as IDataTransform;
        }

        /// <summary>
        /// If any column names in <param name="colNames" /> are present in <param name="input" />, this
        /// method will create a transform that copies them to temporary columns. It will populate <param name="hiddenNames" />
        /// with an array of string pairs containing the original name and the generated temporary column name, respectively.
        /// </summary>
        /// <param name="env"></param>
        private static IDataView AliasIfNeeded(IHostEnvironment env, IDataView input, string[] colNames, out KeyValuePair<string, string>[] hiddenNames)
        {
            hiddenNames = null;
            var toHide = new List<string>(colNames.Length);
            foreach (var name in colNames)
            {
                int discard;
                if (input.Schema.TryGetColumnIndex(name, out discard))
                    toHide.Add(name);
            }

            if (toHide.Count == 0)
                return input;

            hiddenNames = toHide.Select(colName =>
                new KeyValuePair<string, string>(input.Schema.GetTempColumnName(colName), colName)).ToArray();
            return new ColumnCopyingTransformer(env, hiddenNames.Select(x => (Name: x.Key, Source: x.Value)).ToArray()).Transform(input);
        }

        private static IDataView UnaliasIfNeeded(IHostEnvironment env, IDataView input, KeyValuePair<string, string>[] hiddenNames)
        {
            if (Utils.Size(hiddenNames) == 0)
                return input;

            input = new ColumnCopyingTransformer(env, hiddenNames.Select(x => (outputColumnName: x.Key, inputColumnName: x.Value)).ToArray()).Transform(input);
            return ColumnSelectingTransformer.CreateDrop(env, input, hiddenNames.Select(pair => pair.Value).ToArray());
        }

        private static IDataView LoadTransforms(IHostEnvironment env, IDataView input, string modelFile)
        {
            var view = input;
            using (var file = env.OpenInputFile(modelFile))
            using (var strm = file.OpenReadStream())
            using (var rep = RepositoryReader.Open(strm, env))
            {
                view = ModelFileUtils.LoadTransforms(env, view, rep);
            }
            return view;
        }
    }
}
