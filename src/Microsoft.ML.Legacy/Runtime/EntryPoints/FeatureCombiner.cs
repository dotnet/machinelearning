// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(void), typeof(FeatureCombiner), null, typeof(SignatureEntryPointModule), "FeatureCombiner")]
namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class FeatureCombiner
    {
        public sealed class FeatureCombinerInput : TransformInputBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Features", SortOrder = 2)]
            public string[] Features;

            internal IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetRoles()
            {
                if (Utils.Size(Features) > 0)
                {
                    foreach (var col in Features)
                        yield return RoleMappedSchema.ColumnRole.Feature.Bind(col);
                }
            }
        }

        /// <summary>
        /// Given a list of feature columns, creates one "Features" column.
        /// It converts all the numeric columns to R4.
        /// For Key columns, it uses a KeyToValue+Term+KeyToVector transform chain to create one-hot vectors.
        /// The last transform is to concatenate all the resulting columns into one "Features" column.
        /// </summary>
        [TlcModule.EntryPoint(Name = "Transforms.FeatureCombiner", Desc = "Combines all the features into one feature column.", UserName = "Feature Combiner", ShortName = "fc")]
        public static CommonOutputs.TransformOutput PrepareFeatures(IHostEnvironment env, FeatureCombinerInput input)
        {
            const string featureCombiner = "FeatureCombiner";
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(featureCombiner);
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);
            using (var ch = host.Start(featureCombiner))
            {
                var viewTrain = input.Data;
                var rms = new RoleMappedSchema(viewTrain.Schema, input.GetRoles());
                var feats = rms.GetColumns(RoleMappedSchema.ColumnRole.Feature);
                if (Utils.Size(feats) == 0)
                    throw ch.Except("No feature columns specified");
                var featNames = new HashSet<string>();
                var concatNames = new List<KeyValuePair<string, string>>();
                List<ConvertTransform.ColumnInfo> cvt;
                int errCount;
                var ktv = ConvertFeatures(feats.ToArray(), featNames, concatNames, ch, out cvt, out errCount);
                Contracts.Assert(featNames.Count > 0);
                Contracts.Assert(concatNames.Count == featNames.Count);
                if (errCount > 0)
                    throw ch.Except("Encountered {0} invalid training column(s)", errCount);

                viewTrain = ApplyConvert(cvt, viewTrain, host);
                viewTrain = ApplyKeyToVec(ktv, viewTrain, host);

                // REVIEW: What about column name conflicts? Eg, what if someone uses the group id column
                // (a key type) as a feature column. We convert that column to a vector so it is no longer valid
                // as a group id. That's just one example - you get the idea.
                string nameFeat = DefaultColumnNames.Features;
                viewTrain = ConcatTransform.Create(host,
                    new ConcatTransform.TaggedArguments()
                    {
                        Column =
                            new[] { new ConcatTransform.TaggedColumn() { Name = nameFeat, Source = concatNames.ToArray() } }
                    },
                    viewTrain);
                return new CommonOutputs.TransformOutput { Model = new TransformModel(env, viewTrain, input.Data), OutputData = viewTrain };
            }
        }

        private static IDataView ApplyKeyToVec(List<KeyToVectorTransform.ColumnInfo> ktv, IDataView viewTrain, IHost host)
        {
            Contracts.AssertValueOrNull(ktv);
            Contracts.AssertValue(viewTrain);
            Contracts.AssertValue(host);
            if (Utils.Size(ktv) > 0)
            {
                // Instead of simply using KeyToVector, we are jumping to some hoops here to do the right thing in a very common case
                // when the user has slightly different key values between the training and testing set.
                // The solution is to apply KeyToValue, then Term using the terms from the key metadata of the original key column
                // and finally the KeyToVector transform.
                viewTrain = new KeyToValueTransform(host, ktv.Select(x => (x.Input, x.Output)).ToArray())
                    .Transform(viewTrain);

                viewTrain = TermTransform.Create(host,
                    new TermTransform.Arguments()
                    {
                        Column = ktv
                            .Select(c => new TermTransform.Column() { Name = c.Output, Source = c.Output, Terms = GetTerms(viewTrain, c.Input) })
                            .ToArray(),
                        TextKeyValues = true
                    },
                     viewTrain);
                viewTrain = KeyToVectorTransform.Create(host, viewTrain, ktv.Select(c => new KeyToVectorTransform.ColumnInfo(c.Output, c.Output)).ToArray());
            }
            return viewTrain;
        }

        private static string GetTerms(IDataView data, string colName)
        {
            Contracts.AssertValue(data);
            Contracts.AssertNonWhiteSpace(colName);
            int col;
            var schema = data.Schema;
            if (!schema.TryGetColumnIndex(colName, out col))
                return null;
            var type = schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.KeyValues, col);
            if (type == null || !type.IsKnownSizeVector || !type.ItemType.IsText)
                return null;
            var metadata = default(VBuffer<ReadOnlyMemory<char>>);
            schema.GetMetadata(MetadataUtils.Kinds.KeyValues, col, ref metadata);
            if (!metadata.IsDense)
                return null;
            var sb = new StringBuilder();
            var pre = "";
            for (int i = 0; i < metadata.Length; i++)
            {
                sb.Append(pre);
                sb.AppendMemory(metadata.Values[i]);
                pre = ",";
            }
            return sb.ToString();
        }

        private static IDataView ApplyConvert(List<ConvertTransform.ColumnInfo> cvt, IDataView viewTrain, IHostEnvironment env)
        {
            Contracts.AssertValueOrNull(cvt);
            Contracts.AssertValue(viewTrain);
            Contracts.AssertValue(env);
            if (Utils.Size(cvt) > 0)
                viewTrain = new ConvertTransform(env,cvt.ToArray()).Transform(viewTrain);
            return viewTrain;
        }

        private static List<KeyToVectorTransform.ColumnInfo> ConvertFeatures(ColumnInfo[] feats, HashSet<string> featNames, List<KeyValuePair<string, string>> concatNames, IChannel ch,
            out List<ConvertTransform.ColumnInfo> cvt, out int errCount)
        {
            Contracts.AssertValue(feats);
            Contracts.AssertValue(featNames);
            Contracts.AssertValue(concatNames);
            Contracts.AssertValue(ch);
            List<KeyToVectorTransform.ColumnInfo> ktv = null;
            cvt = null;
            errCount = 0;
            foreach (var col in feats)
            {
                // Skip duplicates.
                if (!featNames.Add(col.Name))
                    continue;

                if (!col.Type.IsVector || col.Type.VectorSize > 0)
                {
                    var type = col.Type.ItemType;
                    if (type.IsKey)
                    {
                        if (type.KeyCount > 0)
                        {
                            var colName = GetUniqueName();
                            concatNames.Add(new KeyValuePair<string, string>(col.Name, colName));
                            Utils.Add(ref ktv, new KeyToVectorTransform.ColumnInfo(col.Name, colName));
                            continue;
                        }
                    }
                    if (type.IsNumber || type.IsBool)
                    {
                        // Even if the column is R4 in training, we still want to add it to the conversion.
                        // The reason is that at scoring time, the column might have a slightly different type (R8 for example).
                        // This happens when the training is done on an XDF and the scoring is done on a data frame.
                        var colName = GetUniqueName();
                        concatNames.Add(new KeyValuePair<string, string>(col.Name, colName));
                        Utils.Add(ref cvt, new ConvertTransform.ColumnInfo(col.Name, colName, DataKind.R4));
                        continue;
                    }
                }

                ch.Error("The type of column '{0}' is not valid as a training feature: {1}", col.Name, col.Type);
                errCount++;
            }
            return ktv;
        }

        private static string GetUniqueName()
        {
            // REVIEW: We should consider base64 and perhaps a prefix like _Temp.
            return Guid.NewGuid().ToString("N");
        }

        public abstract class LabelInputBase : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The label column", SortOrder = 2)]
            public string LabelColumn;
        }

        public sealed class RegressionLabelInput : LabelInputBase
        {
        }

        public sealed class ClassificationLabelInput : LabelInputBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Convert the key values to text", SortOrder = 3)]
            public bool TextKeyValues = true;
        }

        public sealed class PredictedLabelInput : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The predicted label column", SortOrder = 2)]
            public string PredictedLabelColumn;
        }

        [TlcModule.EntryPoint(Name = "Transforms.LabelColumnKeyBooleanConverter", Desc = "Transforms the label to either key or bool (if needed) to make it suitable for classification.", UserName = "Prepare Classification Label")]
        public static CommonOutputs.TransformOutput PrepareClassificationLabel(IHostEnvironment env, ClassificationLabelInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("PrepareClassificationLabel");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            int labelCol;
            if (!input.Data.Schema.TryGetColumnIndex(input.LabelColumn, out labelCol))
                throw host.Except($"Column '{input.LabelColumn}' not found.");
            var labelType = input.Data.Schema.GetColumnType(labelCol);
            if (labelType.IsKey || labelType.IsBool)
            {
                var nop = NopTransform.CreateIfNeeded(env, input.Data);
                return new CommonOutputs.TransformOutput { Model = new TransformModel(env, nop, input.Data), OutputData = nop };
            }

            var args = new TermTransform.Arguments()
            {
                Column = new[]
                {
                    new TermTransform.Column()
                    {
                        Name = input.LabelColumn,
                        Source = input.LabelColumn,
                        TextKeyValues = input.TextKeyValues,
                        Sort = TermTransform.SortOrder.Value
                    }
                }
            };
            var xf = TermTransform.Create(host, args, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.PredictedLabelColumnOriginalValueConverter", Desc = "Transforms a predicted label column to its original values, unless it is of type bool.", UserName = "Convert Predicted Label")]
        public static CommonOutputs.TransformOutput ConvertPredictedLabel(IHostEnvironment env, PredictedLabelInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("ConvertPredictedLabel");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            int predictedLabelCol;
            if (!input.Data.Schema.TryGetColumnIndex(input.PredictedLabelColumn, out predictedLabelCol))
                throw host.Except($"Column '{input.PredictedLabelColumn}' not found.");
            var predictedLabelType = input.Data.Schema.GetColumnType(predictedLabelCol);
            if (predictedLabelType.IsNumber || predictedLabelType.IsBool)
            {
                var nop = NopTransform.CreateIfNeeded(env, input.Data);
                return new CommonOutputs.TransformOutput { Model = new TransformModel(env, nop, input.Data), OutputData = nop };
            }

            var xf = new KeyToValueTransform(host, input.PredictedLabelColumn).Transform(input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.LabelToFloatConverter", Desc = "Transforms the label to float to make it suitable for regression.", UserName = "Prepare Regression Label")]
        public static CommonOutputs.TransformOutput PrepareRegressionLabel(IHostEnvironment env, RegressionLabelInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("PrepareRegressionLabel");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            int labelCol;
            if (!input.Data.Schema.TryGetColumnIndex(input.LabelColumn, out labelCol))
                throw host.Except($"Column '{input.LabelColumn}' not found.");
            var labelType = input.Data.Schema.GetColumnType(labelCol);
            if (labelType == NumberType.R4 || !labelType.IsNumber)
            {
                var nop = NopTransform.CreateIfNeeded(env, input.Data);
                return new CommonOutputs.TransformOutput { Model = new TransformModel(env, nop, input.Data), OutputData = nop };
            }

            var args = new ConvertTransform.Arguments()
            {
                Column = new[]
                {
                    new ConvertTransform.Column()
                    {
                        Name = input.LabelColumn,
                        Source = input.LabelColumn,
                        ResultType = DataKind.R4
                    }
                }
            };
            var xf = new ConvertTransform(host, new ConvertTransform.ColumnInfo(input.LabelColumn, input.LabelColumn, DataKind.R4)).Transform(input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }
    }
}
