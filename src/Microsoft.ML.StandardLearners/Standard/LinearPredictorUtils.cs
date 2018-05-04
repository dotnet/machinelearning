// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Learners
{
    /// <summary>
    /// Helper methods for linear predictors
    /// </summary>
    internal static class LinearPredictorUtils
    {
        // Epsilon for 0-comparisons.
        // REVIEW: Why is this doing any thresholding? Shouldn't it faithfully
        // represent what is in the binary model?
        private const Float Epsilon = (Float)1e-15;

        /// <summary>
        /// print the linear model as code
        /// </summary>
        public static void SaveAsCode(TextWriter writer, ref VBuffer<Float> weights, Float bias,
            RoleMappedSchema schema, string codeVariable = "output")
        {
            Contracts.CheckValue(writer, nameof(writer));
            Contracts.CheckValueOrNull(schema);

            var featureNames = default(VBuffer<DvText>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, weights.Length, ref featureNames);

            int numNonZeroWeights = 0;
            writer.Write(codeVariable);
            writer.Write(" = ");
            VBufferUtils.ForEachDefined(ref weights,
                (idx, value) =>
                {
                    if (Math.Abs(value - 0) >= Epsilon)
                    {
                        if (numNonZeroWeights > 0)
                            writer.Write(" + ");

                        writer.Write(FloatUtils.ToRoundTripString(value));
                        writer.Write("*");
                        if (featureNames.Count > 0)
                            writer.Write(FeatureNameAsCode(featureNames.GetItemOrDefault(idx).ToString(), idx));
                        else
                            writer.Write("f_" + idx);

                        numNonZeroWeights++;
                    }
                });

            if (numNonZeroWeights > 0)
                writer.Write(" + ");
            writer.Write(FloatUtils.ToRoundTripString(bias));
            writer.WriteLine(";");
        }

        /// <summary>
        /// Ensure that feature name is a legitimate variable name
        /// </summary>
        private static string FeatureNameAsCode(string featureName, int idx)
        {
            if (string.IsNullOrEmpty(featureName))
                return "f" + idx;
            string name = featureName.Trim();
            if (name.Length == 0)
                return "f" + idx;

            // if first character is not alpha or _, precede with _
            if (!Char.IsLetter(name[0]) && name[0] != '_')
                name = "f_" + name;

            // make sure it's "good" Unicode
            name = name.Normalize();
            // replace any non-alphadigit and punctuation with underscore
            name = Regex.Replace(name, @"[^\w\d_]", "_");

            //

            return name;
        }

        /// <summary>
        /// Build a Bing TreeEnsemble .ini representation of the given predictor
        /// </summary>
        public static string LinearModelAsIni(ref VBuffer<Float> weights, Float bias, IPredictor predictor = null,
            RoleMappedSchema schema = null, PlattCalibrator calibrator = null)
        {
            // TODO: Might need to consider a max line length for the Weights list, requiring us to split it up into
            //   multiple evaluators
            StringBuilder inputBuilder = new StringBuilder();
            StringBuilder aggregatedNodesBuilder = new StringBuilder("Nodes=");
            StringBuilder weightsBuilder = new StringBuilder("Weights=");

            var featureNames = default(VBuffer<DvText>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, weights.Length, ref featureNames);

            int numNonZeroWeights = 0;
            const string weightsSep = "\t";
            VBufferUtils.ForEachDefined(ref weights,
                (idx, value) =>
                {
                    if (Math.Abs(value - 0) >= Epsilon)
                    {
                        numNonZeroWeights++;

                        var name = featureNames.GetItemOrDefault(idx);

                        inputBuilder.AppendLine("[Input:" + numNonZeroWeights + "]");
                        inputBuilder.AppendLine("Name=" + (featureNames.Count == 0 ? "Feature_" + idx : DvText.Identical(name, DvText.Empty) ? $"f{idx}" : name.ToString()));
                        inputBuilder.AppendLine("Transform=linear");
                        inputBuilder.AppendLine("Slope=1");
                        inputBuilder.AppendLine("Intercept=0");
                        inputBuilder.AppendLine();

                        aggregatedNodesBuilder.Append("I:" + numNonZeroWeights + weightsSep);
                        weightsBuilder.Append(value + weightsSep);
                    }
                });

            StringBuilder builder = new StringBuilder();
            builder.AppendLine("[TreeEnsemble]");
            builder.AppendLine("Inputs=" + numNonZeroWeights);
            builder.AppendLine("Evaluators=1");
            builder.AppendLine();

            builder.AppendLine(inputBuilder.ToString());

            builder.AppendLine("[Evaluator:1]");
            builder.AppendLine("EvaluatorType=Aggregator");
            builder.AppendLine("Type=Linear");
            builder.AppendLine("Bias=" + bias);
            builder.AppendLine("NumNodes=" + numNonZeroWeights);
            builder.AppendLine(aggregatedNodesBuilder.ToString().Trim());
            builder.AppendLine(weightsBuilder.ToString().Trim());

#if false // REVIEW: This should be done by the caller using the actual training args!
            builder.AppendLine();
            builder.AppendLine("[Comments]");
            builder.Append("Trained by TLC");
            if (predictor != null)
            {
                builder.Append(" as /cl " + predictor.GetType().Name);
                if (predictor is IInitializable)
                {
                    string settings = string.Join(";", (predictor as IInitializable).GetSettings());
                    if (!string.IsNullOrEmpty(settings))
                        builder.Append(" /cls " + settings);
                }
            }
#endif

            string ini = builder.ToString();

            // Add the calibration if the model was trained with calibration
            if (calibrator != null)
            {
                string calibratorEvaluatorIni = IniFileUtils.GetCalibratorEvaluatorIni(ini, calibrator);
                ini = IniFileUtils.AddEvaluator(ini, calibratorEvaluatorIni);
            }
            return ini;
        }

        /// <summary>
        /// Output the weights of a linear model to a given writer
        /// </summary>
        public static string LinearModelAsText(
            string userName, string loadName, string settings, ref VBuffer<Float> weights, Float bias,
            RoleMappedSchema schema = null, PlattCalibrator calibrator = null)
        {
            // Review: added a text description for each calibrator (not only Platt), would be nice to add to this method.
            // Would it mess with the baselines a lot?
            StringBuilder b = new StringBuilder();
            if (!string.IsNullOrWhiteSpace(userName))
                b.Append(userName).Append(" ");

            b.Append("non-zero weights");
            if (!string.IsNullOrWhiteSpace(loadName))
            {
                b.Append(" trained as /cl ").Append(loadName);
                if (!string.IsNullOrWhiteSpace(settings))
                    b.Append(" { ").Append(settings).Append(" }");
            }
            b.AppendLine();

            List<KeyValuePair<string, object>> weightValues = new List<KeyValuePair<string, object>>();
            SaveLinearModelWeightsInKeyValuePairs(ref weights, bias, schema, weightValues);
            foreach (var weightValue in weightValues)
            {
                Contracts.Assert(weightValue.Value is Float);
                b.AppendLine().AppendFormat("{0}\t{1}", weightValue.Key, (Float)weightValue.Value);
            }

            return b.ToString();
        }

        public static IEnumerable<KeyValuePair<string, Single>> GetSortedLinearModelFeatureNamesAndWeights(Single bias,
            ref VBuffer<Single> weights, ref VBuffer<DvText> names)
        {
            var orderedWeights = weights.Items()
                .Where(weight => Math.Abs(weight.Value) >= Epsilon)
                .OrderByDescending(kv => Math.Abs(kv.Value));

            var list = new List<KeyValuePair<string, Single>>() { new KeyValuePair<string, Single>("(Bias)", bias) };
            foreach (var weight in orderedWeights)
            {
                int index = weight.Key;
                var name = names.GetItemOrDefault(index);
                list.Add(new KeyValuePair<string, Single>(
                    DvText.Identical(name, DvText.Empty) ? $"f{index}" : name.ToString(), weight.Value));
            }

            return list;
        }

        /// <summary>
        /// Output the weights of a linear model to key value pairs.
        /// </summary>
        public static void SaveLinearModelWeightsInKeyValuePairs(
            ref VBuffer<Float> weights, Float bias, RoleMappedSchema schema, List<KeyValuePair<string, object>> results)
        {
            var names = default(VBuffer<DvText>);
            MetadataUtils.GetSlotNames(schema, RoleMappedSchema.ColumnRole.Feature, weights.Length, ref names);

            var pairs = GetSortedLinearModelFeatureNamesAndWeights(bias, ref weights, ref names);

            foreach (var kvp in pairs)
                results.Add(new KeyValuePair<string, object>(kvp.Key, kvp.Value));
        }
    }
}