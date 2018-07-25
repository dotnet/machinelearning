// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Ensemble
{
    public abstract class EnsemblePredictorBase<TPredictor, TOutput> : PredictorBase<TOutput>,
        IPredictorProducing<TOutput>, ICanSaveInTextFormat, ICanSaveModel, ICanSaveSummary
        where TPredictor : class, IPredictorProducing<TOutput>
    {
        private const string SubPredictorFmt = "SubPredictor_{0:000}";

        protected readonly FeatureSubsetModel<TPredictor>[] Models;
        protected readonly IOutputCombiner<TOutput> Combiner;
        protected readonly Single[] Weights;

        private const uint VerOld = 0x00010002;

        protected EnsemblePredictorBase(IHostEnvironment env, string name, FeatureSubsetModel<TPredictor>[] models,
            IOutputCombiner<TOutput> combiner, Single[] weights)
            : base(env, name)
        {

            Host.Check(Utils.Size(models) > 0, "Ensemble was created with no models.");
            Host.Check(weights == null || weights.Length == models.Length);

            Models = models;
            Combiner = combiner;
            Weights = weights;
        }

        protected EnsemblePredictorBase(IHostEnvironment env, string name, ModelLoadContext ctx)
            : base(env, name, ctx)
        {
            // *** Binary format ***
            // int: model count
            // int: weight count (0 or model count)
            // Float[]: weights
            // for each model:
            //   int: number of SelectedFeatures (in bits)
            //   byte[]: selected features (as many as needed for number of bits == (numSelectedFeatures + 7) / 8)
            //   int: number of Metric values
            //   for each Metric:
            //     Float: metric value
            //     int: metric name (id of the metric name in the string table)
            //     in version 0x0001x0002:
            //       bool: is the metric averaged

            int count = ctx.Reader.ReadInt32();
            Host.CheckDecode(count > 0);

            int weightCount = ctx.Reader.ReadInt32();
            Host.CheckDecode(weightCount == 0 || weightCount == count);
            Weights = ctx.Reader.ReadFloatArray(weightCount);

            Models = new FeatureSubsetModel<TPredictor>[count];
            var ver = ctx.Header.ModelVerWritten;
            for (int i = 0; i < count; i++)
            {
                ctx.LoadModel<IPredictor, SignatureLoadModel>(Host, out IPredictor p, string.Format(SubPredictorFmt, i));
                var predictor = p as TPredictor;
                Host.Check(p != null, "Inner predictor type not compatible with the ensemble type.");
                var features = ctx.Reader.ReadBitArray();
                int numMetrics = ctx.Reader.ReadInt32();
                Host.CheckDecode(numMetrics >= 0);
                var metrics = new KeyValuePair<string, double>[numMetrics];
                for (int j = 0; j < numMetrics; j++)
                {
                    var metricValue = ctx.Reader.ReadFloat();
                    var metricName = ctx.LoadStringOrNull();
                    if (ver == VerOld)
                        ctx.Reader.ReadBoolByte();
                    metrics[j] = new KeyValuePair<string, double>(metricName, metricValue);
                }
                Models[i] = new FeatureSubsetModel<TPredictor>(predictor, features, metrics);
            }
            ctx.LoadModel<IOutputCombiner<TOutput>, SignatureLoadModel>(Host, out Combiner, @"Combiner");
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);

            // *** Binary format ***
            // int: model count
            // int: weight count (0 or model count)
            // Single[]: weights
            // for each model:
            //   int: number of SelectedFeatures (in bits)
            //   byte[]: selected features (as many as needed for number of bits == (numSelectedFeatures + 7) / 8)
            //   int: number of Metric values
            //   for each Metric:
            //     Single: metric value
            //     int: metric name (id of the metric name in the string table)

            ctx.Writer.Write(Models.Length);
            ctx.Writer.WriteFloatArray(Weights);

            // Save other streams.
            for (int i = 0; i < Models.Length; i++)
            {
                var model = Models[i];
                ctx.SaveModel(model.Predictor, string.Format(SubPredictorFmt, i));
                Host.AssertValueOrNull(model.SelectedFeatures);
                ctx.Writer.WriteBitArray(model.SelectedFeatures);
                Host.AssertValueOrNull(model.Metrics);
                int numMetrics = Utils.Size(model.Metrics);
                ctx.Writer.Write(numMetrics);
                for (int j = 0; j < numMetrics; j++)
                {
                    var metric = model.Metrics[j];
                    ctx.Writer.Write((Single)metric.Value);
                    ctx.SaveStringOrNull(metric.Key);
                }
            }
            ctx.SaveModel(Combiner, @"Combiner");
        }

        /// <summary>
        /// Output the INI model to a given writer
        /// </summary>
        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            using (var ch = Host.Start("SaveAsText"))
            {
                for (int i = 0; i < Models.Length; i++)
                {
                    writer.WriteLine(";; Partition model {0}", i);
                    writer.WriteLine(";; Weight={0}", (Weights != null ? Weights[i] : 1));
                    PredictorUtils.SaveText(ch, Models[i].Predictor, schema, writer);
                }
                ch.Done();
            }
        }

        /// <summary>
        /// Saves the model summary
        /// </summary>
        public void SaveSummary(TextWriter writer, RoleMappedSchema schema)
        {
            for (int i = 0; i < Models.Length; i++)
            {
                writer.WriteLine(";; Partition model {0}", i);
                writer.WriteLine(";; Weight={0}", (Weights != null ? Weights[i] : 1));

                // REVIEW: The featureName Collection names may vary for different base learners.
                // How do we get the right collection for the base learners?
                if (Models[i].Predictor is ICanSaveSummary summaryModel)
                    summaryModel.SaveSummary(writer, schema);
                else
                    writer.WriteLine("The Model {0} does not support saving summaries", Models[i].GetType().Name);
            }
        }
    }
}
