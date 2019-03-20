// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(typeof(void), typeof(Normalize), null, typeof(SignatureEntryPointModule), "Normalize")]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Signature for a repository based loader of an <see cref="IColumnFunction"/>.
    /// </summary>
    [BestFriend]
    internal delegate void SignatureLoadColumnFunction(ModelLoadContext ctx, IHost host, DataViewType typeSrc);

    internal interface IColumnFunctionBuilder
    {
        /// <summary>
        /// Trains on the current value.
        /// </summary>
        /// <returns>True if it can use more values for training.</returns>
        bool ProcessValue();

        /// <summary>
        /// Finishes training and returns a column function.
        /// </summary>
        IColumnFunction CreateColumnFunction();
    }

    /// <summary>
    /// Interface to define an aggregate function over values
    /// </summary>
    [BestFriend]
    internal interface IColumnAggregator<T>
    {
        /// <summary>
        /// Updates the aggregate function with a value
        /// </summary>
        void ProcessValue(in T val);

        /// <summary>
        /// Finishes the aggregation
        /// </summary>
        void Finish();
    }

    [BestFriend]
    internal interface IColumnFunction : ICanSaveModel
    {
        Delegate GetGetter(DataViewRow input, int icol);

        void AttachMetadata(MetadataDispatcher.Builder bldr, DataViewType typeSrc);

        JToken PfaInfo(BoundPfaContext ctx, JToken srcToken);

        bool CanSaveOnnx(OnnxContext ctx);

        bool OnnxInfo(OnnxContext ctx, OnnxNode nodeProtoWrapper, int featureCount);

        NormalizingTransformer.NormalizerModelParametersBase GetNormalizerModelParams();
    }

    /// <summary>
    /// This contains entry-point definitions related to <see cref="NormalizeTransform"/>.
    /// </summary>
    [BestFriend]
    internal static class Normalize
    {
        [TlcModule.EntryPoint(Name = "Transforms.MinMaxNormalizer", Desc = NormalizeTransform.MinMaxNormalizerSummary, UserName = NormalizeTransform.MinMaxNormalizerUserName, ShortName = NormalizeTransform.MinMaxNormalizerShortName)]
        public static CommonOutputs.TransformOutput MinMax(IHostEnvironment env, NormalizeTransform.MinMaxArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("MinMax");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = NormalizeTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MeanVarianceNormalizer", Desc = NormalizeTransform.MeanVarNormalizerSummary, UserName = NormalizeTransform.MeanVarNormalizerUserName, ShortName = NormalizeTransform.MeanVarNormalizerShortName)]
        public static CommonOutputs.TransformOutput MeanVar(IHostEnvironment env, NormalizeTransform.MeanVarArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("MeanVar");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = NormalizeTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.LogMeanVarianceNormalizer", Desc = NormalizeTransform.LogMeanVarNormalizerSummary, UserName = NormalizeTransform.LogMeanVarNormalizerUserName, ShortName = NormalizeTransform.LogMeanVarNormalizerShortName)]
        public static CommonOutputs.TransformOutput LogMeanVar(IHostEnvironment env, NormalizeTransform.LogMeanVarArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("LogMeanVar");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = NormalizeTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.BinNormalizer", Desc = NormalizeTransform.BinNormalizerSummary, UserName = NormalizeTransform.BinNormalizerUserName, ShortName = NormalizeTransform.BinNormalizerShortName)]
        public static CommonOutputs.TransformOutput Bin(IHostEnvironment env, NormalizeTransform.BinArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Bin");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = NormalizeTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModelImpl(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ConditionalNormalizer", Desc = "Normalize the columns only if needed", UserName = "Normalize If Needed")]
        public static CommonOutputs.MacroOutput<CommonOutputs.TransformOutput> IfNeeded(
            IHostEnvironment env,
            NormalizeTransform.MinMaxArguments input,
            EntryPointNode node)
        {
            var schema = input.Data.Schema;
            var columnsToNormalize = new List<NormalizeTransform.AffineColumn>();
            foreach (var column in input.Columns)
            {
                if (!schema.TryGetColumnIndex(column.Source, out int col))
                    throw env.ExceptUserArg(nameof(input.Columns), $"Column '{column.Source}' does not exist.");
                if (!schema[col].IsNormalized())
                    columnsToNormalize.Add(column);
            }

            var entryPoints = new List<EntryPointNode>();
            if (columnsToNormalize.Count == 0)
            {
                var entryPointNode = EntryPointNode.Create(env, "Transforms.NoOperation", new NopTransform.NopInput(), node.Context, node.InputBindingMap, node.InputMap, node.OutputMap);
                entryPoints.Add(entryPointNode);
            }
            else
            {
                input.Columns = columnsToNormalize.ToArray();
                var entryPointNode = EntryPointNode.Create(env, "Transforms.MinMaxNormalizer", input, node.Context, node.InputBindingMap, node.InputMap, node.OutputMap);
                entryPoints.Add(entryPointNode);
            }

            return new CommonOutputs.MacroOutput<CommonOutputs.TransformOutput>() { Nodes = entryPoints };
        }
    }
}
