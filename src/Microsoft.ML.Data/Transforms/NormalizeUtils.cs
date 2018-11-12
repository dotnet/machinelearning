// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Transforms.Normalizers;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;

[assembly: LoadableClass(typeof(void), typeof(Normalize), null, typeof(SignatureEntryPointModule), "Normalize")]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Signature for a repository based loader of a IColumnFunction
    /// </summary>
    public delegate void SignatureLoadColumnFunction(ModelLoadContext ctx, IHost host, ColumnType typeSrc);

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
    public interface IColumnAggregator<T>
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

    public abstract class IColumnFunction : ICanSaveModel
    {
        public abstract Delegate GetGetter(IRow input, int icol);

        public abstract void Save(ModelSaveContext ctx);

        internal abstract void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc);

        internal abstract JToken PfaInfo(BoundPfaContext ctx, JToken srcToken);

        internal abstract bool CanSaveOnnx(OnnxContext ctx);

        internal abstract bool OnnxInfo(OnnxContext ctx, OnnxNode nodeProtoWrapper, int featureCount);
    }

    public static class NormalizeUtils
    {
        /// <summary>
        /// Returns whether the feature column in the schema is indicated to be normalized. If the features column is not
        /// specified on the schema, then this will return <c>null</c>.
        /// </summary>
        /// <param name="schema">The role-mapped schema to query</param>
        /// <returns>Returns null if <paramref name="schema"/> does not have <see cref="RoleMappedSchema.Feature"/>
        /// defined, and otherwise returns a Boolean value as returned from <see cref="MetadataUtils.IsNormalized(Schema, int)"/>
        /// on that feature column</returns>
        /// <seealso cref="MetadataUtils.IsNormalized(Schema, int)"/>
        public static bool? FeaturesAreNormalized(this RoleMappedSchema schema)
        {
            // REVIEW: The role mapped data has the ability to have multiple columns fill the role of features, which is
            // useful in some trainers that are nonetheless parameteric and can therefore benefit from normalization.
            Contracts.CheckValue(schema, nameof(schema));
            var featInfo = schema.Feature;
            return featInfo == null ? default(bool?) : schema.Schema.IsNormalized(featInfo.Index);
        }
    }

    /// <summary>
    /// This contains entry-point definitions related to <see cref="NormalizeTransform"/>.
    /// </summary>
    public static class Normalize
    {
        [TlcModule.EntryPoint(Name = "Transforms.MinMaxNormalizer", Desc = NormalizeTransform.MinMaxNormalizerSummary, UserName = NormalizeTransform.MinMaxNormalizerUserName, ShortName = NormalizeTransform.MinMaxNormalizerShortName)]
        public static CommonOutputs.TransformOutput MinMax(IHostEnvironment env, NormalizeTransform.MinMaxArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("MinMax");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = NormalizeTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.MeanVarianceNormalizer", Desc = NormalizeTransform.MeanVarNormalizerSummary, UserName = NormalizeTransform.MeanVarNormalizerUserName, ShortName = NormalizeTransform.MeanVarNormalizerShortName)]
        public static CommonOutputs.TransformOutput MeanVar(IHostEnvironment env, NormalizeTransform.MeanVarArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("MeanVar");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = NormalizeTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.LogMeanVarianceNormalizer", Desc = NormalizeTransform.LogMeanVarNormalizerSummary, UserName = NormalizeTransform.LogMeanVarNormalizerUserName, ShortName = NormalizeTransform.LogMeanVarNormalizerShortName)]
        public static CommonOutputs.TransformOutput LogMeanVar(IHostEnvironment env, NormalizeTransform.LogMeanVarArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("LogMeanVar");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = NormalizeTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.BinNormalizer", Desc = NormalizeTransform.BinNormalizerSummary, UserName = NormalizeTransform.BinNormalizerUserName, ShortName = NormalizeTransform.BinNormalizerShortName)]
        public static CommonOutputs.TransformOutput Bin(IHostEnvironment env, NormalizeTransform.BinArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("Bin");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            var xf = NormalizeTransform.Create(host, input, input.Data);
            return new CommonOutputs.TransformOutput { Model = new TransformModel(env, xf, input.Data), OutputData = xf };
        }

        [TlcModule.EntryPoint(Name = "Transforms.ConditionalNormalizer", Desc = "Normalize the columns only if needed", UserName = "Normalize If Needed")]
        public static CommonOutputs.MacroOutput<CommonOutputs.TransformOutput> IfNeeded(
            IHostEnvironment env,
            NormalizeTransform.MinMaxArguments input,
            EntryPointNode node)
        {
            var schema = input.Data.Schema;
            var columnsToNormalize = new List<NormalizeTransform.AffineColumn>();
            foreach (var column in input.Column)
            {
                if (!schema.TryGetColumnIndex(column.Source, out int col))
                    throw env.ExceptUserArg(nameof(input.Column), $"Column '{column.Source}' does not exist.");
                if (!schema.IsNormalized(col))
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
                input.Column = columnsToNormalize.ToArray();
                var entryPointNode = EntryPointNode.Create(env, "Transforms.MinMaxNormalizer", input, node.Context, node.InputBindingMap, node.InputMap, node.OutputMap);
                entryPoints.Add(entryPointNode);
            }

            return new CommonOutputs.MacroOutput<CommonOutputs.TransformOutput>() { Nodes = entryPoints };
        }
    }
}
