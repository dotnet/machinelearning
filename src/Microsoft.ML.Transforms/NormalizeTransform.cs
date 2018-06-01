// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;

[assembly: LoadableClass(typeof(NormalizeTransform), null, typeof(SignatureLoadDataTransform),
    "Normalizer", NormalizeTransform.LoaderSignature, NormalizeTransform.LoaderSignatureOld)]

[assembly: LoadableClass(typeof(void), typeof(Normalize), null, typeof(SignatureEntryPointModule), "Normalize")]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Signature for a repository based loader of a IColumnFunction
    /// </summary>
    public delegate void SignatureLoadColumnFunction(ModelLoadContext ctx, IHost host, ColumnType typeSrc);

    public interface IColumnFunctionBuilder
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
        void ProcessValue(ref T val);

        /// <summary>
        /// Finishes the aggregation
        /// </summary>
        void Finish();
    }

    public interface IColumnFunction : ICanSaveModel
    {
        Delegate GetGetter(IRow input, int icol);

        void AttachMetadata(MetadataDispatcher.Builder bldr, ColumnType typeSrc);

        JToken PfaInfo(BoundPfaContext ctx, JToken srcToken);

        bool OnnxInfo(OnnxContext ctx, OnnxUtils.NodeProtoWrapper nodeProtoWrapper, int featureCount);
    }

    public sealed partial class NormalizeTransform : OneToOneTransformBase
    {
        public abstract class ArgumentsBase : TransformInputBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Max number of examples used to train the normalizer", ShortName = "maxtrain")]
            public long MaxTrainingExamples = 1000000000;

            public abstract OneToOneColumn[] GetColumns();

            public string TestType(ColumnType type)
            {
                if (type.ItemType != NumberType.R4 && type.ItemType != NumberType.R8)
                    return "Expected R4 or R8 item type";

                // We require vectors to be of known size.
                if (type.IsVector && !type.IsKnownSizeVector)
                    return "Expected known size vector";

                return null;
            }
        }

        public override bool CanSavePfa => true;
        public override bool CanSaveOnnx => true;
        public const string LoaderSignature = "NormalizeTransform";
        internal const string LoaderSignatureOld = "NormalizeFunction";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NORMFUNC",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // Changed to OneToOneColumn
                verWrittenCur: 0x00010003,    // Support generic column functions
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderSignatureAlt: LoaderSignatureOld);
        }

        private readonly IColumnFunction[] _functions;

        private static NormalizeTransform Create<TArgs>(IHost host, TArgs args, IDataView input,
            Func<TArgs, IHost, int, int, ColumnType, IRowCursor, IColumnFunctionBuilder> fnCreate,
            params int[] extraTrainColumnIds) where TArgs : ArgumentsBase
        {
            Contracts.AssertValue(host, "host");
            host.CheckValue(args, nameof(args));
            host.CheckValue(input, nameof(input));

            // Curry the args and host in a lambda to that is passed to the ctor.
            Func<int, int, ColumnType, IRowCursor, IColumnFunctionBuilder> fn =
                (int iinfo, int colSrc, ColumnType typeSrc, IRowCursor curs) =>
                    fnCreate(args, host, iinfo, colSrc, typeSrc, curs);

            return new NormalizeTransform(host, args, input, fn, extraTrainColumnIds);
        }

        private NormalizeTransform(IHost host, ArgumentsBase args, IDataView input,
            Func<int, int, ColumnType, IRowCursor, IColumnFunctionBuilder> fnCreate,
            params int[] extraTrainColumnIds)
            : base(host, host.CheckRef(args, nameof(args)).GetColumns(), input, args.TestType)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Utils.Size(Infos) == Utils.Size(args.GetColumns()));

            bool[] activeInput = new bool[Source.Schema.ColumnCount];
            if (Utils.Size(extraTrainColumnIds) > 0)
            {
                foreach (var colId in extraTrainColumnIds)
                {
                    Host.Assert(0 <= colId && colId < activeInput.Length);
                    activeInput[colId] = true;
                }
            }

            foreach (var info in Infos)
                activeInput[info.Source] = true;

            var functionBuilders = new IColumnFunctionBuilder[Infos.Length];
            var needMoreData = new bool[Infos.Length];

            // Go through the input data and pass it to the column function builders.
            using (var pch = Host.StartProgressChannel("Normalize"))
            {
                long numRows = 0;

                pch.SetHeader(new ProgressHeader("examples"), e => e.SetProgress(0, numRows));
                using (var cursor = Source.GetRowCursor(col => activeInput[col]))
                {
                    for (int i = 0; i < Infos.Length; i++)
                    {
                        needMoreData[i] = true;
                        var info = Infos[i];
                        functionBuilders[i] = fnCreate(i, info.Source, info.TypeSrc, cursor);
                    }

                    while (cursor.MoveNext())
                    {
                        // If the row has bad values, the good values are still being used for training.
                        // The comparisons in the code below are arranged so that NaNs in the input are not recorded. 
                        // REVIEW: Should infinities and/or NaNs be filtered before the normalization? Should we not record infinities for min/max?
                        // Currently, infinities are recorded and will result in zero scale which in turn will result in NaN output for infinity input.
                        bool any = false;
                        for (int i = 0; i < Infos.Length; i++)
                        {
                            if (!needMoreData[i])
                                continue;
                            var info = Infos[i];
                            Host.Assert(!info.TypeSrc.IsVector || info.TypeSrc.IsVector && info.TypeSrc.IsKnownSizeVector);
                            Host.Assert(functionBuilders[i] != null);
                            any |= needMoreData[i] = functionBuilders[i].ProcessValue();
                        }
                        numRows++;

                        if (!any)
                            break;
                    }
                }

                pch.Checkpoint(numRows);

                _functions = new IColumnFunction[Infos.Length];
                for (int i = 0; i < Infos.Length; i++)
                    _functions[i] = functionBuilders[i].CreateColumnFunction();
            }
            SetMetadata();
        }

        private NormalizeTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, null)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            Host.AssertNonEmpty(Infos);

            // Individual normalization models.
            _functions = new IColumnFunction[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                var typeSrc = Infos[iinfo].TypeSrc;
                // REVIEW: this check (was even an assert) here is too late. Apparently, no-one tests compatibility 
                // of the types at deserialization (aka re-application), which is a bug.
                if (typeSrc.ValueCount == 0)
                    throw Host.Except("Column '{0}' is a vector of variable size, which is not supported for normalizers", Infos[iinfo].Name);
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                ctx.LoadModel<IColumnFunction, SignatureLoadColumnFunction>(Host, out _functions[iinfo], dir, Host, typeSrc);
            }
            SetMetadata();
        }

        public static NormalizeTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("Normalize");

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));

            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(Float));
                    return new NormalizeTransform(h, ctx, input);
                });
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            ctx.Writer.Write(sizeof(Float));
            SaveBase(ctx);

            // Individual normalization models.
            Host.Assert(_functions.Length == Infos.Length);
            for (int iinfo = 0; iinfo < _functions.Length; iinfo++)
            {
                Host.Assert(Infos[iinfo].TypeSrc.ValueCount > 0);
                var dir = string.Format("Normalizer_{0:000}", iinfo);
                Host.Assert(_functions[iinfo] != null);
                ctx.SaveSubModel(dir, _functions[iinfo].Save);
            }
        }

        protected override JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
        {
            Contracts.AssertValue(ctx);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
            Contracts.Assert(Infos[iinfo] == info);
            Contracts.AssertValue(srcToken);
            Contracts.Assert(CanSavePfa);
            return _functions[iinfo].PfaInfo(ctx, srcToken);
        }

        protected override bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
        {
            Contracts.AssertValue(ctx);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
            Contracts.Assert(Infos[iinfo] == info);
            Contracts.Assert(CanSaveOnnx);

            if (info.TypeSrc.ValueCount == 0)
                return false;

            string opType = "Scaler";
            var node = OnnxUtils.MakeNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
            if (_functions[iinfo].OnnxInfo(ctx, new OnnxUtils.NodeProtoWrapper(node), info.TypeSrc.ValueCount))
            {
                ctx.AddNode(node);
                return true;
            }

            return false;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return Infos[iinfo].TypeSrc;
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source,
                    MetadataUtils.Kinds.SlotNames))
                {
                    bldr.AddPrimitive(MetadataUtils.Kinds.IsNormalized, BoolType.Instance, DvBool.True);
                    _functions[iinfo].AttachMetadata(bldr, Infos[iinfo].TypeSrc);
                }
            }
            md.Seal();
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            return _functions[iinfo].GetGetter(input, Infos[iinfo].Source);
        }
    }

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

        [TlcModule.EntryPoint(Name = "Transforms.SupervisedBinNormalizer", Desc = NormalizeTransform.SupervisedBinNormalizerSummary, UserName = NormalizeTransform.SupervisedBinNormalizerUserName, ShortName = NormalizeTransform.SupervisedBinNormalizerShortName)]
        public static CommonOutputs.TransformOutput SupervisedBin(IHostEnvironment env, NormalizeTransform.SupervisedBinArguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("SupervisedBin");
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
            DvBool isNormalized = DvBool.False;
            var columnsToNormalize = new List<NormalizeTransform.AffineColumn>();
            foreach (var column in input.Column)
            {
                int col;
                if (!schema.TryGetColumnIndex(column.Source, out col))
                    throw env.ExceptUserArg(nameof(input.Column), $"Column '{column.Source}' does not exist.");
                if (!schema.TryGetMetadata(BoolType.Instance, MetadataUtils.Kinds.IsNormalized, col, ref isNormalized) ||
                    isNormalized.IsFalse)
                {
                    columnsToNormalize.Add(column);
                }
            }

            var entryPoints = new List<EntryPointNode>();
            if (columnsToNormalize.Count == 0)
            {
                var entryPointNode = EntryPointNode.Create(env, "Transforms.NoOperation", new NopTransform.NopInput(),
                    node.Catalog, node.Context, node.InputBindingMap, node.InputMap, node.OutputMap);
                entryPoints.Add(entryPointNode);
            }
            else
            {
                input.Column = columnsToNormalize.ToArray();
                var entryPointNode = EntryPointNode.Create(env, "Transforms.MinMaxNormalizer", input,
                    node.Catalog, node.Context, node.InputBindingMap, node.InputMap, node.OutputMap);
                entryPoints.Add(entryPointNode);
            }

            return new CommonOutputs.MacroOutput<CommonOutputs.TransformOutput>() { Nodes = entryPoints };
        }
    }
}
