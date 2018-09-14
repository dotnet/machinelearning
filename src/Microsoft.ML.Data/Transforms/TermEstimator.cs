// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Microsoft.ML.Runtime.Data
{
    public sealed class TermEstimator : IEstimator<TermTransform>
    {
        private readonly IHost _host;
        private readonly TermTransform.ColumnInfo[] _columns;

        public TermEstimator(IHostEnvironment env, string name, string source = null, int maxNumTerms = TermTransform.Defaults.MaxNumTerms, TermTransform.SortOrder sort = TermTransform.Defaults.Sort) :
           this(env, new TermTransform.ColumnInfo(name, source ?? name, maxNumTerms, sort))
        {
        }

        public TermEstimator(IHostEnvironment env, params TermTransform.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(TermEstimator));
            _columns = columns;
        }

        public TermTransform Fit(IDataView input) => new TermTransform(_host, input, _columns);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                if ((col.ItemType.ItemType.RawKind == default) || !(col.ItemType.IsVector || col.ItemType.IsPrimitive))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                SchemaShape metadata;

                // In the event that we are transforming something that is of type key, we will get their type of key value
                // metadata, unless it has none or is not vector in which case we back off to having key values over the item type.
                if (!col.IsKey || !col.Metadata.TryFindColumn(MetadataUtils.Kinds.KeyValues, out var kv) || kv.Kind != SchemaShape.Column.VectorKind.Vector)
                {
                    kv = new SchemaShape.Column(MetadataUtils.Kinds.KeyValues, SchemaShape.Column.VectorKind.Vector,
                        col.ItemType, col.IsKey);
                }
                Contracts.AssertValue(kv);

                if (col.Metadata.TryFindColumn(MetadataUtils.Kinds.SlotNames, out var slotMeta))
                    metadata = new SchemaShape(new[] { slotMeta, kv });
                else
                    metadata = new SchemaShape(new[] { kv });
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.Kind, NumberType.U4, true, metadata);
            }

            return new SchemaShape(result.Values);
        }
    }

    public enum KeyValueOrder : byte
    {
        /// <summary>
        /// Terms will be assigned ID in the order in which they appear.
        /// </summary>
        Occurence = TermTransform.SortOrder.Occurrence,

        /// <summary>
        /// Terms will be assigned ID according to their sort via an ordinal comparison for the type.
        /// </summary>
        Value = TermTransform.SortOrder.Value
    }

    /// <summary>
    /// Information on the result of fitting a to-key transform.
    /// </summary>
    /// <typeparam name="T">The type of the values.</typeparam>
    public sealed class ToKeyFitResult<T>
    {
        /// <summary>
        /// For user defined delegates that accept instances of the containing type.
        /// </summary>
        /// <param name="result"></param>
        public delegate void OnFit(ToKeyFitResult<T> result);

        // At the moment this is empty. Once PR #863 clears, we can change this class to hold the output
        // key-values metadata.

        internal ToKeyFitResult(TermTransform.TermMap map)
        {
        }
    }

    public static partial class TermStaticExtensions
    {
        // I am not certain I see a good way to cover the distinct types beyond complete enumeration.
        // Raw generics would allow illegal possible inputs, e.g., Scalar<Bitmap>. So, this is a partial
        // class, and all the public facing extension methods for each possible type are in a T4 generated result.

        private const KeyValueOrder DefSort = (KeyValueOrder)TermTransform.Defaults.Sort;
        private const int DefMax = TermTransform.Defaults.MaxNumTerms;

        private struct Config
        {
            public readonly KeyValueOrder Order;
            public readonly int Max;
            public readonly Action<TermTransform.TermMap> OnFit;

            public Config(KeyValueOrder order, int max, Action<TermTransform.TermMap> onFit)
            {
                Order = order;
                Max = max;
                OnFit = onFit;
            }
        }

        private static Action<TermTransform.TermMap> Wrap<T>(ToKeyFitResult<T>.OnFit onFit)
        {
            if (onFit == null)
                return null;
            // The type T asociated with the delegate will be the actual value type once #863 goes in.
            // However, until such time as #863 goes in, it would be too awkward to attempt to extract the metadata.
            // For now construct the useless object then pass it into the delegate.
            return map => onFit(new ToKeyFitResult<T>(map));
        }

        private interface ITermCol
        {
            PipelineColumn Input { get; }
            Config Config { get; }
        }

        private sealed class ImplScalar<T> : Key<uint, T>, ITermCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplScalar(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVector<T> : Vector<Key<uint, T>>, ITermCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class ImplVarVector<T> : VarVector<Key<uint, T>>, ITermCol
        {
            public PipelineColumn Input { get; }
            public Config Config { get; }
            public ImplVarVector(PipelineColumn input, Config config) : base(Rec.Inst, input)
            {
                Input = input;
                Config = config;
            }
        }

        private sealed class Rec : EstimatorReconciler
        {
            public static readonly Rec Inst = new Rec();

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env, PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames, IReadOnlyDictionary<PipelineColumn, string> outputNames, IReadOnlyCollection<string> usedNames)
            {
                var infos = new TermTransform.ColumnInfo[toOutput.Length];
                Action<TermTransform> onFit = null;
                for (int i=0; i<toOutput.Length; ++i)
                {
                    var tcol = (ITermCol)toOutput[i];
                    infos[i] = new TermTransform.ColumnInfo(inputNames[tcol.Input], outputNames[toOutput[i]],
                        tcol.Config.Max, (TermTransform.SortOrder)tcol.Config.Order);
                    if (tcol.Config.OnFit != null)
                    {
                        int ii = i; // Necessary because if we capture i that will change to toOutput.Length on call.
                        onFit += tt => tcol.Config.OnFit(tt.GetTermMap(ii));
                    }
                }
                var est = new TermEstimator(env, infos);
                if (onFit == null)
                    return est;
                return est.WithOnFitDelegate(onFit);
            }
        }
    }
}
