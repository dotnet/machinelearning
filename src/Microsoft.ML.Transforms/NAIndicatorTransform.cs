// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(IDataTransform), typeof(NAIndicatorTransform), typeof(NAIndicatorTransform.Arguments), typeof(SignatureDataTransform),
    NAIndicatorTransform.FriendlyName, NAIndicatorTransform.LoadName, "NAIndicator", NAIndicatorTransform.ShortName, DocName = "transform/NAHandle.md")]

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(IDataTransform), typeof(NAIndicatorTransform), null, typeof(SignatureLoadDataTransform),
    NAIndicatorTransform.FriendlyName, NAIndicatorTransform.LoadName)]

[assembly: LoadableClass(NAIndicatorTransform.Summary, typeof(NAIndicatorTransform), null, typeof(SignatureLoadModel),
    NAIndicatorTransform.FriendlyName, NAIndicatorTransform.LoadName)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(NAIndicatorTransform), null, typeof(SignatureLoadRowMapper),
   NAIndicatorTransform.FriendlyName, NAIndicatorTransform.LoadName)]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="NAIndicator"]'/>
    public sealed class NAIndicatorTransform : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
        {
            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        public const string LoadName = "NaIndicatorTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                // REVIEW: temporary name
                modelSignature: "NAIND TF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName,
                loaderAssemblyName: typeof(NAIndicatorTransform).Assembly.FullName);
        }

        internal const string Summary = "Create a boolean output column with the same number of slots as the input column, where the output value"
            + " is true if the value in the input column is missing.";
        internal const string FriendlyName = "NA Indicator Transform";
        internal const string ShortName = "NAInd";

        internal static string TestType(ColumnType type)
        {
            // Item type must have an NA value. We'll get the predicate again later when we're ready to use it.
            Delegate del;
            if (Conversions.Instance.TryGetIsNAPredicate(type.ItemType, out del))
                return null;
            return string.Format("Type '{0}' is not supported by {1} since it doesn't have an NA value",
                type, LoadName);
        }

        public class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            public ColumnInfo(string input, string output)
            {
                Input = input;
                Output = output;
            }
        }

        private const string RegistrationName = nameof(NAIndicatorTransform);

        // The input column types
        private ColumnType[] _inputTypes;
        // The output column types
        private ColumnType[] _outputTypes;

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var type = inputSchema.GetColumnType(srcCol);
            string reason = TestType(type);
            if (reason != null)
                throw Host.ExceptParam(nameof(inputSchema), reason);
        }

        private (ColumnType[], ColumnType[]) GetTypes(ISchema schema)
        {
            var inputTypes = new ColumnType[ColumnPairs.Length];
            var outputTypes = new ColumnType[ColumnPairs.Length];
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                schema.TryGetColumnIndex(ColumnPairs[i].input, out int colSrc);
                var type = schema.GetColumnType(colSrc);

                if (!type.IsVector)
                {
                    inputTypes[i] = type.AsPrimitive;
                    outputTypes[i] = BoolType.Instance;
                }
                else
                {
                    inputTypes[i] = new VectorType(type.ItemType.AsPrimitive, type.AsVector);
                    outputTypes[i] = new VectorType(BoolType.Instance, type.AsVector);
                }
            }
            return (inputTypes, outputTypes);
        }

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="columns">Specifies the names of the input columns for the transform and the resulting output column names.</param>
        public NAIndicatorTransform(IHostEnvironment env, IDataView input, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(NAIndicatorTransform)), GetColumnPairs(columns))
        {
            // Check that all the input columns are present and correct.
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                if (!input.Schema.TryGetColumnIndex(ColumnPairs[i].input, out int srcCol))
                    throw Host.ExceptSchemaMismatch(nameof(input), "input", ColumnPairs[i].input);
                CheckInputColumn(input.Schema, i, srcCol);
            }
            (_inputTypes, _outputTypes) = GetTypes(input.Schema);
        }

        private NAIndicatorTransform(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            Host.AssertValue(ctx);
            _outputTypes = null;
            _inputTypes = null;
        }

        /// <summary>
        /// Factory method for SignatureDataTransform.
        /// </summary>
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];

                cols[i] = new ColumnInfo(item.Source, item.Name);
            };
            return new NAIndicatorTransform(env, input, cols).MakeDataTransform(input);
        }

        /// <summary>
        /// Factory method for SignatureDataTransform.
        /// </summary>
        public static NAIndicatorTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new NAIndicatorTransform(host, ctx);
        }

        /// <summary>
        /// Factory method for SignatureDataTransform.
        /// </summary>
        public static IDataTransform Create(IHostEnvironment env, IDataView input, params ColumnInfo[] columns)
            => new NAIndicatorTransform(env, input, columns).MakeDataTransform(input);

        /// <summary>
        /// Factory method for SignatureDataTransform.
        /// </summary>
        public static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        /// <summary>
        /// Factory method for SignatureDataTransform.
        /// </summary>
        public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        /// <summary>
        /// Returns the isNA predicate for the respective type.
        /// </summary>
        private Delegate GetIsNADelegate(ColumnType type)
        {
            Func<ColumnType, Delegate> func = GetIsNADelegate<int>;
            return Utils.MarshalInvoke(func, type.ItemType.RawType, type);
        }

        private Delegate GetIsNADelegate<T>(ColumnType type)
        {
            return Conversions.Instance.GetIsNAPredicate<T>(type.ItemType);
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            SaveColumns(ctx);
        }

        private ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < ColumnPairs.Length);
            return _outputTypes[iinfo];
        }

        private Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < ColumnPairs.Length);
            disposer = null;
            if (_outputTypes == null || _inputTypes == null)
                (_inputTypes, _outputTypes) = GetTypes(input.Schema);

            if (!_outputTypes[iinfo].IsVector)
                return ComposeGetterOne(input, iinfo);
            return ComposeGetterVec(input, iinfo);
        }

        /// <summary>
        /// Getter generator for single valued inputs.
        /// </summary>
        private ValueGetter<bool> ComposeGetterOne(IRow input, int iinfo)
        {
            Func<IRow, int, ValueGetter<bool>> func = ComposeGetterOne<int>;
            return Utils.MarshalInvoke(func, _outputTypes[iinfo].RawType, input, iinfo);
        }

        private ValueGetter<bool> ComposeGetterOne<T>(IRow input, int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < ColumnPairs.Length);
            Host.Assert(input.IsColumnActive(iinfo));

            var getSrc = input.GetGetter<T>(iinfo);
            var isNA = Conversions.Instance.GetIsNAPredicate<T>(_inputTypes[iinfo]);
            T src = default(T);
            return
                (ref bool dst) =>
                {
                    getSrc(ref src);
                    dst = isNA(ref src);
                };
        }

        /// <summary>
        /// Getter generator for vector valued inputs.
        /// </summary>
        private ValueGetter<VBuffer<bool>> ComposeGetterVec(IRow input, int iinfo)
        {
            Func<IRow, int, ValueGetter<VBuffer<bool>>> func = ComposeGetterVec<int>;
            return Utils.MarshalInvoke(func, _outputTypes[iinfo].ItemType.RawType, input, iinfo);
        }

        private ValueGetter<VBuffer<bool>> ComposeGetterVec<T>(IRow input, int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < ColumnPairs.Length);
            Host.Assert(input.IsColumnActive(iinfo));

            var getSrc = input.GetGetter<VBuffer<T>>(iinfo);
            var isNA = Conversions.Instance.GetIsNAPredicate<T>(_inputTypes[iinfo]);
            var val = default(T);
            bool defaultIsNA = isNA(ref val);
            var src = default(VBuffer<T>);
            var indices = new List<int>();
            return
                (ref VBuffer<bool> dst) =>
                {
                    // Sense indicates if the values added to the indices list represent NAs or non-NAs.
                    bool sense;
                    getSrc(ref src);
                    FindNAs(ref src, isNA, defaultIsNA, indices, out sense);
                    FillValues(src.Length, ref dst, indices, sense);
                };
        }

        /// <summary>
        /// Adds all NAs (or non-NAs) to the indices List.  Whether NAs or non-NAs have been added is indicated by the bool sense.
        /// </summary>
        private void FindNAs<T>(ref VBuffer<T> src, RefPredicate<T> isNA, bool defaultIsNA, List<int> indices, out bool sense)
        {
            Host.AssertValue(isNA);
            Host.AssertValue(indices);

            // Find the indices of all of the NAs.
            indices.Clear();
            var srcValues = src.Values;
            var srcCount = src.Count;
            if (src.IsDense)
            {
                for (int i = 0; i < srcCount; i++)
                {
                    if (isNA(ref srcValues[i]))
                        indices.Add(i);
                }
                sense = true;
            }
            else if (!defaultIsNA)
            {
                var srcIndices = src.Indices;
                for (int ii = 0; ii < srcCount; ii++)
                {
                    if (isNA(ref srcValues[ii]))
                        indices.Add(srcIndices[ii]);
                }
                sense = true;
            }
            else
            {
                // Note that this adds non-NAs to indices -- this is indicated by sense being false.
                var srcIndices = src.Indices;
                for (int ii = 0; ii < srcCount; ii++)
                {
                    if (!isNA(ref srcValues[ii]))
                        indices.Add(srcIndices[ii]);
                }
                sense = false;
            }
        }

        /// <summary>
        ///  Fills indicator values for vectors.  The indices is a list that either holds all of the NAs or all
        ///  of the non-NAs, indicated by sense being true or false respectively.
        /// </summary>
        private void FillValues(int srcLength, ref VBuffer<bool> dst, List<int> indices, bool sense)
        {
            var dstValues = dst.Values;
            var dstIndices = dst.Indices;

            if (indices.Count == 0)
            {
                if (sense)
                {
                    // Return empty VBuffer.
                    dst = new VBuffer<bool>(srcLength, 0, dstValues, dstIndices);
                    return;
                }

                // Return VBuffer filled with 1's.
                Utils.EnsureSize(ref dstValues, srcLength, false);
                for (int i = 0; i < srcLength; i++)
                    dstValues[i] = true;
                dst = new VBuffer<bool>(srcLength, dstValues, dstIndices);
                return;
            }

            if (sense && indices.Count < srcLength / 2)
            {
                // Will produce sparse output.
                int dstCount = indices.Count;
                Utils.EnsureSize(ref dstValues, dstCount, false);
                Utils.EnsureSize(ref dstIndices, dstCount, false);

                indices.CopyTo(dstIndices);
                for (int ii = 0; ii < dstCount; ii++)
                    dstValues[ii] = true;

                Host.Assert(dstCount <= srcLength);
                dst = new VBuffer<bool>(srcLength, dstCount, dstValues, dstIndices);
            }
            else if (!sense && srcLength - indices.Count < srcLength / 2)
            {
                // Will produce sparse output.
                int dstCount = srcLength - indices.Count;
                Utils.EnsureSize(ref dstValues, dstCount, false);
                Utils.EnsureSize(ref dstIndices, dstCount, false);

                // Appends the length of the src to make the loop simpler,
                // as the length of src will never be reached in the loop.
                indices.Add(srcLength);

                int iiDst = 0;
                int iiSrc = 0;
                int iNext = indices[iiSrc];
                for (int i = 0; i < srcLength; i++)
                {
                    Host.Assert(0 <= i && i <= iNext);
                    Host.Assert(iiSrc + iiDst == i);
                    if (i < iNext)
                    {
                        Host.Assert(iiDst < dstCount);
                        dstValues[iiDst] = true;
                        dstIndices[iiDst++] = i;
                    }
                    else
                    {
                        Host.Assert(iiSrc + 1 < indices.Count);
                        Host.Assert(iNext < indices[iiSrc + 1]);
                        iNext = indices[++iiSrc];
                    }
                }
                Host.Assert(srcLength == iiSrc + iiDst);
                Host.Assert(iiDst == dstCount);

                dst = new VBuffer<bool>(srcLength, dstCount, dstValues, dstIndices);
            }
            else
            {
                // Will produce dense output.
                Utils.EnsureSize(ref dstValues, srcLength, false);

                // Appends the length of the src to make the loop simpler,
                // as the length of src will never be reached in the loop.
                indices.Add(srcLength);

                int ii = 0;
                for (int i = 0; i < srcLength; i++)
                {
                    Host.Assert(0 <= i && i <= indices[ii]);
                    if (i == indices[ii])
                    {
                        dstValues[i] = sense;
                        ii++;
                        Host.Assert(ii < indices.Count);
                        Host.Assert(indices[ii - 1] < indices[ii]);
                    }
                    else
                        dstValues[i] = !sense;
                }

                dst = new VBuffer<bool>(srcLength, dstValues, dstIndices);
            }
        }

        protected override IRowMapper MakeRowMapper(ISchema schema)
            => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private sealed class ColInfo
            {
                public readonly string Name;
                public readonly string Source;
                public readonly ColumnType TypeSrc;

                public ColInfo(string name, string source, ColumnType type)
                {
                    Name = name;
                    Source = source;
                    TypeSrc = type;
                }
            }

            private readonly NAIndicatorTransform _parent;
            private readonly ColInfo[] _infos;
            private readonly ColumnType[] _types;
            // The isNA delegates, parallel to Infos.
            private readonly Delegate[] _isNAs;

            public Mapper(NAIndicatorTransform parent, ISchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = CreateInfos(inputSchema);
                _types = new ColumnType[_parent.ColumnPairs.Length];
                _isNAs = new Delegate[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var type = _infos[i].TypeSrc;
                    if (!type.IsVector)
                        _types[i] = BoolType.Instance;
                    else
                        _types[i] = new VectorType(BoolType.Instance, type.AsVector);
                    _isNAs[i] = _parent.GetIsNADelegate(type);
                }
            }

            private ColInfo[] CreateInfos(ISchema inputSchema)
            {
                Host.AssertValue(inputSchema);
                var infos = new ColInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colSrc))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].input);
                    _parent.CheckInputColumn(inputSchema, i, colSrc);
                    var type = inputSchema.GetColumnType(colSrc);
                    infos[i] = new ColInfo(_parent.ColumnPairs[i].output, _parent.ColumnPairs[i].input, type);
                }
                return infos;
            }

            public override RowMapperColumnInfo[] GetOutputColumns()
            {
                var result = new RowMapperColumnInfo[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                    result[i] = new RowMapperColumnInfo(_parent.ColumnPairs[i].output, _types[i], default);
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _infos.Length);
                disposer = null;

                if (!_infos[iinfo].TypeSrc.IsVector)
                    return ComposeGetterOne(input, iinfo);
                return ComposeGetterVec(input, iinfo);
            }

            /// <summary>
            /// Getter generator for single valued inputs.
            /// </summary>
            private ValueGetter<bool> ComposeGetterOne(IRow input, int iinfo)
                => Utils.MarshalInvoke(ComposeGetterOne<int>, _infos[iinfo].TypeSrc.RawType, input, iinfo);

            private ValueGetter<bool> ComposeGetterOne<T>(IRow input, int iinfo)
            {
                var getSrc = input.GetGetter<T>(ColMapNewToOld[iinfo]);
                var src = default(T);
                var isNA = (RefPredicate<T>)_isNAs[iinfo];

                ValueGetter<bool> getter;

                return getter =
                    (ref bool dst) =>
                    {
                        getSrc(ref src);
                        dst = isNA(ref src);
                    };
            }

            /// <summary>
            /// Getter generator for vector valued inputs.
            /// </summary>
            private ValueGetter<VBuffer<bool>> ComposeGetterVec(IRow input, int iinfo)
                => Utils.MarshalInvoke(ComposeGetterVec<int>, _infos[iinfo].TypeSrc.ItemType.RawType, input, iinfo);

            private ValueGetter<VBuffer<bool>> ComposeGetterVec<T>(IRow input, int iinfo)
            {
                var getSrc = input.GetGetter<VBuffer<T>>(ColMapNewToOld[iinfo]);
                var isNA = (RefPredicate<T>)_isNAs[iinfo];
                var val = default(T);
                var defaultIsNA = isNA(ref val);
                var src = default(VBuffer<T>);
                var indices = new List<int>();

                ValueGetter<VBuffer<bool>> getter;

                return getter =
                    (ref VBuffer<bool> dst) =>
                    {
                        // Sense indicates if the values added to the indices list represent NAs or non-NAs.
                        bool sense;
                        getSrc(ref src);
                        _parent.FindNAs(ref src, isNA, defaultIsNA, indices, out sense);
                        _parent.FillValues(src.Length, ref dst, indices, sense);
                    };
            }
        }
    }

    public sealed class NAIndicatorEstimator : IEstimator<NAIndicatorTransform>
    {
        private readonly IHost _host;
        private readonly NAIndicatorTransform.ColumnInfo[] _columns;

        public NAIndicatorEstimator(IHostEnvironment env, string name, string source = null)
            : this(env, new NAIndicatorTransform.ColumnInfo(source ?? name, name))
        {
        }

        public NAIndicatorEstimator(IHostEnvironment env, params NAIndicatorTransform.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(NAIndicatorEstimator));
            _columns = columns;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                string reason = NAIndicatorTransform.TestType(col.ItemType);
                if (reason != null)
                    throw _host.ExceptParam(nameof(inputSchema), reason);
                ColumnType type = !col.ItemType.IsVector ? (ColumnType) BoolType.Instance : new VectorType(BoolType.Instance, col.ItemType.AsVector);
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, col.Kind, type, false, default);
            }
            return new SchemaShape(result.Values);
        }

        public NAIndicatorTransform Fit(IDataView input) => new NAIndicatorTransform(_host, input, _columns);
    }

    /// <summary>
    /// Extension methods for the static-pipeline over <see cref="PipelineColumn"/> objects.
    /// </summary>
    public static class NAIndicatorExtensions
    {
        private interface IColInput
        {
            PipelineColumn Input { get; }
        }

        private sealed class OutScalar<TValue> : Scalar<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutScalar(Scalar<TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVectorColumn<TValue> : Vector<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVectorColumn(Vector<TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class OutVarVectorColumn<TValue> : VarVector<bool>, IColInput
        {
            public PipelineColumn Input { get; }

            public OutVarVectorColumn(VarVector<TValue> input)
                : base(Reconciler.Inst, input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var infos = new NAIndicatorTransform.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var col = (IColInput)toOutput[i];
                    infos[i] = new NAIndicatorTransform.ColumnInfo(inputNames[col.Input], outputNames[toOutput[i]]);
                }
                return new NAIndicatorEstimator(env, infos);
            }
        }

        public static Scalar<bool> IsMissingValue(this Scalar<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<float>(input);
        }

        public static Scalar<bool> IsMissingValue(this Scalar<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<double>(input);
        }

        public static Scalar<bool> IsMissingValue(this Scalar<string> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutScalar<string>(input);
        }

        public static Vector<bool> IsMissingValue(this Vector<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<float>(input);
        }

        public static Vector<bool> IsMissingValue(this Vector<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<double>(input);
        }

        public static Vector<bool> IsMissingValue(this Vector<string> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVectorColumn<string>(input);
        }

        public static VarVector<bool> IsMissingValue(this VarVector<float> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<float>(input);
        }

        public static VarVector<bool> IsMissingValue(this VarVector<double> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<double>(input);
        }

        public static VarVector<bool> IsMissingValue(this VarVector<string> input)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutVarVectorColumn<string>(input);
        }
    }
}