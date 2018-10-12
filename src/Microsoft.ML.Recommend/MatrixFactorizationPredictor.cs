//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Recommend;
using Microsoft.ML.Runtime.Recommend.Internal;

[assembly: LoadableClass(typeof(MatrixFactorizationPredictor), null, typeof(SignatureLoadModel),
    "Matrix Factorization Predictor Executor", MatrixFactorizationPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.Recommend
{
    public sealed class MatrixFactorizationPredictor : IPredictor, ICanSaveModel, ICanSaveInTextFormat, IMIValueMapper, ISchemaBindableMapper,
        IUserHistoryToItemsRecommender
    {
        public const string LoaderSignature = "MFPredictorExec";
        public const string RegistrationName = "MatrixFactorizationPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                "MFPREDIC",
                0x00010001, // Initial
                0x00010001,
                0x00010001,
                "MFPredictorExec",
                null);
        }

        private readonly IHost _host;
        // The number of rows.
        private readonly int _m;
        // The number of columns.
        private readonly int _n;
        // The internal dimension.
        private readonly int _k;
        // The libMF is always single precision, so we will
        // keep it as such even in the double build, even when
        // the TLC output type is noted as being double.

        // Packed _m by _k matrix.
        private readonly Single[] _p;
        // Packed _k by _n matrix.
        private readonly Single[] _q;

        private readonly KeyType _inputYType;
        private readonly KeyType _inputXType;

        public PredictionKind PredictionKind
        {
            get { return PredictionKind.Recommendation; }
        }

        public ColumnType OutputType { get { return NumberType.Float; } }
        // REVIEW tfinley: Worth caching the below?
        public ColumnType InputXType { get { return _inputXType; } }
        public ColumnType InputYType { get { return _inputYType; } }

        internal MatrixFactorizationPredictor(IHostEnvironment env, SafeTrainingAndModelBuffer buffer, KeyType xType, KeyType yType)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(buffer, nameof(buffer));
            _host.CheckValue(xType, nameof(xType));
            _host.CheckValue(yType, nameof(xType));

            buffer.Get(out _m, out _n, out _k, out _p, out _q);
            _host.Assert(xType.RawKind == DataKind.U4);
            _host.Assert(yType.RawKind == DataKind.U4);
            _host.Assert(_n == xType.Count);
            _host.Assert(_m == yType.Count);

            _inputXType = xType;
            _inputYType = yType;
        }

        private MatrixFactorizationPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            // *** Binary format ***
            // int: number of rows (m), the limit on y
            // ulong: Minimum value of the y key-type
            // int: number of columns (n), the limit on x
            // ulong: Minimum value of the x key-type
            // int: internal dimension of matrices (k)
            // Single[m * k]: the row dimension factor matrix P
            // Single[k * n]: the column dimension factor matrix Q

            _m = ctx.Reader.ReadInt32();
            _host.CheckDecode(_m > 0);
            ulong mMin = ctx.Reader.ReadUInt64();
            _host.CheckDecode((ulong)_m <= ulong.MaxValue - mMin);
            _n = ctx.Reader.ReadInt32();
            _host.CheckDecode(_n > 0);
            ulong nMin = ctx.Reader.ReadUInt64();
            _host.CheckDecode((ulong)_n <= ulong.MaxValue - nMin);
            _k = ctx.Reader.ReadInt32();
            _host.CheckDecode(_k > 0);

            _p = Utils.ReadSingleArray(ctx.Reader, checked(_m * _k));
            _q = Utils.ReadSingleArray(ctx.Reader, checked(_n * _k));

            _inputXType = new KeyType(DataKind.U4, nMin, _n);
            _inputYType = new KeyType(DataKind.U4, mMin, _m);
        }

        public static MatrixFactorizationPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MatrixFactorizationPredictor(env, ctx);
        }

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of rows (m), the limit on y
            // ulong: Minimum value of the y key-type
            // int: number of columns (n), the limit on x
            // ulong: Minimum value of the x key-type
            // int: internal dimension of matrices (k)
            // Single[m * k]: the row dimension factor matrix P
            // Single[k * n]: the column dimension factor matrix Q

            _host.Assert(_m > 0);
            _host.Assert(_n > 0);
            _host.Assert(_k > 0);
            ctx.Writer.Write(_m);
            ctx.Writer.Write(_inputYType.Min);
            ctx.Writer.Write(_n);
            ctx.Writer.Write(_inputXType.Min);
            ctx.Writer.Write(_k);
            _host.Assert(Utils.Size(_p) == _m * _k);
            _host.Assert(Utils.Size(_q) == _n * _k);
            Utils.WriteSinglesNoCount(ctx.Writer, _p, _m * _k);
            Utils.WriteSinglesNoCount(ctx.Writer, _q, _n * _k);
        }

        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine("# Imputed matrix is P * Q'");
            writer.WriteLine("# P in R^({0} x {1}), rows correpond to Y item", _m, _k);
            for (int i = 0; i < _p.Length; ++i)
            {
                writer.Write(_p[i].ToString("G"));
                if (i % _k == _k - 1)
                    writer.WriteLine();
                else
                    writer.Write('\t');
            }
            writer.WriteLine("# Q in R^({0} x {1}), rows correpond to X item", _n, _k);
            for (int i = 0; i < _q.Length; ++i)
            {
                writer.Write(_q[i].ToString("G"));
                if (i % _k == _k - 1)
                    writer.WriteLine();
                else
                    writer.Write('\t');
            }
        }

        private ValueGetter<Float> GetGetter(ValueGetter<uint> xGetter, ValueGetter<uint> yGetter)
        {
            _host.AssertValue(xGetter);
            _host.AssertValue(yGetter);

            uint x = 0;
            uint y = 0;

            var mapper = GetMapper<uint, uint, Float>();
            ValueGetter<Float> del =
                (ref Float value) =>
                {
                    xGetter(ref x);
                    yGetter(ref y);
                    mapper(ref x, ref y, ref value);
                };
            return del;
        }

        public ValueMapper<TXIn, TYIn, TOut> GetMapper<TXIn, TYIn, TOut>()
        {
            if (typeof(TXIn) != typeof(uint))
                throw _host.Except("Invalid TXIn in GetMapper: '{0}'", typeof(TXIn));
            if (typeof(TYIn) != typeof(uint))
                throw _host.Except("Invalid TYIn in GetMapper: '{0}'", typeof(TYIn));
            if (typeof(TOut) != typeof(Float))
                throw _host.Except("Invalid TOut in GetMapper: '{0}'", typeof(TOut));

            ValueMapper<uint, uint, Float> mapper = MapperCore;
            return mapper as ValueMapper<TXIn, TYIn, TOut>;
        }

        private void MapperCore(ref uint srcCol, ref uint srcRow, ref Float dst)
        {
            // REVIEW tfinley: The key-type version a bit more "strict" than the predictor
            // version, since the predictor version can't know the maximum bound during
            // training. For higher-than-expected values, the predictor version would return
            // 0, rather than NaN as we do here. It is in my mind an open question as to what
            // is actually correct.
            if (srcRow == 0 || srcRow > _m || srcCol == 0 || srcCol > _n)
            {
                dst = Float.NaN;
                return;
            }
            dst = Score((int)(srcCol - 1), (int)(srcRow - 1));
        }

        private Float Score(int col, int row)
        {
            _host.Assert(0 <= row && row < _m);
            _host.Assert(0 <= col && col < _n);
            Float score = 0;
            int poffset = row * _k;
            int qoffset = col * _k;
            for (int i = 0; i < _k; i++)
                score += _p[poffset + i] * _q[qoffset + i];
            return score;
        }

        public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Contracts.AssertValue(schema);
            return new RowMapper(this, schema, new ScoreMapperSchema(OutputType, MetadataUtils.Const.ScoreColumnKind.Regression));
        }

        private sealed class RowMapper : ISchemaBoundRowMapper
        {
            private readonly MatrixFactorizationPredictor _parent;
            private readonly RoleMappedSchema _inputSchema;
            private readonly ScoreMapperSchema _outputSchema;

            private readonly int _xColIndex;
            private readonly int _yColIndex;
            private readonly string _xColName;
            private readonly string _yColName;

            public ISchema InputSchema => _inputSchema.Schema;

            public RoleMappedSchema InputRoleMappedSchema => _inputSchema;

            public RowMapper(MatrixFactorizationPredictor parent, RoleMappedSchema schema, ScoreMapperSchema outputSchema)
            {
                Contracts.AssertValue(parent);
                _parent = parent;

                var list = schema.GetColumns(RecommendUtils.XKind);
                if (Utils.Size(list) != 1)
                    throw Contracts.Except($"'{RecommendUtils.XKind}' column doesn't exist");
                _xColName = list[0].Name;
                _xColIndex = list[0].Index;

                list = schema.GetColumns(RecommendUtils.YKind);
                if (Utils.Size(list) != 1)
                    throw Contracts.Except($"'{RecommendUtils.YKind}' column doesn't exist");
                _yColName = list[0].Name;
                _yColIndex = list[0].Index;

                CheckInputSchema(schema.Schema, _xColIndex, _yColIndex);
                _inputSchema = schema;
                _outputSchema = outputSchema;
            }

            public ISchema Schema => OutputSchema;

            public ISchema OutputSchema
            {
                get { return _outputSchema; }
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < OutputSchema.ColumnCount; i++)
                {
                    if (predicate(i))
                        return col => (col == _xColIndex || col == _yColIndex);
                }
                return col => false;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RecommendUtils.XKind.Bind(_xColName);
                yield return RecommendUtils.YKind.Bind(_yColName);
            }

            private void CheckInputSchema(ISchema schema, int xCol, int yCol)
            {
                var type = schema.GetColumnType(xCol);
                if (!type.Equals(_parent.InputXType))
                {
                    throw Contracts.ExceptParam(nameof(schema), "Input X type '{0}' incompatible with predictor X type '{1}'",
                        type, _parent.InputXType);
                }
                type = schema.GetColumnType(yCol);
                if (!type.Equals(_parent.InputYType))
                {
                    throw Contracts.ExceptParam(nameof(schema), "Input Y type '{0}' incompatible with predictor Y type '{1}'",
                        type, _parent.InputYType);
                }
            }

            private Delegate[] CreateGetter(IRow input, bool[] active)
            {
                Contracts.CheckValue(input, nameof(input));
                Contracts.Assert(Utils.Size(active) == OutputSchema.ColumnCount);

                var getters = new Delegate[1];
                if (active[0])
                {
                    CheckInputSchema(input.Schema, _xColIndex, _yColIndex);
                    var xGetter = input.GetGetter<uint>(_xColIndex);
                    var yGetter = input.GetGetter<uint>(_yColIndex);
                    getters[0] = _parent.GetGetter(xGetter, yGetter);
                }
                return getters;
            }

            public IRow GetRow(IRow input, Func<int, bool> predicate, out Action disposer)
            {
                var active = Utils.BuildArray(OutputSchema.ColumnCount, predicate);
                var getters = CreateGetter(input, active);
                disposer = null;
                return new SimpleRow(OutputSchema, input, getters);
            }

            public ISchemaBindableMapper Bindable { get { return _parent; } }
        }

        public int UserFeaturesSize { get { return -1; } }
        public KeyType UserIdType { get { return _inputYType; } }
        public KeyType ItemIdType { get { return _inputXType; } }

        public UserHistoryToItemsMapper GetRecommendMapper(int recommendationCount, bool includeHistory)
        {
            _host.CheckParam(recommendationCount > 0, nameof(recommendationCount), "Recommendation count must be positive");

            var scores = new List<KeyValuePair<uint, Float>>(_n);
            var history = new HashSet<uint>();

            UserHistoryToItemsMapper mapper =
                (ref uint userId, ref VBuffer<float> userFeatures, ref VBuffer<uint> items, ref VBuffer<float> weights,
                    ref VBuffer<uint> recoItems, ref VBuffer<float> recoScores) =>
                {
                    scores.Clear();
                    if (!includeHistory)
                    {
                        history.Clear();
                        for (int i = 0; i < items.Count; i++)
                            history.Add(items.Values[i]);
                    }

                    int recoCount = Math.Min(_n, recommendationCount);
                    if (userId == 0)
                        recoCount = 0;
                    else
                    {
                        int row = (int)userId - 1;
                        for (int i = 0; i < _n; i++)
                        {
                            Single rowScore = Single.NegativeInfinity;
                            if (includeHistory || !history.Contains((uint)(i + 1)))
                                rowScore = Score(i, row);

                            scores.Add(new KeyValuePair<uint, float>((uint)i + 1, rowScore));
                        }

                        scores.Sort((x, y) => y.Value.CompareTo(x.Value));
                    }
                    uint[] outItems = Utils.Size(recoItems.Values) < recommendationCount ? new uint[recommendationCount] : recoItems.Values;
                    Single[] outScores = Utils.Size(recoScores.Values) < recommendationCount ? new Single[recommendationCount] : recoScores.Values;

                    int j;
                    for (j = 0; j < recoCount; j++)
                    {
                        outItems[j] = (uint)scores[j].Key;
                        outScores[j] = scores[j].Value;
                    }
                    recoItems = new VBuffer<uint>(recoCount, outItems, recoItems.Indices);
                    recoScores = new VBuffer<Single>(recoCount, outScores, recoScores.Indices);
                };

            return mapper;
        }
    }
}
