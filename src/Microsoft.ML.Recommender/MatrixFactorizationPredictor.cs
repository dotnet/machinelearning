// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Recommender;
using Microsoft.ML.Runtime.Recommender.Internal;
using Microsoft.ML.Runtime.Model.Pfa;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data.IO;

[assembly: LoadableClass(typeof(MatrixFactorizationPredictor), null, typeof(SignatureLoadModel), "Matrix Factorization Predictor Executor", MatrixFactorizationPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(MatrixFactorizationPredictionTransformer), typeof(MatrixFactorizationPredictionTransformer),
    null, typeof(SignatureLoadModel), "", MatrixFactorizationPredictionTransformer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Recommender
{
    public sealed class MatrixFactorizationPredictor : IPredictor, ICanSaveModel, ICanSaveInTextFormat, ISchemaBindableMapper
    {
        internal const string LoaderSignature = "MFPredictor";
        internal const string RegistrationName = "MatrixFactorizationPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FAFAMAPD",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MatrixFactorizationPredictor).Assembly.FullName);
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
        // the ML.NET output type is noted as being double.

        // Packed _m by _k matrix.
        private readonly float[] _p;
        // Packed _k by _n matrix.
        private readonly float[] _q;

        public PredictionKind PredictionKind
        {
            get { return PredictionKind.Recommendation; }
        }

        public ColumnType OutputType { get { return NumberType.Float; } }

        public ColumnType InputXType { get; }
        public ColumnType InputYType { get; }

        internal MatrixFactorizationPredictor(IHostEnvironment env, SafeTrainingAndModelBuffer buffer, KeyType xType, KeyType yType)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(buffer, nameof(buffer));
            _host.CheckValue(xType, nameof(xType));
            _host.CheckValue(yType, nameof(xType));

            _host.Assert(xType.RawKind == DataKind.U4);
            _host.Assert(yType.RawKind == DataKind.U4);
            buffer.Get(out _m, out _n, out _k, out _p, out _q);
            _host.Assert(_n == xType.Count);
            _host.Assert(_m == yType.Count);
            _host.Assert(_p.Length == _m * _k);
            _host.Assert(_q.Length == _n * _k);

            InputXType = xType;
            InputYType = yType;
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
            // float[m * k]: the row dimension factor matrix P
            // float[k * n]: the column dimension factor matrix Q

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

            InputXType = new KeyType(DataKind.U4, nMin, _n);
            InputYType = new KeyType(DataKind.U4, mMin, _m);
        }

        /// <summary>
        /// Load model from the given context
        /// </summary>
        public static MatrixFactorizationPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MatrixFactorizationPredictor(env, ctx);
        }

        /// <summary>
        /// Save model to the given context
        /// </summary>
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
            // float[m * k]: the row dimension factor matrix P
            // float[k * n]: the column dimension factor matrix Q

            _host.Assert(_m > 0);
            _host.Assert(_n > 0);
            _host.Assert(_k > 0);
            ctx.Writer.Write(_m);
            ctx.Writer.Write((InputYType as KeyType).Min);
            ctx.Writer.Write(_n);
            ctx.Writer.Write((InputXType as KeyType).Min);
            ctx.Writer.Write(_k);
            _host.Assert(Utils.Size(_p) == _m * _k);
            _host.Assert(Utils.Size(_q) == _n * _k);
            Utils.WriteSinglesNoCount(ctx.Writer, _p, _m * _k);
            Utils.WriteSinglesNoCount(ctx.Writer, _q, _n * _k);
        }

        /// <summary>
        /// Save the trained matrix factorization model (two factor matrices) in text format
        /// </summary>
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

        private ValueGetter<float> GetGetter(ValueGetter<uint> xGetter, ValueGetter<uint> yGetter)
        {
            _host.AssertValue(xGetter);
            _host.AssertValue(yGetter);

            uint x = 0;
            uint y = 0;

            var mapper = GetMapper<uint, uint, float>();
            ValueGetter<float> del =
                (ref float value) =>
                {
                    xGetter(ref x);
                    yGetter(ref y);
                    mapper(ref x, ref y, ref value);
                };
            return del;
        }

        /// <summary>
        /// Create the mapper required by matrix factorization's predictor. That mapper maps two
        /// index inputs (e.g., row index and column index) to the value located by the two indexes
        /// in the training matrix. In recommender system where the training matrix stores ratings
        /// from users to items, the mappers maps user ID and item ID to the rating of that item given
        /// by the user.
        /// </summary>
        public ValueMapper<TXIn, TYIn, TOut> GetMapper<TXIn, TYIn, TOut>()
        {
            string msg = null;
            msg = "Invalid TXIn in GetMapper: " + typeof(TXIn);
            _host.Check(typeof(TXIn) == typeof(uint), msg);

            msg = "Invalid TYIn in GetMapper: " + typeof(TYIn);
            _host.Check(typeof(TYIn) == typeof(uint), msg);

            msg = "Invalid TOut in GetMapper: " + typeof(TOut);
            _host.Check(typeof(TOut) == typeof(float), msg);

            ValueMapper<uint, uint, float> mapper = MapperCore;
            return mapper as ValueMapper<TXIn, TYIn, TOut>;
        }

        private void MapperCore(ref uint srcCol, ref uint srcRow, ref float dst)
        {
            // REVIEW tfinley: The key-type version a bit more "strict" than the predictor
            // version, since the predictor version can't know the maximum bound during
            // training. For higher-than-expected values, the predictor version would return
            // 0, rather than NaN as we do here. It is in my mind an open question as to what
            // is actually correct.
            if (srcRow == 0 || srcRow > _m || srcCol == 0 || srcCol > _n)
            {
                dst = float.NaN;
                return;
            }
            dst = Score((int)(srcCol - 1), (int)(srcRow - 1));
        }

        private float Score(int col, int row)
        {
            _host.Assert(0 <= row && row < _m);
            _host.Assert(0 <= col && col < _n);
            float score = 0;
            int poffset = row * _k;
            int qoffset = col * _k;
            for (int i = 0; i < _k; i++)
                score += _p[poffset + i] * _q[qoffset + i];
            return score;
        }

        public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(schema);
            return new RowMapper(this, schema, Schema.Create(new ScoreMapperSchema(OutputType, MetadataUtils.Const.ScoreColumnKind.Regression)));
        }

        private sealed class RowMapper : ISchemaBoundRowMapper
        {
            private readonly MatrixFactorizationPredictor _parent;

            private readonly int _xColIndex;
            private readonly int _yColIndex;
            private readonly string _xColName;
            private readonly string _yColName;

            private IHost Host => _parent._host;
            public Schema Schema { get; }
            public Schema InputSchema => InputRoleMappedSchema.Schema;

            public RoleMappedSchema InputRoleMappedSchema { get; }

            public RowMapper(MatrixFactorizationPredictor parent, RoleMappedSchema schema, Schema outputSchema)
            {
                Contracts.AssertValue(parent);
                _parent = parent;

                // Check role of X
                var xList = schema.GetColumns(RecommendUtils.XKind);
                string msg = $"'{RecommendUtils.XKind}' column doesn't exist or not unique";
                Host.Check(Utils.Size(xList) == 1, msg);

                // Check role of Y
                var yList = schema.GetColumns(RecommendUtils.YKind);
                msg = $"'{RecommendUtils.YKind}' column doesn't exist or not unique";
                Host.Check(Utils.Size(yList) == 1, msg);

                _xColName = xList[0].Name;
                _xColIndex = xList[0].Index;

                _yColName = yList[0].Name;
                _yColIndex = yList[0].Index;

                CheckInputSchema(schema.Schema, _xColIndex, _yColIndex);
                InputRoleMappedSchema = schema;
                Schema = outputSchema;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < Schema.ColumnCount; i++)
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
                // See if role X's type matches the one expected in the trained predictor
                var type = schema.GetColumnType(xCol);
                string msg = string.Format("Input X type '{0}' incompatible with predictor X type '{1}'", type, _parent.InputXType);
                Host.CheckParam(type.Equals(_parent.InputXType), nameof(schema), msg);

                // See if role Y's type matches the one expected in the trained predictor
                type = schema.GetColumnType(yCol);
                msg = string.Format("Input Y type '{0}' incompatible with predictor Y type '{1}'", type, _parent.InputYType);
                Host.CheckParam(type.Equals(_parent.InputYType), nameof(schema), msg);
            }

            private Delegate[] CreateGetter(IRow input, bool[] active)
            {
                Host.CheckValue(input, nameof(input));
                Host.Assert(Utils.Size(active) == Schema.ColumnCount);

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
                var active = Utils.BuildArray(Schema.ColumnCount, predicate);
                var getters = CreateGetter(input, active);
                disposer = null;
                return new SimpleRow(Schema, input, getters);
            }

            public ISchemaBindableMapper Bindable { get { return _parent; } }
        }
    }

    public sealed class MatrixFactorizationPredictionTransformer : PredictionTransformerBase<MatrixFactorizationPredictor, GenericScorer>, ICanSaveModel
    {
        public const string LoaderSignature = "MaFactPredXf";
        public string XColumnName { get; }
        public string YColumnName { get; }
        public ColumnType XColumnType { get; }
        public ColumnType YColumnType { get; }
        protected override GenericScorer Scorer { get; set; }

        /// <summary>
        /// Build a transformer based on matrix factorization predictor (model) and the input schema (trainSchema). The created
        /// transformer can only transform IDataView objects compatible to the input schema; that is, that IDataView must contain
        /// columns specified by <see cref="XColumnName"/>, <see cref="XColumnType"/>, <see cref="YColumnName"/>, and <see cref="YColumnType"></see>.
        /// The output column is "Score" by default but user can append a string to it.
        /// </summary>
        /// <param name="host">Eviroment object for showing information</param>
        /// <param name="model">The model trained by one of the training functions in <see cref="MatrixFactorizationTrainer"/></param>
        /// <param name="trainSchema">Targeted schema that containing columns named as xColumnName</param>
        /// <param name="xColumnName">The name of the column used as role X in matrix factorization world</param>
        /// <param name="yColumnName">The name of the column used as role Y in matrix factorization world</param>
        /// <param name="scoreColumnNameSuffix">A string attached to the output column name of this transformer</param>
        public MatrixFactorizationPredictionTransformer(IHostEnvironment host, MatrixFactorizationPredictor model, Schema trainSchema,
            string xColumnName, string yColumnName, string scoreColumnNameSuffix = "")
            :base(Contracts.CheckRef(host, nameof(host)).Register(nameof(MatrixFactorizationPredictionTransformer)), model, trainSchema)
        {
            Host.CheckNonEmpty(xColumnName, nameof(yColumnName));
            Host.CheckNonEmpty(xColumnName, nameof(yColumnName));

            XColumnName = xColumnName;
            YColumnName = yColumnName;

            if (!trainSchema.TryGetColumnIndex(XColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(XColumnName), RecommendUtils.XKind.Value, XColumnName);
            XColumnType = trainSchema.GetColumnType(xCol);
            if (!trainSchema.TryGetColumnIndex(YColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(yCol), RecommendUtils.YKind.Value, YColumnName);

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, model);

            var schema = GetSchema();
            var args = new GenericScorer.Arguments { Suffix = scoreColumnNameSuffix };
            Scorer = new GenericScorer(Host, args, new EmptyDataView(Host, trainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        private RoleMappedSchema GetSchema()
        {
            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RecommendUtils.XKind, XColumnName));
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RecommendUtils.YKind, YColumnName));
            var schema = new RoleMappedSchema(TrainSchema, roles);
            return schema;
        }

        /// <summary>
        /// The counter constructor of re-creating <see cref="MatrixFactorizationPredictionTransformer"/> from the context where
        /// the original transform is saved.
        /// </summary>
        public MatrixFactorizationPredictionTransformer(IHostEnvironment host, ModelLoadContext ctx)
            :base(Contracts.CheckRef(host, nameof(host)).Register(nameof(MatrixFactorizationPredictionTransformer)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // string: the column name of matrix's column ids.
            // string: the column name of matrix's row ids.

            XColumnName = ctx.LoadString();
            YColumnName = ctx.LoadString();

            if (!TrainSchema.TryGetColumnIndex(XColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(XColumnName), RecommendUtils.XKind.Value, XColumnName);
            XColumnType = TrainSchema.GetColumnType(xCol);

            if (!TrainSchema.TryGetColumnIndex(YColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(YColumnName), RecommendUtils.YKind.Value, YColumnName);
            YColumnType = TrainSchema.GetColumnType(yCol);

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, Model);

            var schema = GetSchema();
            var args = new GenericScorer.Arguments { Suffix = "" };
            Scorer = new GenericScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public override Schema GetOutputSchema(Schema inputSchema)
        {
            if (!inputSchema.TryGetColumnIndex(XColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), RecommendUtils.XKind.Value, XColumnName);
            if (!inputSchema.TryGetColumnIndex(YColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), RecommendUtils.YKind.Value, YColumnName);

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // model: prediction model.
            // stream: empty data view that contains train schema.
            // ids of strings: feature columns.
            // float: scorer threshold
            // id of string: scorer threshold column

            ctx.SaveModel(Model, DirModel);
            ctx.SaveBinaryStream(DirTransSchema, writer =>
            {
                using (var ch = Host.Start("Saving train schema"))
                {
                    var saver = new BinarySaver(Host, new BinarySaver.Arguments { Silent = true });
                    DataSaverUtils.SaveDataView(ch, saver, new EmptyDataView(Host, TrainSchema), writer.BaseStream);
                }
            });

            ctx.SaveString(XColumnName);
            ctx.SaveString(YColumnName);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MAFAPRED", // "MA"trix "FA"torization "PRED"iction
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MatrixFactorizationPredictionTransformer).Assembly.FullName);
        }
        private static MatrixFactorizationPredictionTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
            => new MatrixFactorizationPredictionTransformer(env, ctx);

    }
}
