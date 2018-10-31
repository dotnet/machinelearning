// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Recommender;
using Microsoft.ML.Runtime.Recommender.Internal;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(typeof(MatrixFactorizationPredictor), null, typeof(SignatureLoadModel), "Matrix Factorization Predictor Executor", MatrixFactorizationPredictor.LoaderSignature)]

[assembly: LoadableClass(typeof(MatrixFactorizationPredictionTransformer), typeof(MatrixFactorizationPredictionTransformer),
    null, typeof(SignatureLoadModel), "", MatrixFactorizationPredictionTransformer.LoaderSignature)]

namespace Microsoft.ML.Runtime.Recommender
{
    /// <summary>
    /// <see cref="MatrixFactorizationPredictor"/> stores two factor matrices, P and Q, for approximating the training matrix, R, by P * Q,
    /// where * is a matrix multiplication. This predictor expects two inputs, row index and column index, and produces the (approximated)
    /// value at the location specified by the two inputs in R. More specifically, if input row and column indices are u and v, respectively.
    /// The output (a scalar) would be the inner product product of the u-th row in P and the v-th column in Q.
    /// </summary>
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
        private readonly int _numberOfRows;
        // The number of columns.
        private readonly int _numberofColumns;
        // The rank of the factor matrices.
        private readonly int _approximationRank;
        // Packed _numberOfRows by _approximationRank matrix.
        private readonly float[] _leftFactorMatrix;
        // Packed _approximationRank by _numberofColumns matrix.
        private readonly float[] _rightFactorMatrix;

        public PredictionKind PredictionKind
        {
            get { return PredictionKind.Recommendation; }
        }

        public ColumnType OutputType { get { return NumberType.Float; } }

        public ColumnType MatrixColumnIndexType { get; }
        public ColumnType MatrixRowIndexType { get; }

        internal MatrixFactorizationPredictor(IHostEnvironment env, SafeTrainingAndModelBuffer buffer, KeyType matrixColumnIndexType, KeyType matrixRowIndexType)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.Assert(matrixColumnIndexType.RawKind == DataKind.U4);
            _host.Assert(matrixRowIndexType.RawKind == DataKind.U4);
            _host.CheckValue(buffer, nameof(buffer));
            _host.CheckValue(matrixColumnIndexType, nameof(matrixColumnIndexType));
            _host.CheckValue(matrixRowIndexType, nameof(matrixRowIndexType));

            buffer.Get(out _numberOfRows, out _numberofColumns, out _approximationRank, out _leftFactorMatrix, out _rightFactorMatrix);
            _host.Assert(_numberofColumns == matrixColumnIndexType.Count);
            _host.Assert(_numberOfRows == matrixRowIndexType.Count);
            _host.Assert(_leftFactorMatrix.Length == _numberOfRows * _approximationRank);
            _host.Assert(_rightFactorMatrix.Length == _numberofColumns * _approximationRank);

            MatrixColumnIndexType = matrixColumnIndexType;
            MatrixRowIndexType = matrixRowIndexType;
        }

        private MatrixFactorizationPredictor(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            // *** Binary format ***
            // int: number of rows (m), the limit on row
            // ulong: Minimum value of the row key-type
            // int: number of columns (n), the limit on column
            // ulong: Minimum value of the column key-type
            // int: rank of factor matrices (k)
            // float[m * k]: the left factor matrix
            // float[k * n]: the right factor matrix

            _numberOfRows = ctx.Reader.ReadInt32();
            _host.CheckDecode(_numberOfRows > 0);
            ulong mMin = ctx.Reader.ReadUInt64();
            _host.CheckDecode((ulong)_numberOfRows <= ulong.MaxValue - mMin);
            _numberofColumns = ctx.Reader.ReadInt32();
            _host.CheckDecode(_numberofColumns > 0);
            ulong nMin = ctx.Reader.ReadUInt64();
            _host.CheckDecode((ulong)_numberofColumns <= ulong.MaxValue - nMin);
            _approximationRank = ctx.Reader.ReadInt32();
            _host.CheckDecode(_approximationRank > 0);

            _leftFactorMatrix = Utils.ReadSingleArray(ctx.Reader, checked(_numberOfRows * _approximationRank));
            _rightFactorMatrix = Utils.ReadSingleArray(ctx.Reader, checked(_numberofColumns * _approximationRank));

            MatrixColumnIndexType = new KeyType(DataKind.U4, nMin, _numberofColumns);
            MatrixRowIndexType = new KeyType(DataKind.U4, mMin, _numberOfRows);
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
            // int: number of rows (m), the limit on row
            // ulong: Minimum value of the row key-type
            // int: number of columns (n), the limit on column
            // ulong: Minimum value of the column key-type
            // int: rank of factor matrices (k)
            // float[m * k]: the left factor matrix
            // float[k * n]: the right factor matrix

            _host.Check(_numberOfRows > 0, "Number of rows must be positive");
            _host.Check(_numberofColumns > 0, "Number of columns must be positive");
            _host.Check(_approximationRank > 0, "Number of latent factors must be positive");
            ctx.Writer.Write(_numberOfRows);
            ctx.Writer.Write((MatrixRowIndexType as KeyType).Min);
            ctx.Writer.Write(_numberofColumns);
            ctx.Writer.Write((MatrixColumnIndexType as KeyType).Min);
            ctx.Writer.Write(_approximationRank);
            _host.Check(Utils.Size(_leftFactorMatrix) == _numberOfRows * _approximationRank, "Unexpected matrix size of a factor matrix (matrix P in LIBMF paper)");
            _host.Check(Utils.Size(_rightFactorMatrix) == _numberofColumns * _approximationRank, "Unexpected matrix size of a factor matrix (matrix Q in LIBMF paper)");
            Utils.WriteSinglesNoCount(ctx.Writer, _leftFactorMatrix, _numberOfRows * _approximationRank);
            Utils.WriteSinglesNoCount(ctx.Writer, _rightFactorMatrix, _numberofColumns * _approximationRank);
        }

        /// <summary>
        /// Save the trained matrix factorization model (two factor matrices) in text format
        /// </summary>
        public void SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine("# Imputed matrix is P * Q'");
            writer.WriteLine("# P in R^({0} x {1}), rows correpond to Y item", _numberOfRows, _approximationRank);
            for (int i = 0; i < _leftFactorMatrix.Length; ++i)
            {
                writer.Write(_leftFactorMatrix[i].ToString("G"));
                if (i % _approximationRank == _approximationRank - 1)
                    writer.WriteLine();
                else
                    writer.Write('\t');
            }
            writer.WriteLine("# Q in R^({0} x {1}), rows correpond to X item", _numberofColumns, _approximationRank);
            for (int i = 0; i < _rightFactorMatrix.Length; ++i)
            {
                writer.Write(_rightFactorMatrix[i].ToString("G"));
                if (i % _approximationRank == _approximationRank - 1)
                    writer.WriteLine();
                else
                    writer.Write('\t');
            }
        }

        private ValueGetter<float> GetGetter(ValueGetter<uint> matrixColumnIndexGetter, ValueGetter<uint> matrixRowIndexGetter)
        {
            _host.AssertValue(matrixColumnIndexGetter);
            _host.AssertValue(matrixRowIndexGetter);

            uint matrixColumnIndex = 0;
            uint matrixRowIndex = 0;

            var mapper = GetMapper<uint, uint, float>();
            ValueGetter<float> del =
                (ref float value) =>
                {
                    matrixColumnIndexGetter(ref matrixColumnIndex);
                    matrixRowIndexGetter(ref matrixRowIndex);
                    mapper(ref matrixColumnIndex, ref matrixRowIndex, ref value);
                };
            return del;
        }

        /// <summary>
        /// Create the mapper required by matrix factorization's predictor. That mapper maps two
        /// index inputs (e.g., row index and column index) to an approximated value located by the
        /// two indexes in the training matrix. In recommender system where the training matrix stores
        /// ratings from users to items, the mappers maps user ID and item ID to the rating of that
        /// item given by the user.
        /// </summary>
        public ValueMapper<TMatrixColumnIndexIn, TMatrixRowIndexIn, TOut> GetMapper<TMatrixColumnIndexIn, TMatrixRowIndexIn, TOut>()
        {
            string msg = null;
            msg = "Invalid " + nameof(TMatrixColumnIndexIn) + " in GetMapper: " + typeof(TMatrixColumnIndexIn);
            _host.Check(typeof(TMatrixColumnIndexIn) == typeof(uint), msg);

            msg = "Invalid " + nameof(TMatrixRowIndexIn) + " in GetMapper: " + typeof(TMatrixRowIndexIn);
            _host.Check(typeof(TMatrixRowIndexIn) == typeof(uint), msg);

            msg = "Invalid " + nameof(TOut) + " in GetMapper: " + typeof(TOut);
            _host.Check(typeof(TOut) == typeof(float), msg);

            ValueMapper<uint, uint, float> mapper = MapperCore;
            return mapper as ValueMapper<TMatrixColumnIndexIn, TMatrixRowIndexIn, TOut>;
        }

        private void MapperCore(ref uint srcCol, ref uint srcRow, ref float dst)
        {
            // REVIEW: The key-type version a bit more "strict" than the predictor
            // version, since the predictor version can't know the maximum bound during
            // training. For higher-than-expected values, the predictor version would return
            // 0, rather than NaN as we do here. It is in my mind an open question as to what
            // is actually correct.
            if (srcRow == 0 || srcRow > _numberOfRows || srcCol == 0 || srcCol > _numberofColumns)
            {
                dst = float.NaN;
                return;
            }
            dst = Score((int)(srcCol - 1), (int)(srcRow - 1));
        }

        private float Score(int columnIndex, int rowIndex)
        {
            _host.Assert(0 <= rowIndex && rowIndex < _numberOfRows);
            _host.Assert(0 <= columnIndex && columnIndex < _numberofColumns);
            float score = 0;
            // Starting position of the rowIndex-th row in the left factor factor matrix
            int rowOffset = rowIndex * _approximationRank;
            // Starting position of the columnIndex-th column in the right factor factor matrix
            int columnOffset = columnIndex * _approximationRank;
            for (int i = 0; i < _approximationRank; i++)
                score += _leftFactorMatrix[rowOffset + i] * _rightFactorMatrix[columnOffset + i];
            return score;
        }

        /// <summary>
        /// Create a row mapper based on regression scorer. Because matrix factorization predictor maps a tuple of a row ID (u) and a column ID (v)
        /// to the expected numerical value at the u-th row and the v-th column in the considered matrix, it is essentially a regressor.
        /// </summary>
        public ISchemaBoundMapper Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(schema);
            return new RowMapper(env, this, schema, Schema.Create(new ScoreMapperSchema(OutputType, MetadataUtils.Const.ScoreColumnKind.Regression)));
        }

        private sealed class RowMapper : ISchemaBoundRowMapper
        {
            private readonly MatrixFactorizationPredictor _parent;
            // The tail "ColumnIndex" means the column index in IDataView
            private readonly int _matrixColumnIndexColumnIndex;
            private readonly int _matrixRowIndexCololumnIndex;
            // The tail "ColumnName" means the column name in IDataView
            private readonly string _matrixColumnIndexColumnName;
            private readonly string _matrixRowIndexColumnName;
            private IHostEnvironment _env;
            public Schema Schema { get; }
            public Schema InputSchema => InputRoleMappedSchema.Schema;

            public RoleMappedSchema InputRoleMappedSchema { get; }

            public RowMapper(IHostEnvironment env, MatrixFactorizationPredictor parent, RoleMappedSchema schema, Schema outputSchema)
            {
                Contracts.AssertValue(parent);
                _env = env;
                _parent = parent;

                // Check role of matrix column index
                var matrixColumnList = schema.GetColumns(RecommenderUtils.MatrixColumnIndexKind);
                string msg = $"'{RecommenderUtils.MatrixColumnIndexKind}' column doesn't exist or not unique";
                _env.Check(Utils.Size(matrixColumnList) == 1, msg);

                // Check role of matrix row index
                var matrixRowList = schema.GetColumns(RecommenderUtils.MatrixRowIndexKind);
                msg = $"'{RecommenderUtils.MatrixRowIndexKind}' column doesn't exist or not unique";
                _env.Check(Utils.Size(matrixRowList) == 1, msg);

                _matrixColumnIndexColumnName = matrixColumnList[0].Name;
                _matrixColumnIndexColumnIndex = matrixColumnList[0].Index;

                _matrixRowIndexColumnName = matrixRowList[0].Name;
                _matrixRowIndexCololumnIndex = matrixRowList[0].Index;

                CheckInputSchema(schema.Schema, _matrixColumnIndexColumnIndex, _matrixRowIndexCololumnIndex);
                InputRoleMappedSchema = schema;
                Schema = outputSchema;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                for (int i = 0; i < Schema.ColumnCount; i++)
                {
                    if (predicate(i))
                        return col => (col == _matrixColumnIndexColumnIndex || col == _matrixRowIndexCololumnIndex);
                }
                return col => false;
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RecommenderUtils.MatrixColumnIndexKind.Bind(_matrixColumnIndexColumnName);
                yield return RecommenderUtils.MatrixRowIndexKind.Bind(_matrixRowIndexColumnName);
            }

            private void CheckInputSchema(ISchema schema, int matrixColumnIndexCol, int matrixRowIndexCol)
            {
                // See if matrix-column-index role's type matches the one expected in the trained predictor
                var type = schema.GetColumnType(matrixColumnIndexCol);
                string msg = string.Format("Input column index type '{0}' incompatible with predictor's column index type '{1}'", type, _parent.MatrixColumnIndexType);
                _env.CheckParam(type.Equals(_parent.MatrixColumnIndexType), nameof(schema), msg);

                // See if matrix-column-index  role's type matches the one expected in the trained predictor
                type = schema.GetColumnType(matrixRowIndexCol);
                msg = string.Format("Input row index type '{0}' incompatible with predictor' row index type '{1}'", type, _parent.MatrixRowIndexType);
                _env.CheckParam(type.Equals(_parent.MatrixRowIndexType), nameof(schema), msg);
            }

            private Delegate[] CreateGetter(IRow input, bool[] active)
            {
                _env.CheckValue(input, nameof(input));
                _env.Assert(Utils.Size(active) == Schema.ColumnCount);

                var getters = new Delegate[1];
                if (active[0])
                {
                    CheckInputSchema(input.Schema, _matrixColumnIndexColumnIndex, _matrixRowIndexCololumnIndex);
                    var matrixColumnIndexGetter = input.GetGetter<uint>(_matrixColumnIndexColumnIndex);
                    var matrixRowIndexGetter = input.GetGetter<uint>(_matrixRowIndexCololumnIndex);
                    getters[0] = _parent.GetGetter(matrixColumnIndexGetter, matrixRowIndexGetter);
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
        public string MatrixColumnIndexColumnName { get; }
        public string MatrixRowIndexColumnName { get; }
        public ColumnType MatrixColumnIndexColumnType { get; }
        public ColumnType MatrixRowIndexColumnType { get; }
        protected override GenericScorer Scorer { get; set; }

        /// <summary>
        /// Build a transformer based on matrix factorization predictor (model) and the input schema (trainSchema). The created
        /// transformer can only transform IDataView objects compatible to the input schema; that is, that IDataView must contain
        /// columns specified by <see cref="MatrixColumnIndexColumnName"/>, <see cref="MatrixColumnIndexColumnType"/>, <see cref="MatrixRowIndexColumnName"/>, and <see cref="MatrixRowIndexColumnType"></see>.
        /// The output column is "Score" by default but user can append a string to it.
        /// </summary>
        /// <param name="env">Eviroment object for showing information</param>
        /// <param name="model">The model trained by one of the training functions in <see cref="MatrixFactorizationTrainer"/></param>
        /// <param name="trainSchema">Targeted schema that containing columns named as xColumnName</param>
        /// <param name="matrixColumnIndexColumnName">The name of the column used as role <see cref="RecommenderUtils.MatrixColumnIndexKind"/> in matrix factorization world</param>
        /// <param name="matrixRowIndexColumnName">The name of the column used as role <see cref="RecommenderUtils.MatrixRowIndexKind"/> in matrix factorization world</param>
        /// <param name="scoreColumnNameSuffix">A string attached to the output column name of this transformer</param>
        public MatrixFactorizationPredictionTransformer(IHostEnvironment env, MatrixFactorizationPredictor model, Schema trainSchema,
            string matrixColumnIndexColumnName, string matrixRowIndexColumnName, string scoreColumnNameSuffix = "")
            :base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MatrixFactorizationPredictionTransformer)), model, trainSchema)
        {
            Host.CheckNonEmpty(matrixColumnIndexColumnName, nameof(matrixRowIndexColumnName));
            Host.CheckNonEmpty(matrixColumnIndexColumnName, nameof(matrixRowIndexColumnName));

            MatrixColumnIndexColumnName = matrixColumnIndexColumnName;
            MatrixRowIndexColumnName = matrixRowIndexColumnName;

            if (!trainSchema.TryGetColumnIndex(MatrixColumnIndexColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(MatrixColumnIndexColumnName), RecommenderUtils.MatrixColumnIndexKind.Value, MatrixColumnIndexColumnName);
            MatrixColumnIndexColumnType = trainSchema.GetColumnType(xCol);
            if (!trainSchema.TryGetColumnIndex(MatrixRowIndexColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(yCol), RecommenderUtils.MatrixRowIndexKind.Value, MatrixRowIndexColumnName);

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, model);

            var schema = GetSchema();
            var args = new GenericScorer.Arguments { Suffix = scoreColumnNameSuffix };
            Scorer = new GenericScorer(Host, args, new EmptyDataView(Host, trainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        private RoleMappedSchema GetSchema()
        {
            var roles = new List<KeyValuePair<RoleMappedSchema.ColumnRole, string>>();
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RecommenderUtils.MatrixColumnIndexKind, MatrixColumnIndexColumnName));
            roles.Add(new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RecommenderUtils.MatrixRowIndexKind, MatrixRowIndexColumnName));
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

            MatrixColumnIndexColumnName = ctx.LoadString();
            MatrixRowIndexColumnName = ctx.LoadString();

            if (!TrainSchema.TryGetColumnIndex(MatrixColumnIndexColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(MatrixColumnIndexColumnName), RecommenderUtils.MatrixColumnIndexKind.Value, MatrixColumnIndexColumnName);
            MatrixColumnIndexColumnType = TrainSchema.GetColumnType(xCol);

            if (!TrainSchema.TryGetColumnIndex(MatrixRowIndexColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(MatrixRowIndexColumnName), RecommenderUtils.MatrixRowIndexKind.Value, MatrixRowIndexColumnName);
            MatrixRowIndexColumnType = TrainSchema.GetColumnType(yCol);

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, Model);

            var schema = GetSchema();
            var args = new GenericScorer.Arguments { Suffix = "" };
            Scorer = new GenericScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        public override Schema GetOutputSchema(Schema inputSchema)
        {
            if (!inputSchema.TryGetColumnIndex(MatrixColumnIndexColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), RecommenderUtils.MatrixColumnIndexKind.Value, MatrixColumnIndexColumnName);
            if (!inputSchema.TryGetColumnIndex(MatrixRowIndexColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), RecommenderUtils.MatrixRowIndexKind.Value, MatrixRowIndexColumnName);

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

            ctx.SaveString(MatrixColumnIndexColumnName);
            ctx.SaveString(MatrixRowIndexColumnName);
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
