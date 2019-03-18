// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Recommender;
using Microsoft.ML.Recommender.Internal;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Recommender;

[assembly: LoadableClass(typeof(MatrixFactorizationModelParameters), null, typeof(SignatureLoadModel), "Matrix Factorization Predictor Executor", MatrixFactorizationModelParameters.LoaderSignature)]

[assembly: LoadableClass(typeof(MatrixFactorizationPredictionTransformer), typeof(MatrixFactorizationPredictionTransformer),
    null, typeof(SignatureLoadModel), "", MatrixFactorizationPredictionTransformer.LoaderSignature)]

namespace Microsoft.ML.Trainers.Recommender
{
    /// <summary>
    /// Model parameters for matrix factorization recommender.
    /// </summary>
    /// <remarks>
    /// <see cref="MatrixFactorizationModelParameters"/> stores two factor matrices, P and Q, for approximating the training matrix, R, by P * Q,
    /// where * is a matrix multiplication. This model expects two inputs, row index and column index, and produces the (approximated)
    /// value at the location specified by the two inputs in R. More specifically, if input row and column indices are u and v, respectively.
    /// The output (a scalar) would be the inner product product of the u-th row in P and the v-th column in Q.
    /// </remarks>
    public sealed class MatrixFactorizationModelParameters : IPredictor, ICanSaveModel, ICanSaveInTextFormat, ISchemaBindableMapper
    {
        internal const string LoaderSignature = "MFPredictor";
        internal const string RegistrationName = "MatrixFactorizationPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FAFAMAPD",
                // verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Removed Min in KeyType
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(MatrixFactorizationModelParameters).Assembly.FullName);
        }
        private const uint VersionNoMinCount = 0x00010002;

        private readonly IHost _host;

        ///<summary> The number of rows.</summary>
        public readonly int NumberOfRows;

        ///<summary> The number of columns.</summary>
        public readonly int NumberOfColumns;

        ///<summary> The rank of the factor matrices.</summary>
        public readonly int ApproximationRank;

        /// <summary>
        /// Left approximation matrix
        /// </summary>
        /// <remarks>
        /// This is two dimensional matrix with size of <see cref="NumberOfRows"/> * <see cref="ApproximationRank"/> flattened into one-dimensional matrix.
        /// Row by row.
        /// </remarks>
        public IReadOnlyList<float> LeftFactorMatrix => _leftFactorMatrix;

        private readonly float[] _leftFactorMatrix;
        /// <summary>
        /// Right approximation matrix
        /// </summary>
        /// <remarks>
        /// This is two dimensional matrix with size of <see cref="ApproximationRank"/> * <see cref="NumberOfColumns"/> flattened into one-dimensional matrix.
        /// Row by row.
        /// </remarks>
        public IReadOnlyList<float> RightFactorMatrix => _rightFactorMatrix;

        private readonly float[] _rightFactorMatrix;

        PredictionKind IPredictor.PredictionKind => PredictionKind.Recommendation;

        private DataViewType OutputType => NumberDataViewType.Single;

        internal DataViewType MatrixColumnIndexType { get; }
        internal DataViewType MatrixRowIndexType { get; }

        internal MatrixFactorizationModelParameters(IHostEnvironment env, SafeTrainingAndModelBuffer buffer, KeyType matrixColumnIndexType, KeyType matrixRowIndexType)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.Assert(matrixColumnIndexType.RawType == typeof(uint));
            _host.Assert(matrixRowIndexType.RawType == typeof(uint));
            _host.CheckValue(buffer, nameof(buffer));
            _host.CheckValue(matrixColumnIndexType, nameof(matrixColumnIndexType));
            _host.CheckValue(matrixRowIndexType, nameof(matrixRowIndexType));
            buffer.Get(out NumberOfRows, out NumberOfColumns, out ApproximationRank, out var leftFactorMatrix, out var rightFactorMatrix);
            _leftFactorMatrix = leftFactorMatrix;
            _rightFactorMatrix = rightFactorMatrix;
            _host.Assert(NumberOfColumns == matrixColumnIndexType.GetCountAsInt32(_host));
            _host.Assert(NumberOfRows == matrixRowIndexType.GetCountAsInt32(_host));
            _host.Assert(_leftFactorMatrix.Length == NumberOfRows * ApproximationRank);
            _host.Assert(_rightFactorMatrix.Length == ApproximationRank * NumberOfColumns);

            MatrixColumnIndexType = matrixColumnIndexType;
            MatrixRowIndexType = matrixRowIndexType;
        }

        private MatrixFactorizationModelParameters(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            // *** Binary format ***
            // int: number of rows (m), the limit on row
            // int: number of columns (n), the limit on column
            // int: rank of factor matrices (k)
            // float[m * k]: the left factor matrix
            // float[k * n]: the right factor matrix

            NumberOfRows = ctx.Reader.ReadInt32();
            _host.CheckDecode(NumberOfRows > 0);
            if (ctx.Header.ModelVerWritten < VersionNoMinCount)
            {
                ulong mMin = ctx.Reader.ReadUInt64();
                // We no longer support non zero Min for KeyType.
                _host.CheckDecode(mMin == 0);
                _host.CheckDecode((ulong)NumberOfRows <= ulong.MaxValue - mMin);
            }
            NumberOfColumns = ctx.Reader.ReadInt32();
            _host.CheckDecode(NumberOfColumns > 0);
            if (ctx.Header.ModelVerWritten < VersionNoMinCount)
            {
                ulong nMin = ctx.Reader.ReadUInt64();
                // We no longer support non zero Min for KeyType.
                _host.CheckDecode(nMin == 0);
                _host.CheckDecode((ulong)NumberOfColumns <= ulong.MaxValue - nMin);
            }
            ApproximationRank = ctx.Reader.ReadInt32();
            _host.CheckDecode(ApproximationRank > 0);

            _leftFactorMatrix = Utils.ReadSingleArray(ctx.Reader, checked(NumberOfRows * ApproximationRank));
            _rightFactorMatrix = Utils.ReadSingleArray(ctx.Reader, checked(NumberOfColumns * ApproximationRank));

            MatrixColumnIndexType = new KeyType(typeof(uint), NumberOfColumns);
            MatrixRowIndexType = new KeyType(typeof(uint), NumberOfRows);
        }

        /// <summary>
        /// Load model from the given context
        /// </summary>
        private static MatrixFactorizationModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new MatrixFactorizationModelParameters(env, ctx);
        }

        /// <summary>
        /// Save model to the given context
        /// </summary>
        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of rows (m), the limit on row
            // int: number of columns (n), the limit on column
            // int: rank of factor matrices (k)
            // float[m * k]: the left factor matrix
            // float[k * n]: the right factor matrix

            _host.Check(NumberOfRows > 0, "Number of rows must be positive");
            _host.Check(NumberOfColumns > 0, "Number of columns must be positive");
            _host.Check(ApproximationRank > 0, "Number of latent factors must be positive");
            ctx.Writer.Write(NumberOfRows);
            ctx.Writer.Write(NumberOfColumns);
            ctx.Writer.Write(ApproximationRank);
            _host.Check(Utils.Size(_leftFactorMatrix) == NumberOfRows * ApproximationRank, "Unexpected matrix size of a factor matrix (matrix P in LIBMF paper)");
            _host.Check(Utils.Size(_rightFactorMatrix) == NumberOfColumns * ApproximationRank, "Unexpected matrix size of a factor matrix (matrix Q in LIBMF paper)");
            Utils.WriteSinglesNoCount(ctx.Writer, _leftFactorMatrix);
            Utils.WriteSinglesNoCount(ctx.Writer, _rightFactorMatrix);
        }

        /// <summary>
        /// Save the trained matrix factorization model (two factor matrices) in text format
        /// </summary>
        void ICanSaveInTextFormat.SaveAsText(TextWriter writer, RoleMappedSchema schema)
        {
            writer.WriteLine("# Imputed matrix is P * Q'");
            writer.WriteLine("# P in R^({0} x {1}), rows correpond to Y item", NumberOfRows, ApproximationRank);
            for (int i = 0; i < _leftFactorMatrix.Length; ++i)
            {
                writer.Write(_leftFactorMatrix[i].ToString("G"));
                if (i % ApproximationRank == ApproximationRank - 1)
                    writer.WriteLine();
                else
                    writer.Write('\t');
            }
            writer.WriteLine("# Q in R^({0} x {1}), rows correpond to X item", NumberOfColumns, ApproximationRank);
            for (int i = 0; i < _rightFactorMatrix.Length; ++i)
            {
                writer.Write(_rightFactorMatrix[i].ToString("G"));
                if (i % ApproximationRank == ApproximationRank - 1)
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
                    mapper(in matrixColumnIndex, ref matrixRowIndex, ref value);
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
        private ValueMapper<TMatrixColumnIndexIn, TMatrixRowIndexIn, TOut> GetMapper<TMatrixColumnIndexIn, TMatrixRowIndexIn, TOut>()
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

        private void MapperCore(in uint srcCol, ref uint srcRow, ref float dst)
        {
            // REVIEW: The key-type version a bit more "strict" than the predictor
            // version, since the predictor version can't know the maximum bound during
            // training. For higher-than-expected values, the predictor version would return
            // 0, rather than NaN as we do here. It is in my mind an open question as to what
            // is actually correct.
            if (srcRow == 0 || srcRow > NumberOfRows || srcCol == 0 || srcCol > NumberOfColumns)
            {
                dst = float.NaN;
                return;
            }
            dst = Score((int)(srcCol - 1), (int)(srcRow - 1));
        }

        private float Score(int columnIndex, int rowIndex)
        {
            _host.Assert(0 <= rowIndex && rowIndex < NumberOfRows);
            _host.Assert(0 <= columnIndex && columnIndex < NumberOfColumns);
            float score = 0;
            // Starting position of the rowIndex-th row in the left factor factor matrix
            int rowOffset = rowIndex * ApproximationRank;
            // Starting position of the columnIndex-th column in the right factor factor matrix
            int columnOffset = columnIndex * ApproximationRank;
            for (int i = 0; i < ApproximationRank; i++)
                score += _leftFactorMatrix[rowOffset + i] * _rightFactorMatrix[columnOffset + i];
            return score;
        }

        /// <summary>
        /// Create a row mapper based on regression scorer. Because matrix factorization predictor maps a tuple of a row ID (u) and a column ID (v)
        /// to the expected numerical value at the u-th row and the v-th column in the considered matrix, it is essentially a regressor.
        /// </summary>
        ISchemaBoundMapper ISchemaBindableMapper.Bind(IHostEnvironment env, RoleMappedSchema schema)
        {
            Contracts.AssertValue(env);
            env.AssertValue(schema);
            return new RowMapper(env, this, schema, ScoreSchemaFactory.Create(OutputType, AnnotationUtils.Const.ScoreColumnKind.Regression));
        }

        private sealed class RowMapper : ISchemaBoundRowMapper
        {
            private readonly MatrixFactorizationModelParameters _parent;
            // The tail "ColumnIndex" means the column index in IDataView
            private readonly int _matrixColumnIndexColumnIndex;
            private readonly int _matrixRowIndexCololumnIndex;
            // The tail "ColumnName" means the column name in IDataView
            private readonly string _matrixColumnIndexColumnName;
            private readonly string _matrixRowIndexColumnName;
            private IHostEnvironment _env;
            public DataViewSchema InputSchema => InputRoleMappedSchema.Schema;
            public DataViewSchema OutputSchema { get; }

            public RoleMappedSchema InputRoleMappedSchema { get; }

            public RowMapper(IHostEnvironment env, MatrixFactorizationModelParameters parent, RoleMappedSchema schema, DataViewSchema outputSchema)
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
                OutputSchema = outputSchema;
            }

            /// <summary>
            /// Given a set of columns, return the input columns that are needed to generate those output columns.
            /// </summary>
            public IEnumerable<DataViewSchema.Column> GetDependenciesForNewColumns(IEnumerable<DataViewSchema.Column> dependingColumns)
            {
                if (dependingColumns.Count() == 0)
                    return Enumerable.Empty<DataViewSchema.Column>();

                return InputSchema.Where(col => col.Index == _matrixColumnIndexColumnIndex || col.Index == _matrixRowIndexCololumnIndex);
            }

            public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
            {
                yield return RecommenderUtils.MatrixColumnIndexKind.Bind(_matrixColumnIndexColumnName);
                yield return RecommenderUtils.MatrixRowIndexKind.Bind(_matrixRowIndexColumnName);
            }

            private void CheckInputSchema(DataViewSchema schema, int matrixColumnIndexCol, int matrixRowIndexCol)
            {
                // See if matrix-column-index role's type matches the one expected in the trained predictor
                var type = schema[matrixColumnIndexCol].Type;
                string msg = string.Format("Input column index type '{0}' incompatible with predictor's column index type '{1}'", type, _parent.MatrixColumnIndexType);
                _env.CheckParam(type.Equals(_parent.MatrixColumnIndexType), nameof(schema), msg);

                // See if matrix-column-index  role's type matches the one expected in the trained predictor
                type = schema[matrixRowIndexCol].Type;
                msg = string.Format("Input row index type '{0}' incompatible with predictor' row index type '{1}'", type, _parent.MatrixRowIndexType);
                _env.CheckParam(type.Equals(_parent.MatrixRowIndexType), nameof(schema), msg);
            }

            private Delegate[] CreateGetter(DataViewRow input, bool[] active)
            {
                _env.CheckValue(input, nameof(input));
                _env.Assert(Utils.Size(active) == OutputSchema.Count);

                var getters = new Delegate[1];
                if (active[0])
                {
                    // First check if expected columns are ok and then create getters to acccess those columns' values.
                    CheckInputSchema(input.Schema, _matrixColumnIndexColumnIndex, _matrixRowIndexCololumnIndex);
                    var matrixColumnIndexGetter = RowCursorUtils.GetGetterAs<uint>(NumberDataViewType.UInt32, input, _matrixColumnIndexColumnIndex);
                    var matrixRowIndexGetter = RowCursorUtils.GetGetterAs<uint>(NumberDataViewType.UInt32, input, _matrixRowIndexCololumnIndex);

                    // Assign the getter of the prediction score. It maps a pair of matrix column index and matrix row index to a scalar.
                    getters[0] = _parent.GetGetter(matrixColumnIndexGetter, matrixRowIndexGetter);
                }
                return getters;
            }

            DataViewRow ISchemaBoundRowMapper.GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
            {
                var activeArray = Utils.BuildArray(OutputSchema.Count, activeColumns);
                var getters = CreateGetter(input, activeArray);
                return new SimpleRow(OutputSchema, input, getters);
            }

            public ISchemaBindableMapper Bindable => _parent;
        }
    }

    /// <summary>
    /// Trains a <see cref="MatrixFactorizationModelParameters"/>. It factorizes the training matrix into the product of two low-rank matrices.
    /// </summary>
    public sealed class MatrixFactorizationPredictionTransformer : PredictionTransformerBase<MatrixFactorizationModelParameters>
    {
        internal const string LoaderSignature = "MaFactPredXf";
        internal string MatrixColumnIndexColumnName { get; }
        internal string MatrixRowIndexColumnName { get; }
        internal DataViewType MatrixColumnIndexColumnType { get; }
        internal DataViewType MatrixRowIndexColumnType { get; }

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
        internal MatrixFactorizationPredictionTransformer(IHostEnvironment env, MatrixFactorizationModelParameters model, DataViewSchema trainSchema,
            string matrixColumnIndexColumnName, string matrixRowIndexColumnName, string scoreColumnNameSuffix = "")
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MatrixFactorizationPredictionTransformer)), model, trainSchema)
        {
            Host.CheckNonEmpty(matrixColumnIndexColumnName, nameof(matrixRowIndexColumnName));
            Host.CheckNonEmpty(matrixColumnIndexColumnName, nameof(matrixRowIndexColumnName));

            MatrixColumnIndexColumnName = matrixColumnIndexColumnName;
            MatrixRowIndexColumnName = matrixRowIndexColumnName;

            if (!trainSchema.TryGetColumnIndex(MatrixColumnIndexColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(MatrixColumnIndexColumnName), "matrixColumnIndex", MatrixColumnIndexColumnName);
            MatrixColumnIndexColumnType = trainSchema[xCol].Type;
            if (!trainSchema.TryGetColumnIndex(MatrixRowIndexColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(yCol), "matrixRowIndex", MatrixRowIndexColumnName);
            MatrixRowIndexColumnType = trainSchema[yCol].Type;

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
        private MatrixFactorizationPredictionTransformer(IHostEnvironment host, ModelLoadContext ctx)
            : base(Contracts.CheckRef(host, nameof(host)).Register(nameof(MatrixFactorizationPredictionTransformer)), ctx)
        {
            // *** Binary format ***
            // <base info>
            // string: the column name of matrix's column ids.
            // string: the column name of matrix's row ids.

            MatrixColumnIndexColumnName = ctx.LoadString();
            MatrixRowIndexColumnName = ctx.LoadString();

            if (!TrainSchema.TryGetColumnIndex(MatrixColumnIndexColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(MatrixColumnIndexColumnName), "matrixColumnIndex", MatrixColumnIndexColumnName);
            MatrixColumnIndexColumnType = TrainSchema[xCol].Type;

            if (!TrainSchema.TryGetColumnIndex(MatrixRowIndexColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(MatrixRowIndexColumnName), "matrixRowIndex", MatrixRowIndexColumnName);
            MatrixRowIndexColumnType = TrainSchema[yCol].Type;

            BindableMapper = ScoreUtils.GetSchemaBindableMapper(Host, Model);

            var schema = GetSchema();
            var args = new GenericScorer.Arguments { Suffix = "" };
            Scorer = new GenericScorer(Host, args, new EmptyDataView(Host, TrainSchema), BindableMapper.Bind(Host, schema), schema);
        }

        /// <summary>
        /// Schema propagation for transformers.
        /// Returns the output schema of the data, if the input schema is like the one provided.
        /// </summary>
        public override DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            if (!inputSchema.TryGetColumnIndex(MatrixColumnIndexColumnName, out int xCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "matrixColumnIndex", MatrixColumnIndexColumnName);
            if (!inputSchema.TryGetColumnIndex(MatrixRowIndexColumnName, out int yCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "matrixRowIndex", MatrixRowIndexColumnName);

            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        private protected override void SaveModel(ModelSaveContext ctx)
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
