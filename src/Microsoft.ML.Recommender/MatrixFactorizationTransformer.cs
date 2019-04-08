using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Recommender;

namespace Microsoft.ML.Recommender
{
    public sealed class MatrixFactorizationTransformer : RowToRowTransformerBase
    {
        internal const string Summary = "Transforms to map row/column index to its latent representations and column/row indexes with similar latent representations.";
        internal const string UserName = "Matrix Factorization Transform";
        internal const string ShortName = "MaFaTrans";
        internal const string LoaderSignature = "MatrixFactorizationTransform";

        public class Options : MatrixFactorizationTrainer.Options
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Output column name of the row index's latent representation.")]
            public string ColumnIndexLatentOutputColumnName = nameof(ColumnIndexLatentOutputColumnName).Replace("ColumnName", "");

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Output column name of the row index's latent representation.")]
            public string RowIndexLatentOutputColumnName = nameof(RowIndexLatentOutputColumnName).Replace("ColumnName", "");

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "This is an output column's name. The associated output column carries row indexes whose " +
                "latent representations are similar to that of the given input column index.")]
            public string SimilarColumnIndexesOutputColumnName = nameof(SimilarColumnIndexesOutputColumnName).Replace("ColumnName", "");

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "This is an output column's name. The associated output column carries column indexes whose " +
                "latent representations are similar to that of the given input row index.")]
            public string SimilarRowIndexesOutputColumnName = nameof(SimilarRowIndexesOutputColumnName).Replace("ColumnName", "");

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The number of similar indexes.")]
            public int SimilarIndexCount = 3;

            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "Whether to include known index pairs.")]
            public bool ExcludeKnownIndexPairs = true;
        }

        private readonly string _matrixRowIndexColumnName;
        private readonly string _matrixColumnIndexColumnName;
        private readonly string _labelColumnName;

        /// <summary>
        /// Vector length of columns called <see cref="_similarRowIndexesOutputColumnName"/> and <see cref="_similarColumnIndexesOutputColumnName"/>.
        /// </summary>
        private readonly int _similarIndexCount;

        private readonly bool _excludeKnownIndexPairs;

        /// <summary>
        /// Column name of <see cref="RowIndexLatentOutputColumnType"/>.
        /// </summary>
        private readonly string _rowIndexLatentOutputColumnName;

        /// <summary>
        /// Column name of <see cref="ColumnIndexLatentOutputColumnType"/>.
        /// </summary>
        private readonly string _columnIndexLatentOutputColumnName;

        /// <summary>
        /// Column name of <see cref="SimilarRowIndexesOutputColumnType"/>.
        /// </summary>
        private readonly string _similarRowIndexesOutputColumnName;

        /// <summary>
        /// Column name of <see cref="SimilarColumnIndexesOutputColumnType"/>.
        /// </summary>
        private readonly string _similarColumnIndexesOutputColumnName;

        /// <summary>
        /// The underlying matrix factorization model.
        /// </summary>
        private MatrixFactorizationModelParameters _submodel;

        public MatrixFactorizationModelParameters SubModel => _submodel;

        private uint[] _similarRowsAtColumn;
        private uint[] _similarColumnsAtRow;

        private DataViewType RowIndexLatentOutputColumnType => new VectorDataViewType(NumberDataViewType.Single, _submodel.ApproximationRank);
        private DataViewType ColumnIndexLatentOutputColumnType => new VectorDataViewType(NumberDataViewType.Single, _submodel.ApproximationRank);
        private DataViewType SimilarRowIndexesOutputColumnType => new VectorDataViewType((KeyDataViewType)_submodel.MatrixRowIndexType, _similarIndexCount);
        private DataViewType SimilarColumnIndexesOutputColumnType => new VectorDataViewType((KeyDataViewType)_submodel.MatrixColumnIndexType, _similarIndexCount);

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "MaFaTran",
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
            loaderAssemblyName: typeof(MatrixFactorizationTransformer).Assembly.FullName);
        }

        private void CheckInitialConfiguration()
        {
            // Output column names cannot be null.
            Host.CheckValue(_columnIndexLatentOutputColumnName, nameof(_columnIndexLatentOutputColumnName), "Cannot be null");
            Host.CheckValue(_rowIndexLatentOutputColumnName, nameof(_rowIndexLatentOutputColumnName), "Cannot be null");
            Host.CheckValue(_similarColumnIndexesOutputColumnName, nameof(_similarColumnIndexesOutputColumnName), "Cannot be null");
            Host.CheckValue(_similarRowIndexesOutputColumnName, nameof(_similarRowIndexesOutputColumnName), "Cannot be null");

            // Output column names must be unique.
            var outputColumnNames = new HashSet<string>(GetNewColumnNames);
            var errMsg = string.Format("Output column names {0} conflict", outputColumnNames.ToList());
            Host.Check(outputColumnNames.Count == 4, errMsg);
        }

        internal MatrixFactorizationTransformer(IHostEnvironment env, Options options) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MatrixFactorizationModelParameters)))
        {
            // Input column names.
            _matrixColumnIndexColumnName = options.MatrixColumnIndexColumnName;
            _matrixRowIndexColumnName = options.MatrixRowIndexColumnName;
            _labelColumnName = options.LabelColumnName;

            _similarIndexCount = options.SimilarIndexCount;
            _excludeKnownIndexPairs = options.ExcludeKnownIndexPairs;
            _columnIndexLatentOutputColumnName = options.ColumnIndexLatentOutputColumnName;
            _rowIndexLatentOutputColumnName = options.RowIndexLatentOutputColumnName;
            _similarColumnIndexesOutputColumnName = options.SimilarColumnIndexesOutputColumnName;
            _similarRowIndexesOutputColumnName = options.SimilarRowIndexesOutputColumnName;

            CheckInitialConfiguration();

        }

        internal void Fit(IDataView data, Options options)
        {
            // Do matrix factorization and extract the trained model.
            var trainer = new MatrixFactorizationTrainer(Host, options);
            _submodel = trainer.Fit(data).Model;

            // In the training matrix, row indexes can be found at the i-th column are stored at knownIndexesAtColumns[i].
            var knownRowsAtColumn = new HashSet<uint>[_submodel.NumberOfColumns].Select(set => new HashSet<uint>()).ToArray();

            // In the training matrix, column indexes can be found at the i-th row are stored at knownIndexesAtRows[i].
            var knownColumnsAtRow = new HashSet<uint>[_submodel.NumberOfRows].Select(set => new HashSet<uint>()).ToArray();

            // Fill knownRowsAtColumn and knowColumnsAtRow if know row-column pairs should be excluded from the similar row-column
            // pairs identified by this transform. This is useful when user only wants to discover relations not told by the training
            // data.
            if (options.ExcludeKnownIndexPairs)
                PrepareKnownIndexPairs(data, options.MatrixColumnIndexColumnName, options.MatrixRowIndexColumnName,
                    _submodel.NumberOfColumns, _submodel.NumberOfRows, ref knownRowsAtColumn, ref knownColumnsAtRow);

            // Shared buffer to store latent vector.
            var queryVector = new float[_submodel.ApproximationRank];

            // Column indexes with latent representations similar to the i-th row's are stored at knownSimilarColumnIndexes[i].
            var similarColumnsAtRow = new List<uint>();
            for (int u = 0; u < _submodel.NumberOfRows; ++u)
            {
                // Extract the u-th row's latent vector.
                int pos = u * _submodel.ApproximationRank;
                for (int k = 0; k < _submodel.ApproximationRank; ++k)
                    queryVector[k] = _submodel.LeftFactorMatrix[pos + k];

                // Find u-th row's similar columns' indexes.
                similarColumnsAtRow.AddRange(ComputeSimilarIndexes(_similarIndexCount, queryVector, _submodel.RightFactorMatrix, knownColumnsAtRow[u]));
            }
            _similarColumnsAtRow = similarColumnsAtRow.ToArray();

            // Row indexes similar to the i-th row is stored at knownIndexesAtRows[i].
            var similarRowsAtColumn = new List<uint>();
            for (int v = 0; v < _submodel.NumberOfColumns; ++v)
            {
                // Extract the v-th column's latent vector.
                int pos = v * _submodel.ApproximationRank;
                for (int k = 0; k < _submodel.ApproximationRank; ++k)
                    queryVector[k] = _submodel.RightFactorMatrix[pos + k];

                // Find v-th column's similar columns.
                similarRowsAtColumn.AddRange(ComputeSimilarIndexes(_similarIndexCount, queryVector, _submodel.LeftFactorMatrix, knownRowsAtColumn[v]));
            }
            _similarRowsAtColumn = similarRowsAtColumn.ToArray();
        }

        /// <summary>
        /// This function extracts
        ///   (1) column indexes at one row, and
        ///   (2) row indexes at one column.
        /// The row indexes at the v-th column will be stored in <paramref name="knownRowsAtColumn"/>[v].
        /// The column indexes at the u-th row will be stored in <paramref name="knownColumnsAtRow"/>[u].
        /// </summary>
        private static void PrepareKnownIndexPairs(IDataView data, string matrixColumnName, string matrixRowName, int numberOfColumns, int numberOfRows,
            ref HashSet<uint>[] knownRowsAtColumn, ref HashSet<uint>[] knownColumnsAtRow)
        {
            var matrixColumnIndexColumn = data.Schema[matrixColumnName];
            var matrixRowIndexColumn = data.Schema[matrixRowName];
            var cursor = data.GetRowCursor(matrixColumnIndexColumn, matrixRowIndexColumn);

            var matrixColumnIndexGetter = RowCursorUtils.GetGetterAs<uint>(NumberDataViewType.UInt32, cursor, matrixColumnIndexColumn.Index);
            var matrixRowIndexGetter = RowCursorUtils.GetGetterAs<uint>(NumberDataViewType.UInt32, cursor, matrixRowIndexColumn.Index);

            // Buffer variable.
            uint matrixColumnIndex = 0u;
            // Buffer variable.
            uint matrixRowIndex = 0u;
            // Scan through all matrix elements.
            while (cursor.MoveNext())
            {
                matrixColumnIndexGetter(ref matrixColumnIndex);
                matrixRowIndexGetter(ref matrixRowIndex);
                if (matrixColumnIndex == 0 || matrixRowIndex == 0)
                    continue;
                knownRowsAtColumn[matrixColumnIndex - 1].Add(matrixRowIndex - 1);
                knownColumnsAtRow[matrixRowIndex - 1].Add(matrixColumnIndex - 1);
            }
        }

        /// <summary>
        /// Let <paramref name="queryVector"/> be p and <paramref name="latentMatrix"/> be Q = [q_1, ..., q_n].
        /// This function returns q_v, v = 1, ..., <paramref name="selectedCount"/> which can make largest inner products with p.
        /// The returned value may not contain any index in <paramref name="bannedIndexes"/>.
        /// </summary>
        private static uint[] ComputeSimilarIndexes(int selectedCount, IReadOnlyList<float> queryVector, IReadOnlyList<float> latentMatrix, HashSet<uint> bannedIndexes)
        {
            Contracts.Assert(selectedCount > 0);
            Contracts.AssertValue(queryVector);
            Contracts.AssertValue(latentMatrix);

            // Length of latent vector.
            int d = queryVector.Count;
            // Q is a flattened d-by-n matrix.
            int n = latentMatrix.Count / d;

            // Compute scores.
            var scores = new float[n];
            for (int v = 0; v < n; ++v)
            {
                var pos = v * d;
                for (int k = 0; k < d; ++k)
                    scores[v] += queryVector[k] * latentMatrix[pos + k];
            }

            // Filter out banned indexes and then sort remained indexes by their scores.
            var indexes = Enumerable.Range(0, n).Where(index => !bannedIndexes.Contains((uint)index)).
                OrderByDescending(index => scores[index]).ToArray();

            var selectedIndexes = new uint[selectedCount];
            for (int i = 0; i < selectedCount; ++i)
            {
                if (i > scores.Length)
                    break;
                // Key values are 1-based indexes.
                selectedIndexes[i] = (uint)indexes[i] + 1;
            }

            return selectedIndexes;
        }

        private MatrixFactorizationTransformer(IHostEnvironment env, ModelLoadContext ctx) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(MatrixFactorizationModelParameters)))
        {
            ctx.CheckAtModel(GetVersionInfo());

            _matrixColumnIndexColumnName = ctx.Reader.ReadString();
            _matrixRowIndexColumnName = ctx.Reader.ReadString();
            _similarIndexCount = ctx.Reader.ReadInt32();
            _excludeKnownIndexPairs = ctx.Reader.ReadBoolean();
            _columnIndexLatentOutputColumnName = ctx.Reader.ReadString();
            _rowIndexLatentOutputColumnName = ctx.Reader.ReadString();
            _similarColumnIndexesOutputColumnName = ctx.Reader.ReadString();
            _similarRowIndexesOutputColumnName = ctx.Reader.ReadString();

            CheckInitialConfiguration();

            _submodel = new MatrixFactorizationModelParameters(env, ctx);

            _similarColumnsAtRow = Utils.ReadUIntArray(ctx.Reader, checked(_submodel.NumberOfRows * _submodel.ApproximationRank));
            _similarRowsAtColumn = Utils.ReadUIntArray(ctx.Reader, checked(_submodel.NumberOfColumns * _submodel.ApproximationRank));
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.Write(_matrixColumnIndexColumnName);
            ctx.Writer.Write(_matrixRowIndexColumnName);
            ctx.Writer.Write(_similarIndexCount);
            ctx.Writer.Write(_excludeKnownIndexPairs);
            ctx.Writer.Write(_columnIndexLatentOutputColumnName);
            ctx.Writer.Write(_rowIndexLatentOutputColumnName);
            ctx.Writer.Write(_similarColumnIndexesOutputColumnName);
            ctx.Writer.Write(_similarRowIndexesOutputColumnName);
            ((ICanSaveModel)_submodel).Save(ctx);
            Utils.WriteUIntStream(ctx.Writer, _similarColumnsAtRow);
            Utils.WriteUIntStream(ctx.Writer, _similarRowsAtColumn);
        }

        private string[] GetNewColumnNames => new[] { _columnIndexLatentOutputColumnName, _rowIndexLatentOutputColumnName,
                _similarColumnIndexesOutputColumnName, _similarRowIndexesOutputColumnName };

        private DataViewType[] GetNewColumnTypes => new[] { RowIndexLatentOutputColumnType, ColumnIndexLatentOutputColumnType, SimilarRowIndexesOutputColumnType, SimilarColumnIndexesOutputColumnType };

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
        {
            return new MatrixFactorizationMapper(Host, this, schema);
        }

        private class MatrixFactorizationMapper : MapperBase
        {
            private MatrixFactorizationTransformer _parent;

            internal MatrixFactorizationMapper(IHost host, MatrixFactorizationTransformer parent, DataViewSchema inputSchema) :
                base(host, inputSchema, parent)
            {
                Contracts.AssertValue(parent);
                _parent = parent;
            }

            /// <summary>
            /// Returns columns generated by <see cref="MatrixFactorizationMapper"/>.
            /// </summary>
            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var outputColumns = new DataViewSchema.DetachedColumn[4];

                for (int i = 0; i < _parent.GetNewColumnNames.Count() ; ++i)
                    outputColumns[i] = new DataViewSchema.DetachedColumn(_parent.GetNewColumnNames[i], _parent.GetNewColumnTypes[i]);

                return outputColumns;
            }

            /// <summary>
            /// Helper function shared by <see cref="MakeRowLatentVectorGetter(DataViewRow)"/> and <see cref="MakeColumnLatentVectorGetter(DataViewRow)"/>.
            /// </summary>
            private Delegate MakeLatentVectorGetter(DataViewRow input, DataViewSchema.Column indexColumn, IReadOnlyList<float> factorMatrix)
            {
                // Get the getter of the row index.
                var indexGetter = RowCursorUtils.GetGetterAs<uint>(NumberDataViewType.UInt32, input, indexColumn.Index);

                // Buffer used to store index being mapped to a latent vector.
                uint index = 0;

                // Rank of matrix factorization trained.
                var rank = _parent._submodel.ApproximationRank;

                // Starting position of a index's latent vector in the factor matrix.
                var position = 0;

                ValueGetter<VBuffer<float>> del =
                    (ref VBuffer<float> value) =>
                    {
                        var editor = VBufferEditor.Create(ref value, rank);
                        // Find the index.
                        indexGetter(ref index);
                        // We compute the starting position of the found index's latent vector and
                        // then the latent vector is copied to the input buffer.
                        position = (int)(index - 1) * rank;
                        for (int i = 0; i < rank; ++i)
                            editor.Values[i] = factorMatrix[position + i];
                        value = editor.Commit();
                    };

                return del;
            }

            private Delegate MakeRowLatentVectorGetter(DataViewRow input)
            {
                // Get the getter of the row index.
                var indexColumn = input.Schema[_parent._matrixRowIndexColumnName];
                // Row indexes' latent vectors, which is a flattened matrix.
                var leftFactorMatrix = _parent._submodel.LeftFactorMatrix;

                return MakeLatentVectorGetter(input, indexColumn, leftFactorMatrix);
            }

            private Delegate MakeColumnLatentVectorGetter(DataViewRow input)
            {
                // Get the getter of the column index.
                var indexColumn = input.Schema[_parent._matrixColumnIndexColumnName];
                // Column indexes' latent vectors, which is a flattened matrix.
                var rightFactorMatrix = _parent._submodel.RightFactorMatrix;

                return MakeLatentVectorGetter(input, indexColumn, rightFactorMatrix);
            }

            private Delegate MakeSimilarIndexGetter(DataViewRow input, DataViewSchema.Column column, IReadOnlyList<uint> candidates)
            {
                // Get the getter of the index.
                var indexGetter = RowCursorUtils.GetGetterAs<uint>(NumberDataViewType.UInt32, input, column.Index);

                uint index = 0;
                int position = 0;
                ValueGetter<VBuffer<uint>> del =
                    (ref VBuffer<uint> value) =>
                    {
                        var editor = VBufferEditor.Create(ref value, _parent._similarIndexCount);
                        indexGetter(ref index);
                        // Because layout of _parent._similarColumnIndexes is [column ids similar to row 0, column ids simialr to row 1, ...],
                        // the starting position of the u-th row's similar columns start at u * (# of similar indexes per row).
                        position = (int)(index - 1) * _parent._similarIndexCount;
                        for (int i = 0; i < _parent._similarIndexCount; ++i)
                            editor.Values[i] = candidates[position + i];
                        value = editor.Commit();
                    };
                return del;
            }

            /// <summary>
            /// Map matrix-row index to its similar columns' indexes.
            /// </summary>
            private Delegate MakeSimilarColumnIndexesGetter(DataViewRow input)
            {
                return MakeSimilarIndexGetter(input, input.Schema[_parent._matrixRowIndexColumnName], _parent._similarColumnsAtRow);
            }

            /// <summary>
            /// Map matrix-column index to its similar rows' indexes.
            /// </summary>
            private Delegate MakeSimilarRowIndexesGetter(DataViewRow input)
            {
                return MakeSimilarIndexGetter(input, input.Schema[_parent._matrixColumnIndexColumnName], _parent._similarRowsAtColumn);
            }

            /// <summary>
            /// Create getters for columns generated by <see cref="MatrixFactorizationMapper"/>.
            /// </summary>
            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.Assert(0 <= iinfo && iinfo < _parent.GetNewColumnNames.Count());

                switch (iinfo)
                {
                    case 0:
                        return MakeColumnLatentVectorGetter(input);
                    case 1:
                        return MakeRowLatentVectorGetter(input);
                    case 2:
                        return MakeSimilarRowIndexesGetter(input);
                    case 3:
                        return MakeSimilarColumnIndexesGetter(input);
                    default:
                        return null;
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                var indexesOfUsedInputColumns = new HashSet<int>();
                indexesOfUsedInputColumns.Add(InputSchema[_parent._matrixColumnIndexColumnName].Index);
                indexesOfUsedInputColumns.Add(InputSchema[_parent._matrixRowIndexColumnName].Index);
                indexesOfUsedInputColumns.Add(InputSchema[_parent._labelColumnName].Index);

                return i => indexesOfUsedInputColumns.Contains(i);
            }

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    /// <summary>
    /// A class implementing the estimator interface of the <see cref="MatrixFactorizationTransformer"/>.
    /// </summary>
    public sealed class MatrixFactorizationEstimator : IEstimator<MatrixFactorizationTransformer>
    {
        private MatrixFactorizationTransformer.Options _options;
        private MatrixFactorizationTransformer _transformer;
        private IHost _host;

        internal MatrixFactorizationEstimator(IHostEnvironment env,
                string columnIndexLatentOutputColumnName,
                string rowIndexLatentOutputColumnName,
                string similarColumnIndexesOutputColumnName,
                string similarRowIndexesOutputColumnName,
                string labelColumnName,
                string matrixColumnIndexColumnName,
                string matrixRowIndexColumnName,
                int similarIndexCount,
                bool excludeKnownIndexPairs,
                int approximationRank = MatrixFactorizationTrainer.Defaults.ApproximationRank,
                double learningRate = MatrixFactorizationTrainer.Defaults.LearningRate,
                int numberOfIterations = MatrixFactorizationTrainer.Defaults.NumIterations)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(MatrixFactorizationEstimator));

            _options = new MatrixFactorizationTransformer.Options
            {
                ColumnIndexLatentOutputColumnName = columnIndexLatentOutputColumnName,
                RowIndexLatentOutputColumnName = rowIndexLatentOutputColumnName,
                SimilarColumnIndexesOutputColumnName = similarColumnIndexesOutputColumnName,
                SimilarRowIndexesOutputColumnName = similarRowIndexesOutputColumnName,
                LabelColumnName = labelColumnName,
                MatrixColumnIndexColumnName = matrixColumnIndexColumnName,
                MatrixRowIndexColumnName = matrixRowIndexColumnName,
                SimilarIndexCount = similarIndexCount,
                ExcludeKnownIndexPairs = excludeKnownIndexPairs,
                ApproximationRank = approximationRank,
                LearningRate = learningRate
            };
            _transformer = new MatrixFactorizationTransformer(env, _options);
        }

        internal MatrixFactorizationEstimator(IHostEnvironment env, MatrixFactorizationTransformer.Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            Contracts.CheckValue(options, nameof(options));
            _host = env.Register(nameof(MatrixFactorizationEstimator));
            _options = options;
            _transformer = new MatrixFactorizationTransformer(env, _options);
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var result = inputSchema.ToDictionary(x => x.Name);

            result[_options.ColumnIndexLatentOutputColumnName] = new SchemaShape.Column(_options.ColumnIndexLatentOutputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);
            result[_options.RowIndexLatentOutputColumnName] = new SchemaShape.Column(_options.RowIndexLatentOutputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);
            result[_options.SimilarColumnIndexesOutputColumnName] = new SchemaShape.Column(_options.SimilarColumnIndexesOutputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.UInt32, true);
            result[_options.SimilarRowIndexesOutputColumnName] = new SchemaShape.Column(_options.SimilarRowIndexesOutputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.UInt32, true);

            return new SchemaShape(result.Values);
        }

        public MatrixFactorizationTransformer Fit(IDataView input)
        {
            _transformer.Fit(input, _options);
            return _transformer;
        }
    }
}
