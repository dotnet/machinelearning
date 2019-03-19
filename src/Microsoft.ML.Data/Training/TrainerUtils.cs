// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// Options for creating a <see cref="TrainingCursorBase"/> from a <see cref="RoleMappedData"/> with specified standard columns active.
    /// </summary>
    [Flags]
    [BestFriend]
    internal enum CursOpt : uint
    {
        Weight = 0x01,
        Group = 0x02,
        Id = 0x04,
        Label = 0x08,
        Features = 0x10,

        // Row filtering options.
        AllowBadWeights = 0x0100,
        AllowBadGroups = 0x0200,
        AllowBadLabels = 0x0800,
        AllowBadFeatures = 0x1000,

        // Bad to the bone.
        AllowBadEverything = AllowBadWeights | AllowBadGroups | AllowBadLabels | AllowBadFeatures,

        AllWeights = Weight | AllowBadWeights,
        AllGroups = Group | AllowBadGroups,
        AllLabels = Label | AllowBadLabels,
        AllFeatures = Features | AllowBadFeatures,
    }

    [BestFriend]
    internal static class TrainerUtils
    {
        /// <summary>
        /// Check for a standard (known-length vector of float) feature column.
        /// </summary>
        public static void CheckFeatureFloatVector(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            if (!data.Schema.Feature.HasValue)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a feature column.");
            var col = data.Schema.Feature.Value;
            Contracts.Assert(!col.IsHidden);
            if (!(col.Type is VectorType vecType && vecType.Size > 0 && vecType.ItemType == NumberDataViewType.Single))
                throw Contracts.ExceptParam(nameof(data), "Training feature column '{0}' must be a known-size vector of R4, but has type: {1}.", col.Name, col.Type);
        }

        /// <summary>
        /// Check for a standard (known-length vector of float) feature column and determine its length.
        /// </summary>
        public static void CheckFeatureFloatVector(this RoleMappedData data, out int length)
        {
            CheckFeatureFloatVector(data);

            // If the above function is generalized, this needs to be as well.
            Contracts.AssertValue(data);
            Contracts.Assert(data.Schema.Feature.HasValue);
            var col = data.Schema.Feature.Value;
            Contracts.Assert(!col.IsHidden);
            var colType = col.Type as VectorType;
            Contracts.Assert(colType != null && colType.IsKnownSize);
            Contracts.Assert(colType.ItemType == NumberDataViewType.Single);
            length = colType.Size;
        }

        /// <summary>
        /// Check for a standard binary classification label.
        /// </summary>
        public static void CheckBinaryLabel(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            if (!data.Schema.Label.HasValue)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a label column.");
            var col = data.Schema.Label.Value;
            Contracts.Assert(!col.IsHidden);
            if (col.Type != BooleanDataViewType.Instance && col.Type != NumberDataViewType.Single && col.Type != NumberDataViewType.Double && !(col.Type is KeyType keyType && keyType.Count == 2))
            {
                KeyType colKeyType = col.Type as KeyType;
                if (colKeyType != null)
                {
                    if (colKeyType.Count == 1)
                    {
                        throw Contracts.ExceptParam(nameof(data),
                            "The label column '{0}' of the training data has only one class. Two classes are required for binary classification.",
                            col.Name);
                    }
                    else if (colKeyType.Count > 2)
                    {
                        throw Contracts.ExceptParam(nameof(data),
                            "The label column '{0}' of the training data has more than two classes. Only two classes are allowed for binary classification.",
                            col.Name);
                    }
                }
                throw Contracts.ExceptParam(nameof(data),
                    "The label column '{0}' of the training data has a data type not suitable for binary classification: {1}. Type must be Bool, R4, R8 or Key with two classes.",
                    col.Name, col.Type);
            }
        }

        /// <summary>
        /// Check for a standard regression label.
        /// </summary>
        public static void CheckRegressionLabel(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            if (!data.Schema.Label.HasValue)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a label column.");
            var col = data.Schema.Label.Value;
            Contracts.Assert(!data.Schema.Schema[col.Index].IsHidden);
            if (col.Type != NumberDataViewType.Single && col.Type != NumberDataViewType.Double)
            {
                throw Contracts.ExceptParam(nameof(data),
                    "Training label column '{0}' type isn't suitable for regression: {1}. Type must be R4 or R8.", col.Name, col.Type);
            }
        }

        /// <summary>
        /// Check for a standard multi-class label and determine its cardinality. If the column is a
        /// key type, it must have known cardinality. For other numeric types, this scans the data
        /// to determine the cardinality.
        /// </summary>
        public static void CheckMulticlassLabel(this RoleMappedData data, out int count)
        {
            Contracts.CheckValue(data, nameof(data));

            if (!data.Schema.Label.HasValue)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a label column.");
            var col = data.Schema.Label.Value;
            Contracts.Assert(!col.IsHidden);
            if (col.Type is KeyType keyType && keyType.Count > 0)
            {
                if (keyType.Count >= Utils.ArrayMaxSize)
                    throw Contracts.ExceptParam(nameof(data), "Maximum label is too large for multi-class: {0}.", keyType.Count);
                count = (int)keyType.Count;
                return;
            }

            // REVIEW: Support other numeric types.
            if (col.Type != NumberDataViewType.Single && col.Type != NumberDataViewType.Double)
                throw Contracts.ExceptParam(nameof(data), "Training label column '{0}' type is not valid for multi-class: {1}. Type must be R4 or R8.", col.Name, col.Type);

            int max = -1;
            using (var cursor = new FloatLabelCursor(data))
            {
                while (cursor.MoveNext())
                {
                    int cls = (int)cursor.Label;
                    if (cls != cursor.Label || cls < 0)
                    {
                        throw Contracts.ExceptParam(nameof(data),
                            "Training label column '{0}' contains invalid values for multi-class: {1}.", col.Name, cursor.Label);
                    }
                    if (max < cls)
                        max = cls;
                }
            }

            if (max < 0)
                throw Contracts.ExceptParam(nameof(data), "Training label column '{0}' contains no valid values for multi-class.", col.Name);
            // REVIEW: Should we impose some smaller limit on the max?
            if (max >= Utils.ArrayMaxSize)
                throw Contracts.ExceptParam(nameof(data), "Maximum label is too large for multi-class: {0}.", max);

            count = max + 1;
        }

        /// <summary>
        /// Check for a standard regression label.
        /// </summary>
        public static void CheckMultiOutputRegressionLabel(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            if (!data.Schema.Label.HasValue)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a label column.");
            var col = data.Schema.Label.Value;
            Contracts.Assert(!col.IsHidden);
            if (!(col.Type is VectorType vectorType
                && vectorType.IsKnownSize
                && vectorType.ItemType == NumberDataViewType.Single))
                throw Contracts.ExceptParam(nameof(data), "Training label column '{0}' must be a known-size vector of R4, but has type: {1}.", col.Name, col.Type);
        }

        public static void CheckOptFloatWeight(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            if (!data.Schema.Weight.HasValue)
                return;
            var col = data.Schema.Weight.Value;
            Contracts.Assert(!col.IsHidden);
            if (col.Type != NumberDataViewType.Single && col.Type != NumberDataViewType.Double)
                throw Contracts.ExceptParam(nameof(data), "Training weight column '{0}' must be of floating point numeric type, but has type: {1}.", col.Name, col.Type);
        }

        public static void CheckOptGroup(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            if (!data.Schema.Group.HasValue)
                return;
            var col = data.Schema.Group.Value;
            Contracts.Assert(!col.IsHidden);
            if (col.Type is KeyType)
                return;
            throw Contracts.ExceptParam(nameof(data), "Training group column '{0}' type is invalid: {1}. Must be Key type.", col.Name, col.Type);
        }

        private static IEnumerable<DataViewSchema.Column> CreatePredicate(RoleMappedData data, CursOpt opt, IEnumerable<int> extraCols)
        {
            Contracts.AssertValue(data);
            Contracts.AssertValueOrNull(extraCols);

            var columns = extraCols == null ?
                new List<DataViewSchema.Column>() :
                data.Data.Schema.Where(c => extraCols.Contains(c.Index)).ToList();

            if ((opt & CursOpt.Label) != 0 && data.Schema.Label.HasValue)
                columns.Add(data.Schema.Label.Value);
            if ((opt & CursOpt.Features) != 0 && data.Schema.Feature.HasValue)
                columns.Add(data.Schema.Feature.Value);
            if ((opt & CursOpt.Weight) != 0 && data.Schema.Weight.HasValue)
                columns.Add(data.Schema.Weight.Value);
            if ((opt & CursOpt.Group) != 0 && data.Schema.Group.HasValue)
                columns.Add(data.Schema.Group.Value);
            return columns;
        }

        /// <summary>
        /// Create a row cursor for the RoleMappedData with the indicated standard columns active.
        /// This does not verify that the columns exist, but merely activates the ones that do exist.
        /// </summary>
        public static DataViewRowCursor CreateRowCursor(this RoleMappedData data, CursOpt opt, Random rand, IEnumerable<int> extraCols = null)
            => data.Data.GetRowCursor(CreatePredicate(data, opt, extraCols), rand);

        /// <summary>
        /// Create a row cursor set for the <see cref="RoleMappedData"/> with the indicated standard columns active.
        /// This does not verify that the columns exist, but merely activates the ones that do exist.
        /// </summary>
        public static DataViewRowCursor[] CreateRowCursorSet(this RoleMappedData data,
            CursOpt opt, int n, Random rand, IEnumerable<int> extraCols = null)
            => data.Data.GetRowCursorSet(CreatePredicate(data, opt, extraCols), n, rand);

        /// <summary>
        /// Get the getter for the feature column, assuming it is a vector of float.
        /// </summary>
        public static ValueGetter<VBuffer<float>> GetFeatureFloatVectorGetter(this DataViewRow row, RoleMappedSchema schema)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(schema.Schema == row.Schema, nameof(schema), "schemas don't match!");
            Contracts.CheckParam(schema.Feature.HasValue, nameof(schema), "Missing feature column");

            return row.GetGetter<VBuffer<float>>(schema.Feature.Value);
        }

        /// <summary>
        /// Get the getter for the feature column, assuming it is a vector of float.
        /// </summary>
        public static ValueGetter<VBuffer<float>> GetFeatureFloatVectorGetter(this DataViewRow row, RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));
            return GetFeatureFloatVectorGetter(row, data.Schema);
        }

        /// <summary>
        /// Get a getter for the label as a float. This assumes that the label column type
        /// has already been validated as appropriate for the kind of training being done.
        /// </summary>
        public static ValueGetter<float> GetLabelFloatGetter(this DataViewRow row, RoleMappedSchema schema)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(schema.Schema == row.Schema, nameof(schema), "schemas don't match!");
            Contracts.CheckParam(schema.Label.HasValue, nameof(schema), "Missing label column");

            return RowCursorUtils.GetLabelGetter(row, schema.Label.Value.Index);
        }

        /// <summary>
        /// Get a getter for the label as a float. This assumes that the label column type
        /// has already been validated as appropriate for the kind of training being done.
        /// </summary>
        public static ValueGetter<float> GetLabelFloatGetter(this DataViewRow row, RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));
            return GetLabelFloatGetter(row, data.Schema);
        }

        /// <summary>
        /// Get the getter for the weight column, or null if there is no weight column.
        /// </summary>
        public static ValueGetter<float> GetOptWeightFloatGetter(this DataViewRow row, RoleMappedSchema schema)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(schema.Schema == row.Schema, "schemas don't match!");

            var col = schema.Weight;
            if (!col.HasValue)
                return null;
            return RowCursorUtils.GetGetterAs<float>(NumberDataViewType.Single, row, col.Value.Index);
        }

        public static ValueGetter<float> GetOptWeightFloatGetter(this DataViewRow row, RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));
            return GetOptWeightFloatGetter(row, data.Schema);
        }

        /// <summary>
        /// Get the getter for the group column, or null if there is no group column.
        /// </summary>
        public static ValueGetter<ulong> GetOptGroupGetter(this DataViewRow row, RoleMappedSchema schema)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(schema.Schema == row.Schema, "schemas don't match!");

            var col = schema.Group;
            if (!col.HasValue)
                return null;
            return RowCursorUtils.GetGetterAs<ulong>(NumberDataViewType.UInt64, row, col.Value.Index);
        }

        public static ValueGetter<ulong> GetOptGroupGetter(this DataViewRow row, RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));
            return GetOptGroupGetter(row, data.Schema);
        }

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the label column for binary classification tasks.
        /// </summary>
        /// <param name="labelColumn">name of the label column</param>
        public static SchemaShape.Column MakeBoolScalarLabel(string labelColumn)
            => new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false);

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the float type columns.
        /// </summary>
        /// <param name="columnName">name of the column</param>
        public static SchemaShape.Column MakeR4ScalarColumn(string columnName)
            => new SchemaShape.Column(columnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false);

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the label column for regression tasks.
        /// </summary>
        /// <param name="columnName">name of the weight column</param>
        public static SchemaShape.Column MakeU4ScalarColumn(string columnName)
        {
            if (columnName == null)
                return default;

            return new SchemaShape.Column(columnName, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true);
        }

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the feature column.
        /// </summary>
        /// <param name="featureColumn">name of the feature column</param>
        public static SchemaShape.Column MakeR4VecFeature(string featureColumn)
            => new SchemaShape.Column(featureColumn, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the weight column.
        /// </summary>
        /// <param name="weightColumn">name of the weight column</param>
        public static SchemaShape.Column MakeR4ScalarWeightColumn(string weightColumn)
        {
            if (weightColumn == null)
                return default;
            return new SchemaShape.Column(weightColumn, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false);
        }

        /// <summary>
        /// This is a shim class to translate the more contemporaneous <see cref="ITrainerEstimator{TTransformer, TPredictor}"/>
        /// style transformers into the older now disfavored <see cref="ITrainer{TPredictor}"/> idiom, for components that still
        /// need to operate via that older mechanism. (Mostly command line invocations, and so on.).
        /// </summary>
        /// <typeparam name="TModel">The type of the new model parameters.</typeparam>
        /// <typeparam name="TPredictor">The type corresponding to the legacy predictor.</typeparam>
        private sealed class TrainerEstimatorToTrainerShim<TModel, TPredictor> : ITrainer<TPredictor>
            where TModel : class, TPredictor
            where TPredictor : IPredictor
        {
            public TrainerInfo Info { get; }
            public PredictionKind PredictionKind { get; }

            private readonly ITrainerEstimator<ISingleFeaturePredictionTransformer<TModel>, TModel> _trainer;
            private readonly IHostEnvironment _env;

            public TrainerEstimatorToTrainerShim(IHostEnvironment env, ITrainerEstimator<ISingleFeaturePredictionTransformer<TModel>, TModel> trainer)
            {
                Contracts.AssertValue(env);
                _env = env;
                _env.AssertValue(trainer);
                _env.Assert(trainer is ITrainer);

                var oldTrainer = (ITrainer)trainer;
                Info = oldTrainer.Info;
                PredictionKind = oldTrainer.PredictionKind;

                _trainer = trainer;
            }

            public TPredictor Train(TrainContext context)
            {
                _env.CheckValue(context, nameof(context));
                // For the purpose of mapping into the estimator, we assume that the input estimator does not have
                // any custom overrides for the column names defined.
                var tschema = context.TrainingSet.Schema;
                var nameMap = new List<(string outName, string inName)>();
                if (tschema.Feature?.Name is string fname && fname != DefaultColumnNames.Features)
                    nameMap.Add((DefaultColumnNames.Features, fname));
                if (tschema.Label?.Name is string lname && lname != DefaultColumnNames.Label)
                    nameMap.Add((DefaultColumnNames.Label, lname));
                if (tschema.Weight?.Name is string wname && wname != DefaultColumnNames.Weight)
                    nameMap.Add((DefaultColumnNames.Weight, wname));
                if (tschema.Group?.Name is string gname && gname != DefaultColumnNames.GroupId)
                    nameMap.Add((DefaultColumnNames.GroupId, gname));
                if (tschema.Group?.Name is string iname && iname != DefaultColumnNames.Item)
                    nameMap.Add((DefaultColumnNames.Item, iname));
                if (tschema.Group?.Name is string uname && uname != DefaultColumnNames.User)
                    nameMap.Add((DefaultColumnNames.User, uname));

                var data = context.TrainingSet.Data;
                if (nameMap.Count > 0)
                {
                    var estimator = new ColumnCopyingEstimator(_env, nameMap.ToArray());
                    data = estimator.Fit(data).Transform(data);
                }
                var predictionTransformer = _trainer.Fit(data);
                var model = predictionTransformer.Model;
                if (model is TPredictor pred)
                    return pred;
                throw _env.Except($"Training resulted in a model of type {model.GetType().Name}.");
            }

            IPredictor ITrainer.Train(TrainContext context) => Train(context);
        }

        /// <summary>
        /// This is a shim for legacy code that takes the more modern <see cref="ITrainerEstimator{TTransformer, TPredictor}"/>
        /// interface, and maps it to the legacy code that wants an <see cref="ITrainer{TPredictor}"/>. The goal should be to
        /// remove reliance on that interface if possible, but this may not be practical in the immediate term, so for the benefit
        /// of scenarios like this we have this convenience function.
        /// </summary>
        /// <typeparam name="T">The trainer estimator type.</typeparam>
        /// <typeparam name="TModel">The type of the model produced by the estimator.</typeparam>
        /// <typeparam name="TPredictor">The type of the predictor to be produced by the predictor.</typeparam>
        /// <param name="env">The host environment.</param>
        /// <param name="trainer">The trainer estimator.</param>
        /// <returns>An implementation of the legacy trainer interface.</returns>
        public static ITrainer<TPredictor> MapTrainerEstimatorToTrainer<T, TModel, TPredictor>(IHostEnvironment env, T trainer)
            where T : ITrainerEstimator<ISingleFeaturePredictionTransformer<TModel>, TModel>, ITrainer
            where TModel : class, TPredictor
            where TPredictor : IPredictor
        {
            return new TrainerEstimatorToTrainerShim<TModel, TPredictor>(env, trainer);
        }
    }

    /// <summary>
    /// This is the base class for a data cursor. Data cursors are specially typed
    /// "convenience" cursor-like objects, less general than a <see cref="DataViewRowCursor"/> but
    /// more convenient for common access patterns that occur in machine learning. For
    /// example, the common idiom of iterating over features/labels/weights while skipping
    /// "bad" features, labels, and weights. There will be two typical access patterns for
    /// users of the cursor. The first is just creation of the cursor using a constructor;
    /// this is best for one-off accesses of the data. The second access pattern, best for
    /// repeated accesses, is to use a cursor factory (usually a nested class of the cursor
    /// class). This keeps track of what filtering options were actually useful.
    /// </summary>
    [BestFriend]
    internal abstract class TrainingCursorBase : IDisposable
    {
        public DataViewRow Row => _cursor;

        private readonly DataViewRowCursor _cursor;
        private readonly Action<CursOpt> _signal;

        public long SkippedRowCount { get; private set; }
        public long KeptRowCount { get; private set; }

        /// <summary>
        /// The base constructor class for the factory-based cursor creation.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="signal">This method is called </param>
        protected TrainingCursorBase(DataViewRowCursor input, Action<CursOpt> signal)
        {
            Contracts.AssertValue(input);
            Contracts.AssertValueOrNull(signal);
            _cursor = input;
            _signal = signal;
        }

        protected static DataViewRowCursor CreateCursor(RoleMappedData data, CursOpt opt, Random rand, params int[] extraCols)
        {
            Contracts.AssertValue(data);
            Contracts.AssertValueOrNull(rand);
            return data.CreateRowCursor(opt, rand, extraCols);
        }

        /// <summary>
        /// This method is called by <see cref="MoveNext"/> in the event we have reached the end
        /// of the cursoring. The intended usage is that it returns what flags will be passed to the signal
        /// delegate of the cursor, indicating what additional options should be specified on subsequent
        /// passes over the data. The base implementation checks if any rows were skipped, and if none were
        /// skipped, it signals the context that it needn't bother with any filtering checks.
        ///
        /// Because the result will be "or"-red, a perfectly acceptable implementation is that this
        /// return the default <see cref="CursOpt"/>, in which case the flags will not ever change.
        ///
        /// If the cursor was created with a signal delegate, the return value of this method will be sent
        /// to that delegate.
        /// </summary>
        protected virtual CursOpt CursoringCompleteFlags()
        {
            return SkippedRowCount == 0 ? CursOpt.AllowBadEverything : default(CursOpt);
        }

        /// <summary>
        /// Calls Cursor.MoveNext() and this.Accept() repeatedly until this.Accept() returns true.
        /// Returns false if Cursor.MoveNext() returns false. If you call Cursor.MoveNext() directly,
        /// also call this.Accept() to fetch the values of the current row. Note that if this.Accept()
        /// returns false, it's possible that not all values were fetched.
        /// </summary>
        public bool MoveNext()
        {
            for (; ; )
            {
                if (!_cursor.MoveNext())
                {
                    if (_signal != null)
                        _signal(CursoringCompleteFlags());
                    return false;
                }
                if (Accept())
                {
                    KeptRowCount++;
                    return true;
                }
                SkippedRowCount++;
            }
        }

        /// <summary>
        /// This fetches and validates values for the standard active columns.
        /// It is called automatically by MoveNext(). Client code should only need
        /// to deal with this if it calls MoveNext() or MoveMany() on the underlying
        /// IRowCursor directly. That is, this is only for very advanced scenarios.
        /// </summary>
        public virtual bool Accept()
        {
            return true;
        }

        public void Dispose()
        {
            _cursor.Dispose();
        }

        /// <summary>
        /// This is the base class for a data cursor factory. The factory is a reusable object,
        /// created with data and cursor options. From external non-implementing users it will
        /// appear to be more or less stateless, but internally it is keeping track of what sorts
        /// of filtering it needs to perform. For example, if we construct the factory with the
        /// option that it needs to filter out rows with bad feature values, but on the first
        /// iteration it is revealed there are no bad feature values, then it would be a complete
        /// waste of time to check on subsequent iterations over the data whether there are bad
        /// feature values again.
        /// </summary>
        public abstract class FactoryBase<TCurs>
            where TCurs : TrainingCursorBase
        {
            private readonly RoleMappedData _data;
            private readonly CursOpt _initOpts;

            private readonly object _lock;
            private CursOpt _opts;

            public RoleMappedData Data => _data;

            protected FactoryBase(RoleMappedData data, CursOpt opt)
            {
                Contracts.CheckValue(data, nameof(data));

                _data = data;
                _opts = _initOpts = opt;
                _lock = new object();
            }

            private void SignalCore(CursOpt opt)
            {
                lock (_lock)
                    _opts |= opt;
            }

            /// <summary>
            /// The typed analog to <see cref="IDataView.GetRowCursor(IEnumerable{DataViewSchema.Column},Random)"/>.
            /// </summary>
            /// <param name="rand">Non-null if we are requesting a shuffled cursor.</param>
            /// <param name="extraCols">The extra columns to activate on the row cursor
            /// in addition to those required by the factory's options.</param>
            /// <returns>The wrapping typed cursor.</returns>
            public TCurs Create(Random rand = null, params int[] extraCols)
            {
                CursOpt opt;
                lock (_lock)
                    opt = _opts;

                var input = _data.CreateRowCursor(opt, rand, extraCols);
                return CreateCursorCore(input, _data, opt, SignalCore);
            }

            /// <summary>
            /// The typed analog to <see cref="IDataView.GetRowCursorSet"/>, this provides a
            /// partitioned cursoring of the data set, appropriate to multithreaded algorithms
            /// that want to consume parallel cursors without any consolidation.
            /// </summary>
            /// <param name="n">Suggested degree of parallelism.</param>
            /// <param name="rand">Non-null if we are requesting a shuffled cursor.</param>
            /// <param name="extraCols">The extra columns to activate on the row cursor
            /// in addition to those required by the factory's options.</param>
            /// <returns>The cursor set. Note that this needn't necessarily be of size
            /// <paramref name="n"/>.</returns>
            public TCurs[] CreateSet(int n, Random rand = null, params int[] extraCols)
            {
                CursOpt opt;
                lock (_lock)
                    opt = _opts;

                // Users of this method will tend to consume the cursors in the set in separate
                // threads,  and so gain benefit from the parallel transformation of the data.
                var inputs = _data.CreateRowCursorSet(opt, n, rand, extraCols);
                Contracts.Assert(Utils.Size(inputs) > 0);

                Action<CursOpt> signal;
                if (inputs.Length > 1)
                    signal = new AndAccumulator(SignalCore, inputs.Length).Signal;
                else
                    signal = SignalCore;

                var res = new TCurs[inputs.Length];
                for (int i = 0; i < res.Length; i++)
                    res[i] = CreateCursorCore(inputs[i], _data, opt, signal);

                return res;
            }

            /// <summary>
            /// Called by both the <see cref="Create"/> and <see cref="CreateSet"/> factory methods. Implementors
            /// should instantiate the particular wrapping cursor.
            /// </summary>
            /// <param name="input">The row cursor we will wrap.</param>
            /// <param name="data">The data from which the row cursor was instantiated.</param>
            /// <param name="opt">The cursor options this row cursor was created with.</param>
            /// <param name="signal">The action that our wrapping cursor will call. Implementors of the cursor
            /// do not usually call it directly, but instead override
            /// <see cref="TrainingCursorBase.CursoringCompleteFlags"/>, whose return value is used to call
            /// this action.</param>
            /// <returns></returns>
            protected abstract TCurs CreateCursorCore(DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal);

            /// <summary>
            /// Accumulates signals from cursors, anding them together. Once it has
            /// all of the information it needs to signal the factory itself, it will
            /// do so.
            /// </summary>
            private sealed class AndAccumulator
            {
                private readonly Action<CursOpt> _signal;
                private readonly int _lim;
                private int _count;
                private CursOpt _opts;

                public AndAccumulator(Action<CursOpt> signal, int lim)
                {
                    Contracts.AssertValue(signal);
                    Contracts.Assert(lim > 0);
                    _signal = signal;
                    _lim = lim;
                    _opts = ~default(CursOpt);
                }

                public void Signal(CursOpt opt)
                {
                    lock (this)
                    {
                        Contracts.Assert(_count < _lim);
                        _opts &= opt;
                        if (++_count == _lim)
                            _signal(_opts);
                    }
                }
            }
        }
    }

    /// <summary>
    /// This supports Weight (float), Group (ulong), and Id (RowId) columns.
    /// </summary>
    [BestFriend]
    internal class StandardScalarCursor : TrainingCursorBase
    {
        private readonly ValueGetter<float> _getWeight;
        private readonly ValueGetter<ulong> _getGroup;
        private readonly ValueGetter<DataViewRowId> _getId;
        private readonly bool _keepBadWeight;
        private readonly bool _keepBadGroup;

        public long BadWeightCount { get; private set; }
        public long BadGroupCount { get; private set; }

        public float Weight;
        public ulong Group;
        public DataViewRowId Id;

        public StandardScalarCursor(RoleMappedData data, CursOpt opt, Random rand = null, params int[] extraCols)
            : this(CreateCursor(data, opt, rand, extraCols), data, opt)
        {
        }

        protected StandardScalarCursor(DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal = null)
            : base(input, signal)
        {
            Contracts.AssertValue(data);

            if ((opt & CursOpt.Weight) != 0)
            {
                _getWeight = Row.GetOptWeightFloatGetter(data);
                _keepBadWeight = (opt & CursOpt.AllowBadWeights) != 0;
            }
            if ((opt & CursOpt.Group) != 0)
            {
                _getGroup = Row.GetOptGroupGetter(data);
                _keepBadGroup = (opt & CursOpt.AllowBadGroups) != 0;
            }
            if ((opt & CursOpt.Id) != 0)
                _getId = Row.GetIdGetter();
            Weight = 1;
            Group = 0;
        }

        protected override CursOpt CursoringCompleteFlags()
        {
            CursOpt opt = base.CursoringCompleteFlags();
            if (BadWeightCount == 0)
                opt |= CursOpt.AllowBadWeights;
            if (BadGroupCount == 0)
                opt |= CursOpt.AllowBadGroups;
            return opt;
        }

        public override bool Accept()
        {
            if (!base.Accept())
                return false;
            if (_getWeight != null)
            {
                _getWeight(ref Weight);
                if (!_keepBadWeight && !(0 < Weight && Weight < float.PositiveInfinity))
                {
                    BadWeightCount++;
                    return false;
                }
            }
            if (_getGroup != null)
            {
                _getGroup(ref Group);
                if (!_keepBadGroup && Group == 0)
                {
                    BadGroupCount++;
                    return false;
                }
            }
            if (_getId != null)
                _getId(ref Id);
            return true;
        }

        public sealed class Factory : FactoryBase<StandardScalarCursor>
        {
            public Factory(RoleMappedData data, CursOpt opt)
                : base(data, opt)
            {
            }

            protected override StandardScalarCursor CreateCursorCore(DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal)
                => new StandardScalarCursor(input, data, opt, signal);
        }
    }

    /// <summary>
    /// This derives from <see cref="StandardScalarCursor"/> and adds the feature column
    /// as a <see cref="VBuffer{Float}"/>.
    /// </summary>
    [BestFriend]
    internal class FeatureFloatVectorCursor : StandardScalarCursor
    {
        private readonly ValueGetter<VBuffer<float>> _get;
        private readonly bool _keepBad;

        public long BadFeaturesRowCount { get; private set; }

        public VBuffer<float> Features;

        public FeatureFloatVectorCursor(RoleMappedData data, CursOpt opt = CursOpt.Features,
            Random rand = null, params int[] extraCols)
            : this(CreateCursor(data, opt, rand, extraCols), data, opt)
        {
        }

        protected FeatureFloatVectorCursor(DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal = null)
            : base(input, data, opt, signal)
        {
            if ((opt & CursOpt.Features) != 0 && data.Schema.Feature != null)
            {
                _get = Row.GetFeatureFloatVectorGetter(data);
                _keepBad = (opt & CursOpt.AllowBadFeatures) != 0;
            }
        }

        protected override CursOpt CursoringCompleteFlags()
        {
            var opt = base.CursoringCompleteFlags();
            if (BadFeaturesRowCount == 0)
                opt |= CursOpt.AllowBadFeatures;
            return opt;
        }

        public override bool Accept()
        {
            if (!base.Accept())
                return false;
            if (_get != null)
            {
                _get(ref Features);
                if (!_keepBad && !FloatUtils.IsFinite(Features.GetValues()))
                {
                    BadFeaturesRowCount++;
                    return false;
                }
            }
            return true;
        }

        public new sealed class Factory : FactoryBase<FeatureFloatVectorCursor>
        {
            public Factory(RoleMappedData data, CursOpt opt = CursOpt.Features)
                : base(data, opt)
            {
            }

            protected override FeatureFloatVectorCursor CreateCursorCore(DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal)
            {
                return new FeatureFloatVectorCursor(input, data, opt, signal);
            }
        }
    }

    /// <summary>
    /// This derives from the FeatureFloatVectorCursor and adds the Label (float) column.
    /// </summary>
    [BestFriend]
    internal class FloatLabelCursor : FeatureFloatVectorCursor
    {
        private readonly ValueGetter<float> _get;
        private readonly bool _keepBad;

        public long BadLabelCount { get; private set; }

        public float Label;

        public FloatLabelCursor(RoleMappedData data, CursOpt opt = CursOpt.Label,
            Random rand = null, params int[] extraCols)
            : this(CreateCursor(data, opt, rand, extraCols), data, opt)
        {
        }

        protected FloatLabelCursor(DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal = null)
            : base(input, data, opt, signal)
        {
            if ((opt & CursOpt.Label) != 0 && data.Schema.Label != null)
            {
                _get = Row.GetLabelFloatGetter(data);
                _keepBad = (opt & CursOpt.AllowBadLabels) != 0;
            }
        }

        protected override CursOpt CursoringCompleteFlags()
        {
            var opt = base.CursoringCompleteFlags();
            if (BadLabelCount == 0)
                opt |= CursOpt.AllowBadLabels;
            return opt;
        }

        public override bool Accept()
        {
            // Get the label first since base includes the features (the expensive part).
            if (_get != null)
            {
                _get(ref Label);
                if (!_keepBad && !FloatUtils.IsFinite(Label))
                {
                    BadLabelCount++;
                    return false;
                }
            }
            return base.Accept();
        }

        public new sealed class Factory : FactoryBase<FloatLabelCursor>
        {
            public Factory(RoleMappedData data, CursOpt opt = CursOpt.Label)
                : base(data, opt)
            {
            }

            protected override FloatLabelCursor CreateCursorCore(DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal)
            {
                return new FloatLabelCursor(input, data, opt, signal);
            }
        }
    }

    /// <summary>
    /// This derives from the FeatureFloatVectorCursor and adds the Label (int) column,
    /// enforcing multi-class semantics.
    /// </summary>
    [BestFriend]
    internal class MulticlassLabelCursor : FeatureFloatVectorCursor
    {
        private readonly int _classCount;
        private readonly ValueGetter<float> _get;
        private readonly bool _keepBad;

        public long BadLabelCount { get; private set; }

        private float _raw;
        public int Label;

        public MulticlassLabelCursor(int classCount, RoleMappedData data, CursOpt opt = CursOpt.Label,
            Random rand = null, params int[] extraCols)
            : this(classCount, CreateCursor(data, opt, rand, extraCols), data, opt)
        {
        }

        protected MulticlassLabelCursor(int classCount, DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal = null)
            : base(input, data, opt, signal)
        {
            Contracts.Assert(classCount >= 0);
            _classCount = classCount;

            if ((opt & CursOpt.Label) != 0 && data.Schema.Label != null)
            {
                _get = Row.GetLabelFloatGetter(data);
                _keepBad = (opt & CursOpt.AllowBadLabels) != 0;
            }
        }

        protected override CursOpt CursoringCompleteFlags()
        {
            var opt = base.CursoringCompleteFlags();
            if (BadLabelCount == 0)
                opt |= CursOpt.AllowBadLabels;
            return opt;
        }

        public override bool Accept()
        {
            // Get the label first since base includes the features (the expensive part).
            if (_get != null)
            {
                _get(ref _raw);
                Label = (int)_raw;
                if (!_keepBad && !(Label == _raw && (0 <= _raw && (_raw < _classCount || _classCount == 0))))
                {
                    BadLabelCount++;
                    return false;
                }
            }
            return base.Accept();
        }

        public new sealed class Factory : FactoryBase<MulticlassLabelCursor>
        {
            private readonly int _classCount;

            public Factory(int classCount, RoleMappedData data, CursOpt opt = CursOpt.Label)
                : base(data, opt)
            {
                // Zero means that any non-negative integer value is fine.
                Contracts.CheckParamValue(classCount >= 0, classCount, nameof(classCount), "Must be non-negative");
                _classCount = classCount;
            }

            protected override MulticlassLabelCursor CreateCursorCore(DataViewRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal)
            {
                return new MulticlassLabelCursor(_classCount, input, data, opt, signal);
            }
        }
    }
}
