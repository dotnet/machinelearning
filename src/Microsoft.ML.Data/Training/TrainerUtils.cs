// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Training
{
    /// <summary>
    /// Options for creating a row cursor from a RoleMappedData with specified standard columns active.
    /// </summary>
    [Flags]
    public enum CursOpt : uint
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

    public static class TrainerUtils
    {
        /// <summary>
        /// Check for a standard (known-length vector of float) feature column.
        /// </summary>
        public static void CheckFeatureFloatVector(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            var col = data.Schema.Feature;
            if (col == null)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a feature column.");
            Contracts.Assert(!data.Schema.Schema.IsHidden(col.Index));
            if (!col.Type.IsKnownSizeVector || col.Type.ItemType != NumberType.Float)
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
            Contracts.Assert(data.Schema.Feature != null);
            Contracts.Assert(!data.Schema.Schema.IsHidden(data.Schema.Feature.Index));
            Contracts.Assert(data.Schema.Feature.Type.IsKnownSizeVector);
            Contracts.Assert(data.Schema.Feature.Type.ItemType == NumberType.Float);
            length = data.Schema.Feature.Type.VectorSize;
        }

        /// <summary>
        /// Check for a standard binary classification label.
        /// </summary>
        public static void CheckBinaryLabel(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            var col = data.Schema.Label;
            if (col == null)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a label column.");
            Contracts.Assert(!data.Schema.Schema.IsHidden(col.Index));
            if (!col.Type.IsBool && col.Type != NumberType.R4 && col.Type != NumberType.R8 && col.Type.KeyCount != 2)
            {
                if (col.Type.IsKey)
                {
                    if (col.Type.KeyCount == 1)
                    {
                        throw Contracts.ExceptParam(nameof(data),
                            "The label column '{0}' of the training data has only one class. Two classes are required for binary classification.",
                            col.Name);
                    }
                    else if (col.Type.KeyCount > 2)
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

            var col = data.Schema.Label;
            if (col == null)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a label column.");
            Contracts.Assert(!data.Schema.Schema.IsHidden(col.Index));
            if (col.Type != NumberType.R4 && col.Type != NumberType.R8)
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
        public static void CheckMultiClassLabel(this RoleMappedData data, out int count)
        {
            Contracts.CheckValue(data, nameof(data));

            var col = data.Schema.Label;
            if (col == null)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a label column.");
            Contracts.Assert(!data.Schema.Schema.IsHidden(col.Index));
            if (col.Type.KeyCount > 0)
            {
                count = col.Type.KeyCount;
                return;
            }

            // REVIEW: Support other numeric types.
            if (col.Type != NumberType.R4 && col.Type != NumberType.R8)
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

            var col = data.Schema.Label;
            if (col == null)
                throw Contracts.ExceptParam(nameof(data), "Training data must specify a label column.");
            Contracts.Assert(!data.Schema.Schema.IsHidden(col.Index));
            if (!col.Type.IsKnownSizeVector || col.Type.ItemType != NumberType.Float)
                throw Contracts.ExceptParam(nameof(data), "Training label column '{0}' must be a known-size vector of R4, but has type: {1}.", col.Name, col.Type);
        }

        public static void CheckOptFloatWeight(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            var col = data.Schema.Weight;
            if (col == null)
                return;
            Contracts.Assert(!data.Schema.Schema.IsHidden(col.Index));
            if (col.Type != NumberType.R4 && col.Type != NumberType.R8)
                throw Contracts.ExceptParam(nameof(data), "Training weight column '{0}' must be of floating point numeric type, but has type: {1}.", col.Name, col.Type);
        }

        public static void CheckOptGroup(this RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));

            var col = data.Schema.Group;
            if (col == null)
                return;
            Contracts.Assert(!data.Schema.Schema.IsHidden(col.Index));
            if (col.Type.IsKey)
                return;
            throw Contracts.ExceptParam(nameof(data), "Training group column '{0}' type is invalid: {1}. Must be Key type.", col.Name, col.Type);
        }

        private static Func<int, bool> CreatePredicate(RoleMappedData data, CursOpt opt, IEnumerable<int> extraCols)
        {
            Contracts.AssertValue(data);
            Contracts.AssertValueOrNull(extraCols);

            var cols = new HashSet<int>();
            if ((opt & CursOpt.Label) != 0)
                AddOpt(cols, data.Schema.Label);
            if ((opt & CursOpt.Features) != 0)
                AddOpt(cols, data.Schema.Feature);
            if ((opt & CursOpt.Weight) != 0)
                AddOpt(cols, data.Schema.Weight);
            if ((opt & CursOpt.Group) != 0)
                AddOpt(cols, data.Schema.Group);
            if (extraCols != null)
            {
                foreach (var col in extraCols)
                    cols.Add(col);
            }
            return cols.Contains;
        }

        /// <summary>
        /// Create a row cursor for the RoleMappedData with the indicated standard columns active.
        /// This does not verify that the columns exist, but merely activates the ones that do exist.
        /// </summary>
        public static IRowCursor CreateRowCursor(this RoleMappedData data, CursOpt opt, IRandom rand, IEnumerable<int> extraCols = null)
            => data.Data.GetRowCursor(CreatePredicate(data, opt, extraCols), rand);

        /// <summary>
        /// Create a row cursor set for the RoleMappedData with the indicated standard columns active.
        /// This does not verify that the columns exist, but merely activates the ones that do exist.
        /// </summary>
        public static IRowCursor[] CreateRowCursorSet(this RoleMappedData data, out IRowCursorConsolidator consolidator,
            CursOpt opt, int n, IRandom rand, IEnumerable<int> extraCols = null)
            => data.Data.GetRowCursorSet(out consolidator, CreatePredicate(data, opt, extraCols), n, rand);

        private static void AddOpt(HashSet<int> cols, ColumnInfo info)
        {
            Contracts.AssertValue(cols);
            if (info != null)
                cols.Add(info.Index);
        }

        /// <summary>
        /// Get the getter for the feature column, assuming it is a vector of float.
        /// </summary>
        public static ValueGetter<VBuffer<float>> GetFeatureFloatVectorGetter(this IRow row, RoleMappedSchema schema)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(schema.Schema == row.Schema, nameof(schema), "schemas don't match!");
            Contracts.CheckParam(schema.Feature != null, nameof(schema), "Missing feature column");

            return row.GetGetter<VBuffer<float>>(schema.Feature.Index);
        }

        /// <summary>
        /// Get the getter for the feature column, assuming it is a vector of float.
        /// </summary>
        public static ValueGetter<VBuffer<float>> GetFeatureFloatVectorGetter(this IRow row, RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));
            return GetFeatureFloatVectorGetter(row, data.Schema);
        }

        /// <summary>
        /// Get a getter for the label as a float. This assumes that the label column type
        /// has already been validated as appropriate for the kind of training being done.
        /// </summary>
        public static ValueGetter<float> GetLabelFloatGetter(this IRow row, RoleMappedSchema schema)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(schema.Schema == row.Schema, nameof(schema), "schemas don't match!");
            Contracts.CheckParam(schema.Label != null, nameof(schema), "Missing label column");

            return RowCursorUtils.GetLabelGetter(row, schema.Label.Index);
        }

        /// <summary>
        /// Get a getter for the label as a float. This assumes that the label column type
        /// has already been validated as appropriate for the kind of training being done.
        /// </summary>
        public static ValueGetter<float> GetLabelFloatGetter(this IRow row, RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));
            return GetLabelFloatGetter(row, data.Schema);
        }

        /// <summary>
        /// Get the getter for the weight column, or null if there is no weight column.
        /// </summary>
        public static ValueGetter<float> GetOptWeightFloatGetter(this IRow row, RoleMappedSchema schema)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(schema.Schema == row.Schema, "schemas don't match!");
            Contracts.CheckValueOrNull(schema.Weight);

            var col = schema.Weight;
            if (col == null)
                return null;
            return RowCursorUtils.GetGetterAs<float>(NumberType.Float, row, col.Index);
        }

        public static ValueGetter<float> GetOptWeightFloatGetter(this IRow row, RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));
            return GetOptWeightFloatGetter(row, data.Schema);
        }

        /// <summary>
        /// Get the getter for the group column, or null if there is no group column.
        /// </summary>
        public static ValueGetter<ulong> GetOptGroupGetter(this IRow row, RoleMappedSchema schema)
        {
            Contracts.CheckValue(row, nameof(row));
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.Check(schema.Schema == row.Schema, "schemas don't match!");
            Contracts.CheckValueOrNull(schema.Group);

            var col = schema.Group;
            if (col == null)
                return null;
            return RowCursorUtils.GetGetterAs<ulong>(NumberType.U8, row, col.Index);
        }

        public static ValueGetter<ulong> GetOptGroupGetter(this IRow row, RoleMappedData data)
        {
            Contracts.CheckValue(data, nameof(data));
            return GetOptGroupGetter(row, data.Schema);
        }

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the label column for binary classification tasks.
        /// </summary>
        /// <param name="labelColumn">name of the label column</param>
        public static SchemaShape.Column MakeBoolScalarLabel(string labelColumn)
            => new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false);

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the label column for regression tasks.
        /// </summary>
        /// <param name="labelColumn">name of the weight column</param>
        public static SchemaShape.Column MakeR4ScalarLabel(string labelColumn)
            => new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the label column for regression tasks.
        /// </summary>
        /// <param name="labelColumn">name of the weight column</param>
        public static SchemaShape.Column MakeU4ScalarLabel(string labelColumn)
            => new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.U4, true);

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the feature column.
        /// </summary>
        /// <param name="featureColumn">name of the feature column</param>
        public static SchemaShape.Column MakeR4VecFeature(string featureColumn)
            => new SchemaShape.Column(featureColumn, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);

        /// <summary>
        /// The <see cref="SchemaShape.Column"/> for the weight column.
        /// </summary>
        /// <param name="weightColumn">name of the weight column</param>
        public static SchemaShape.Column MakeR4ScalarWeightColumn(string weightColumn)
        {
            if (weightColumn == null)
                return null;
            return new SchemaShape.Column(weightColumn, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false);
        }

        /// <summary>
        /// Check that the label, feature, weights, groupId column names are not supplied in the args of the constructor, through the advancedSettings parameter,
        /// for cases when the public constructor is called.
        /// The recommendation is to set the column names directly.
        /// </summary>
        public static void CheckArgsHaveDefaultColNames(IHostEnvironment host, LearnerInputBaseWithGroupId args)
        {
            Action<string, string> checkArgColName = (defaultColName, argValue) =>
            {
                if (argValue != defaultColName)
                    throw host.Except($"Don't supply a value for the {defaultColName} column in the arguments, as it will be ignored. Specify them in the loader, or constructor instead instead.");
            };

            // check that the users didn't specify different label, group, feature, weights in the args, from what they supplied directly
            checkArgColName(DefaultColumnNames.Label, args.LabelColumn);
            checkArgColName(DefaultColumnNames.Features, args.FeatureColumn);
            checkArgColName(DefaultColumnNames.Weight, args.WeightColumn);

            if(args.GroupIdColumn != null)
                checkArgColName(DefaultColumnNames.GroupId, args.GroupIdColumn);
        }

        public static void CheckArgsAndAdvancedSettingMismatch<T>(IChannel channel, T methodParam, T defaultVal, T setting, string argName)
        {
            // if, after applying the advancedArgs delegate, the args are different that the default value
            // and are also different than the value supplied directly to the xtension method, warn the user.
            if (!setting.Equals(defaultVal) && !setting.Equals(methodParam))
                channel.Warning($"The value supplied to advanced settings , is different than the value supplied directly. Using value {setting} for {argName}");
        }
    }

    /// <summary>
    /// This is the base class for a data cursor. Data cursors are specially typed
    /// "convenience" cursor-like objects, less general than a <see cref="IRowCursor"/> but
    /// more convenient for common access patterns that occur in machine learning. For
    /// example, the common idiom of iterating over features/labels/weights while skipping
    /// "bad" features, labels, and weights. There will be two typical access patterns for
    /// users of the cursor. The first is just creation of the cursor using a constructor;
    /// this is best for one-off accesses of the data. The second access pattern, best for
    /// repeated accesses, is to use a cursor factory (usually a nested class of the cursor
    /// class). This keeps track of what filtering options were actually useful.
    /// </summary>
    public abstract class TrainingCursorBase : IDisposable
    {
        public IRow Row { get { return _cursor; } }

        private readonly IRowCursor _cursor;
        private readonly Action<CursOpt> _signal;

        private long _skipCount;
        private long _keptCount;

        public long SkippedRowCount { get { return _skipCount; } }
        public long KeptRowCount { get { return _keptCount; } }

        /// <summary>
        /// The base constructor class for the factory-based cursor creation.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="signal">This method is called </param>
        protected TrainingCursorBase(IRowCursor input, Action<CursOpt> signal)
        {
            Contracts.AssertValue(input);
            Contracts.AssertValueOrNull(signal);
            _cursor = input;
            _signal = signal;
        }

        protected static IRowCursor CreateCursor(RoleMappedData data, CursOpt opt, IRandom rand, params int[] extraCols)
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
                    _keptCount++;
                    return true;
                }
                _skipCount++;
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

            public RoleMappedData Data { get { return _data; } }

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
            /// The typed analog to <see cref="IDataView.GetRowCursor(Func{int,bool},IRandom)"/>.
            /// </summary>
            /// <param name="rand">Non-null if we are requesting a shuffled cursor.</param>
            /// <param name="extraCols">The extra columns to activate on the row cursor
            /// in addition to those required by the factory's options.</param>
            /// <returns>The wrapping typed cursor.</returns>
            public TCurs Create(IRandom rand = null, params int[] extraCols)
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
            public TCurs[] CreateSet(int n, IRandom rand = null, params int[] extraCols)
            {
                CursOpt opt;
                lock (_lock)
                    opt = _opts;

                // The intended use of this sort of thing is for cases where we have no interest in
                // doing consolidation at all, that is, the consuming endpoint using these typed
                // cursors wants to consume them as a set.
                IRowCursorConsolidator consolidator;
                var inputs = _data.CreateRowCursorSet(out consolidator, opt, n, rand, extraCols);
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
            protected abstract TCurs CreateCursorCore(IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal);

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
    /// This supports Weight (float), Group (ulong), and Id (UInt128) columns.
    /// </summary>
    public class StandardScalarCursor : TrainingCursorBase
    {
        private readonly ValueGetter<float> _getWeight;
        private readonly ValueGetter<ulong> _getGroup;
        private readonly ValueGetter<UInt128> _getId;
        private readonly bool _keepBadWeight;
        private readonly bool _keepBadGroup;

        private long _badWeightCount;
        private long _badGroupCount;
        public long BadWeightCount { get { return _badWeightCount; } }
        public long BadGroupCount { get { return _badGroupCount; } }

        public float Weight;
        public ulong Group;
        public UInt128 Id;

        public StandardScalarCursor(RoleMappedData data, CursOpt opt, IRandom rand = null, params int[] extraCols)
            : this(CreateCursor(data, opt, rand, extraCols), data, opt)
        {
        }

        protected StandardScalarCursor(IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal = null)
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
                    _badWeightCount++;
                    return false;
                }
            }
            if (_getGroup != null)
            {
                _getGroup(ref Group);
                if (!_keepBadGroup && Group == 0)
                {
                    _badGroupCount++;
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

            protected override StandardScalarCursor CreateCursorCore(IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal)
                => new StandardScalarCursor(input, data, opt, signal);
        }
    }

    /// <summary>
    /// This derives from <see cref="StandardScalarCursor"/> and adds the feature column
    /// as a <see cref="VBuffer{Float}"/>.
    /// </summary>
    public class FeatureFloatVectorCursor : StandardScalarCursor
    {
        private readonly ValueGetter<VBuffer<float>> _get;
        private readonly bool _keepBad;

        private long _badCount;
        public long BadFeaturesRowCount { get { return _badCount; } }

        public VBuffer<float> Features;

        public FeatureFloatVectorCursor(RoleMappedData data, CursOpt opt = CursOpt.Features,
            IRandom rand = null, params int[] extraCols)
            : this(CreateCursor(data, opt, rand, extraCols), data, opt)
        {
        }

        protected FeatureFloatVectorCursor(IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal = null)
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
                if (!_keepBad && !FloatUtils.IsFinite(Features.Values, Features.Count))
                {
                    _badCount++;
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

            protected override FeatureFloatVectorCursor CreateCursorCore(IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal)
            {
                return new FeatureFloatVectorCursor(input, data, opt, signal);
            }
        }
    }

    /// <summary>
    /// This derives from the FeatureFloatVectorCursor and adds the Label (float) column.
    /// </summary>
    public class FloatLabelCursor : FeatureFloatVectorCursor
    {
        private readonly ValueGetter<float> _get;
        private readonly bool _keepBad;

        private long _badCount;

        public long BadLabelCount { get { return _badCount; } }

        public float Label;

        public FloatLabelCursor(RoleMappedData data, CursOpt opt = CursOpt.Label,
            IRandom rand = null, params int[] extraCols)
            : this(CreateCursor(data, opt, rand, extraCols), data, opt)
        {
        }

        protected FloatLabelCursor(IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal = null)
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
                    _badCount++;
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

            protected override FloatLabelCursor CreateCursorCore(IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal)
            {
                return new FloatLabelCursor(input, data, opt, signal);
            }
        }
    }

    /// <summary>
    /// This derives from the FeatureFloatVectorCursor and adds the Label (int) column,
    /// enforcing multi-class semantics.
    /// </summary>
    public class MultiClassLabelCursor : FeatureFloatVectorCursor
    {
        private readonly int _classCount;
        private readonly ValueGetter<float> _get;
        private readonly bool _keepBad;

        private long _badCount;
        public long BadLabelCount { get { return _badCount; } }

        private float _raw;
        public int Label;

        public MultiClassLabelCursor(int classCount, RoleMappedData data, CursOpt opt = CursOpt.Label,
            IRandom rand = null, params int[] extraCols)
            : this(classCount, CreateCursor(data, opt, rand, extraCols), data, opt)
        {
        }

        protected MultiClassLabelCursor(int classCount, IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal = null)
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
                    _badCount++;
                    return false;
                }
            }
            return base.Accept();
        }

        public new sealed class Factory : FactoryBase<MultiClassLabelCursor>
        {
            private readonly int _classCount;

            public Factory(int classCount, RoleMappedData data, CursOpt opt = CursOpt.Label)
                : base(data, opt)
            {
                // Zero means that any non-negative integer value is fine.
                Contracts.CheckParamValue(classCount >= 0, classCount, nameof(classCount), "Must be non-negative");
                _classCount = classCount;
            }

            protected override MultiClassLabelCursor CreateCursorCore(IRowCursor input, RoleMappedData data, CursOpt opt, Action<CursOpt> signal)
            {
                return new MultiClassLabelCursor(_classCount, input, data, opt, signal);
            }
        }
    }
}
