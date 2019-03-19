// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(SymbolicSgdTrainer), typeof(SymbolicSgdTrainer.Options),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    SymbolicSgdTrainer.UserNameValue,
    SymbolicSgdTrainer.LoadNameValue,
    SymbolicSgdTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(SymbolicSgdTrainer), null, typeof(SignatureEntryPointModule), SymbolicSgdTrainer.LoadNameValue)]

namespace Microsoft.ML.Trainers
{
    using TPredictor = CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>;

    /// <include file='doc.xml' path='doc/members/member[@name="SymSGD"]/*' />
    public sealed class SymbolicSgdTrainer : TrainerEstimatorBase<BinaryPredictionTransformer<TPredictor>, TPredictor>
    {
        internal const string LoadNameValue = "SymbolicSGD";
        internal const string UserNameValue = "Symbolic SGD (binary)";
        internal const string ShortName = "SymSGD";

        ///<summary> Advanced options for trainer.</summary>
        public sealed class Options : TrainerInputBaseWithLabel
        {
            /// <summary>
            /// Degree of lock-free parallelism. Determinism not guaranteed if this is set to higher than 1.
            /// The default value is the number of logical cores that are available on the system.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Degree of lock-free parallelism. Determinism not guaranteed. " +
                "Multi-threading is not supported currently.", ShortName = "nt")]
            public int? NumberOfThreads;

            /// <summary>
            /// Number of passes over the data.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of passes over the data.", ShortName = "iter", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "1,5,10,20,30,40,50")]
            [TlcModule.SweepableDiscreteParam("NumberOfIterations", new object[] { 1, 5, 10, 20, 30, 40, 50 })]
            public int NumberOfIterations = Defaults.NumberOfIterations;

            /// <summary>
            /// Tolerance for difference in average loss in consecutive passes.
            /// If the reduction on loss is smaller than the specified tolerance in one iteration, the training process will be terminated.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Tolerance for difference in average loss in consecutive passes.", ShortName = "tol")]
            public float Tolerance = Defaults.Tolerance;

            /// <summary>
            /// Learning rate. A larger value can potentially reduce the training time but incur numerical instability and over-fitting.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate", ShortName = "lr", NullName = "<Auto>", SortOrder = 51)]
            [TGUI(SuggestedSweeps = "<Auto>,1e1,1e0,1e-1,1e-2,1e-3")]
            [TlcModule.SweepableDiscreteParam("LearningRate", new object[] { "<Auto>", 1e1f, 1e0f, 1e-1f, 1e-2f, 1e-3f })]
            public float? LearningRate;

            /// <summary>
            /// L2 regularization.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 regularization", ShortName = "l2", SortOrder = 52)]
            [TGUI(SuggestedSweeps = "0.0,1e-5,1e-5,1e-6,1e-7")]
            [TlcModule.SweepableDiscreteParam("L2Regularization", new object[] { 0.0f, 1e-5f, 1e-5f, 1e-6f, 1e-7f })]
            public float L2Regularization = Defaults.L2Regularization;

            /// <summary>
            /// The number of iterations each thread learns a local model until combining it with the
            /// global model. Low value means more updated global model and high value means less cache traffic.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of iterations each thread learns a local model until combining it with the " +
                "global model. Low value means more updated global model and high value means less cache traffic.", ShortName = "freq", NullName = "<Auto>")]
            [TGUI(SuggestedSweeps = "<Auto>,5,20")]
            [TlcModule.SweepableDiscreteParam("UpdateFrequency", new object[] { "<Auto>", 5, 20 })]
            public int? UpdateFrequency;

            /// <summary>
            /// The acceleration memory budget in MB.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The acceleration memory budget in MB", ShortName = "accelMemBudget")]
            public long MemorySize = Defaults.MemorySize;

            /// <summary>
            /// Set to <see langword="true" /> causes the data to shuffle.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Shuffle data?", ShortName = "shuf")]
            public bool Shuffle = Defaults.Shuffle;

            /// <summary>
            /// Apply weight to the positive class, for imbalanced data.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Apply weight to the positive class, for imbalanced data", ShortName = "piw")]
            public float PositiveInstanceWeight = Defaults.PositiveInstanceWeight;

            internal void Check(IExceptionContext ectx)
            {
                ectx.CheckUserArg(LearningRate == null || LearningRate.Value > 0, nameof(LearningRate), "Must be positive.");
                ectx.CheckUserArg(NumberOfIterations > 0, nameof(NumberOfIterations), "Must be positive.");
                ectx.CheckUserArg(PositiveInstanceWeight > 0, nameof(PositiveInstanceWeight), "Must be positive");
                ectx.CheckUserArg(UpdateFrequency == null || UpdateFrequency > 0, nameof(UpdateFrequency), "Must be positive");
            }
        }

        [BestFriend]
        internal static class Defaults
        {
            public const float PositiveInstanceWeight = 1;
            public const bool Shuffle = true;
            public const long MemorySize = 1024;
            public const float L2Regularization = 0;
            public const float Tolerance = 1e-4f;
            public const int NumberOfIterations = 50;
        }

        public override TrainerInfo Info { get; }

        private readonly Options _options;

        /// <summary>
        /// This method ensures that the data meets the requirements of this trainer and its
        /// subclasses, injects necessary transforms, and throws if it couldn't meet them.
        /// </summary>
        /// <param name="ch">The channel</param>
        /// <param name="examples">The training examples</param>
        /// <param name="weightSetCount">Gets the length of weights and bias array. For binary classification and regression,
        /// this is 1. For multi-class classification, this equals the number of classes on the label.</param>
        /// <returns>A potentially modified version of <paramref name="examples"/></returns>
        private RoleMappedData PrepareDataFromTrainingExamples(IChannel ch, RoleMappedData examples, out int weightSetCount)
        {
            ch.AssertValue(examples);
            CheckLabel(examples, out weightSetCount);
            examples.CheckFeatureFloatVector();
            var idvToShuffle = examples.Data;
            IDataView idvToFeedTrain;
            if (idvToShuffle.CanShuffle)
                idvToFeedTrain = idvToShuffle;
            else
            {
                var shuffleArgs = new RowShufflingTransformer.Options
                {
                    PoolOnly = false,
                    ForceShuffle = _options.Shuffle
                };
                idvToFeedTrain = new RowShufflingTransformer(Host, shuffleArgs, idvToShuffle);
            }

            ch.Assert(idvToFeedTrain.CanShuffle);

            var roles = examples.Schema.GetColumnRoleNames();
            var examplesToFeedTrain = new RoleMappedData(idvToFeedTrain, roles);

            ch.Assert(examplesToFeedTrain.Schema.Label.HasValue);
            ch.Assert(examplesToFeedTrain.Schema.Feature.HasValue);
            if (examples.Schema.Weight.HasValue)
                ch.Assert(examplesToFeedTrain.Schema.Weight.HasValue);

            ch.Check(examplesToFeedTrain.Schema.Feature.Value.Type is VectorType vecType && vecType.Size > 0, "Training set has no features, aborting training.");
            return examplesToFeedTrain;
        }

        private protected override TPredictor TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            using (var ch = Host.Start("Training"))
            {
                var initPred = context.InitialPredictor;
                var linearInitPred = initPred as LinearModelParameters;
                // If initial predictor is set, it must be a linear model.
                // If initPred is null (i.e., not set), the following check will always be bypassed.
                // If initPred is not null, then the following checks if a LinearModelParameters is loaded to linearInitPred.
                Host.CheckParam(initPred == null || linearInitPred != null, nameof(context),
                    "Initial predictor was not a linear predictor.");

                var preparedData = PrepareDataFromTrainingExamples(ch, context.TrainingSet, out int weightSetCount);
                return TrainCore(ch, preparedData, linearInitPred, weightSetCount);
            }
        }

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        /// <summary>
        /// Initializes a new instance of <see cref="SymbolicSgdTrainer"/>
        /// </summary>
        internal SymbolicSgdTrainer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadNameValue), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName),
                  TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
            Host.CheckValue(options, nameof(options));
            options.Check(Host);

            _options = options;
            Info = new TrainerInfo(supportIncrementalTrain: true);
        }

        private TPredictor CreatePredictor(VBuffer<float> weights, float bias)
        {
            Host.CheckParam(weights.Length > 0, nameof(weights));

            VBuffer<float> maybeSparseWeights = default;
            VBufferUtils.CreateMaybeSparseCopy(in weights, ref maybeSparseWeights,
                Conversions.Instance.GetIsDefaultPredicate<float>(NumberDataViewType.Single));
            var predictor = new LinearBinaryModelParameters(Host, in maybeSparseWeights, bias);
            return new ParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator>(Host, predictor, new PlattCalibrator(Host, -1, 0));
        }

        private protected override BinaryPredictionTransformer<TPredictor> MakeTransformer(TPredictor model, DataViewSchema trainSchema)
             => new BinaryPredictionTransformer<TPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Continues the training of <see cref="SymbolicSgdTrainer"/> using an already trained <paramref name="modelParameters"/>
        /// a <see cref="BinaryPredictionTransformer"/>.
        /// </summary>
        public BinaryPredictionTransformer<TPredictor> Fit(IDataView trainData, LinearModelParameters modelParameters)
            => TrainTransformer(trainData, initPredictor: modelParameters);

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        [TlcModule.EntryPoint(Name = "Trainers.SymSgdBinaryClassifier",
            Desc = "Train a symbolic SGD.",
            UserName = SymbolicSgdTrainer.UserNameValue,
            ShortName = SymbolicSgdTrainer.ShortName)]
        internal static CommonOutputs.BinaryClassificationOutput TrainSymSgd(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainSymSGD");
            host.CheckValue(options, nameof(options));
            EntryPointUtils.CheckInputArgs(host, options);

            return TrainerEntryPointsUtils.Train<Options, CommonOutputs.BinaryClassificationOutput>(host, options,
                () => new SymbolicSgdTrainer(host, options),
                () => TrainerEntryPointsUtils.FindColumn(host, options.TrainingData.Schema, options.LabelColumnName));
        }

        // We buffer instances from the cursor (limited to memorySize) and passes that buffer to
        // the native code to learn for multiple instances by one interop call.

        /// <summary>
        /// This struct holds the information about the size, label and isDense of each instance
        /// to be able to pass it to the native code.
        /// </summary>
        private readonly struct InstanceProperties
        {
            public readonly int FeatureCount;
            public readonly float Label;
            public readonly bool IsDense;

            public InstanceProperties(int featureCount, float label, bool isDense)
            {
                FeatureCount = featureCount;
                Label = label;
                IsDense = isDense;
            }
        }

        /// <summary>
        /// ArrayManager stores multiple arrays of type <typeparamref name="T"/> in a "very long" array whose size is specified by accelChunkSize.
        /// Once one of the very long arrays is full, another one is allocated to store additional arrays. The required memory
        /// for this buffering is limited by memorySize.
        ///
        /// Note that these very long arrays can be reused. This means that learning can be done in batches without the overhead associated
        /// with allocation.
        ///
        /// The benefit of this way of storage is that only a handful of new calls will be needed
        /// which saves time.
        /// </summary>
        /// <typeparam name="T">The type of arrays to be stored</typeparam>
        private sealed class ArrayManager<T> : IDisposable
        {
            /// <summary>
            /// This structure is used for pinning very long arrays to stop GC from moving them.
            /// The reason for this design is that when these arrays are passed to native code,
            /// GC does not move the objects.
            /// </summary>
            private struct VeryLongArray
            {
                public T[] Buffer;
                public GCHandle GcHandle;
                public int Length => Buffer.Length;
                public VeryLongArray(int veryLongArrayLength)
                {
                    Buffer = new T[veryLongArrayLength];
                    GcHandle = GCHandle.Alloc(Buffer, GCHandleType.Pinned);
                }

                public void Free()
                {
                    GcHandle.Free();
                }
            }
            // This list holds very long arrays.
            private readonly List<VeryLongArray> _storage;
            // Length of each very long array
            // This is not readonly because there might be an instance where the length of the
            // instance is longer than _veryLongArrayLength and we have to adjust it
            private int _veryLongArrayLength;
            // This index is used to walk over _storage list. During storing or giving an array,
            // we are at _storage[_storageIndex].
            private int _storageIndex;
            // This index is used within a very long array from _storage[_storageIndex]. During storing or
            // giving an array, we are at _storage[_storageIndex][_indexInCurArray].
            private int _indexInCurArray;
            // This is used to access AccelMemBudget, AccelChunkSize and UsedMemory
            private readonly SymbolicSgdTrainer _trainer;

            private readonly IChannel _ch;

            // Size of type T
            private readonly int _sizeofT;

            /// <summary>
            /// Constructor for initializing _storage and other indices.
            /// </summary>
            /// <param name="trainer"></param>
            /// <param name="ch"></param>
            public ArrayManager(SymbolicSgdTrainer trainer, IChannel ch)
            {
                _storage = new List<VeryLongArray>();
                // Setting the default value to 2^17.
                _veryLongArrayLength = (1 << 17);
                _indexInCurArray = 0;
                _storageIndex = 0;
                _trainer = trainer;
                _ch = ch;
                _sizeofT = Marshal.SizeOf(typeof(T));
            }

            /// <summary>
            /// </summary>
            /// <returns>Returns if the allocation was successful</returns>
            private bool CheckAndAllocate()
            {
                // Check if this allocation violates the memorySize.
                if (_trainer.UsedMemory + _veryLongArrayLength * _sizeofT <= _trainer.AcceleratedMemoryBudgetBytes)
                {
                    // Add the additional allocation to UsedMemory
                    _trainer.UsedMemory += _veryLongArrayLength * _sizeofT;
                    _storage.Add(new VeryLongArray(_veryLongArrayLength));
                    return true;
                }
                // If allocation violates the budget, bail.
                return false;
            }

            /// <summary>
            /// This method checks if an array of size <paramref name="size"/> fits in _storage[_storageIndex][_indexInCurArray.._indexInCurArray+size-1].
            /// </summary>
            /// <param name="size">The size of the array to fit in the very long array _storage[_storageIndex] </param>
            /// <returns></returns>
            private bool FitsInCurArray(int size)
            {
                _ch.Assert(_storage[_storageIndex].Length == _veryLongArrayLength);
                return _indexInCurArray <= _veryLongArrayLength - size;
            }

            /// <summary>
            /// Tries to add span <paramref name="instArray"/> to the storage without violating the restriction of memorySize.
            /// </summary>
            /// <param name="instArray">The span to be added</param>
            /// <returns>Return if the allocation was successful</returns>
            public bool AddToStorage(ReadOnlySpan<T> instArray)
            {
                var instArrayLength = instArray.Length;
                _ch.Assert(instArrayLength * _sizeofT * 2 < _trainer.AcceleratedMemoryBudgetBytes);
                if (instArrayLength > _veryLongArrayLength)
                {
                    // In this case, we need to increase _veryLongArrayLength.
                    if (_indexInCurArray == 0 && _storageIndex == 0)
                    {
                        // If there are no instances loaded, all of the allocated very long arrays need to be deallocated
                        // and longer _veryLongArrayLength be used instead.
                        DeallocateVeryLongArrays();
                        _storage.Clear();

                        _veryLongArrayLength = instArrayLength;
                    }
                    else
                    {
                        // If there are already instances loaded into the _storage, train on them.
                        return false;
                    }
                }
                // Special case that happens only when _storage is empty
                if (_storage.Count == 0)
                {
                    if (!CheckAndAllocate())
                        return false;
                    _indexInCurArray = 0;
                }
                // Check if instArray can be fitted in the current setup.
                else if (!FitsInCurArray(instArrayLength))
                {
                    // Check if we reached the end of _storage. If so try to allocate a new very long array.
                    // Otherwise, there are more very long arrays left, just move to the next one.
                    if (_storageIndex == _storage.Count - 1)
                    {
                        if (!CheckAndAllocate())
                            return false;
                    }
                    _indexInCurArray = 0;
                    _storageIndex++;
                }
                instArray.CopyTo(_storage[_storageIndex].Buffer.AsSpan(_indexInCurArray));
                _indexInCurArray += instArrayLength;
                return true;
            }

            /// <summary>
            /// This is a soft clear, meaning that it doesn't reallocate, only sets _storageIndex and
            /// _indexInCurArray to 0.
            /// </summary>
            public void ResetIndexing()
            {
                _storageIndex = 0;
                _indexInCurArray = 0;
            }

            /// <summary>
            /// Gives an array of <paramref name="size"/>.
            /// </summary>
            /// <param name="size">The size of array to give</param>
            /// <param name="outGcHandle"></param>
            /// <param name="outArrayStartIndex"></param>
            public void GiveArrayOfSize(int size, out GCHandle? outGcHandle, out int outArrayStartIndex)
            {
                // Generally it is the user responsibility to not ask for an array of a size that has not been
                // previously allocated.

                // In case no allocation has occured.
                if (_storage.Count == 0)
                {
                    outGcHandle = null;
                    outArrayStartIndex = 0;
                }
                else
                {
                    // Check if the array fits in _storage[_storageIndex].
                    if (!FitsInCurArray(size))
                    {
                        // If not, it must be in the next very long array.
                        _storageIndex++;
                        _indexInCurArray = 0;
                    }
                    outGcHandle = _storage[_storageIndex].GcHandle;
                    outArrayStartIndex = _indexInCurArray;
                    _indexInCurArray += size;
                }
            }

            private void DeallocateVeryLongArrays()
            {
                foreach (var veryLongArray in _storage)
                    veryLongArray.Free();
            }

            public void Dispose()
            {
                DeallocateVeryLongArrays();
            }
        }

        /// <summary>
        /// This class manages the buffering for instances
        /// </summary>
        private sealed class InputDataManager : IDisposable
        {
            // This ArrayManager is used for indices of instances
            private readonly ArrayManager<int> _instIndices;
            // This ArrayManager is used for values of instances
            private readonly ArrayManager<float> _instValues;
            // This is a list of the properties of instances that are buffered.
            private readonly List<InstanceProperties> _instanceProperties;
            private readonly FloatLabelCursor.Factory _cursorFactory;
            private FloatLabelCursor _cursor;
            // This is used as a mechanism to make sure that the memorySize restriction is not violated.
            private bool _cursorMoveNext;
            // This is the index to go over the instances in instanceProperties
            private int _instanceIndex;
            // This is used to access AccelMemBudget, AccelChunkSize and UsedMemory
            private readonly SymbolicSgdTrainer _trainer;
            private readonly IChannel _ch;

            // Whether memorySize was big enough to load the entire instances into the buffer
            private bool _isFullyLoaded;
            public bool IsFullyLoaded => _isFullyLoaded;
            public int Count => _instanceProperties.Count;

            // Tells if we have gone through the dataset entirely.
            public bool FinishedTheLoad => !_cursorMoveNext;

            public InputDataManager(SymbolicSgdTrainer trainer, FloatLabelCursor.Factory cursorFactory, IChannel ch)
            {
                _instIndices = new ArrayManager<int>(trainer, ch);
                _instValues = new ArrayManager<float>(trainer, ch);
                _instanceProperties = new List<InstanceProperties>();

                _cursorFactory = cursorFactory;
                _ch = ch;
                _cursor = cursorFactory.Create();
                _cursorMoveNext = _cursor.MoveNext();
                _isFullyLoaded = true;

                _instanceIndex = 0;

                _trainer = trainer;
            }

            // Has to be called for cursoring through the data
            public void RestartLoading(bool needShuffle, IHost host)
            {
                _cursor.Dispose();
                if (needShuffle)
                    _cursor = _cursorFactory.Create(RandomUtils.Create(host.Rand.Next()));
                else
                    _cursor = _cursorFactory.Create();
                _cursorMoveNext = _cursor.MoveNext();
            }

            /// <summary>
            /// This method tries to load as much as possible from the cursor into the buffer until the memorySize is reached.
            /// </summary>
            public void LoadAsMuchAsPossible()
            {
                _instValues.ResetIndexing();
                _instIndices.ResetIndexing();
                _instanceProperties.Clear();

                while (_cursorMoveNext)
                {
                    var featureValues = _cursor.Features.GetValues();
                    int featureCount = featureValues.Length;
                    // If the instance has no feature, ignore it!
                    if (featureCount == 0)
                    {
                        _cursorMoveNext = _cursor.MoveNext();
                        continue;
                    }

                    // We assume that cursor.Features.values are represented by float and cursor.Features.indices are represented by int
                    // We conservatively assume that an instance is sparse and therefore, it has an array of Floats and ints for values and indices
                    int perNonZeroInBytes = sizeof(float) + sizeof(int);
                    if (featureCount > _trainer.AcceleratedMemoryBudgetBytes / perNonZeroInBytes)
                    {
                        // Hopefully this never happens. But the memorySize must >= perNonZeroInBytes * length(the longest instance).
                        throw _ch.Except("Acceleration memory budget is too small! Need at least {0} MB for at least one of the instances",
                            featureCount * perNonZeroInBytes / (1024 * 1024));
                    }

                    bool couldLoad = true;
                    if (!_cursor.Features.IsDense)
                        // If it is a sparse instance, load its indices to instIndices buffer
                        couldLoad = _instIndices.AddToStorage(_cursor.Features.GetIndices());
                    // Load values of an instance into instValues
                    if (couldLoad)
                        couldLoad = _instValues.AddToStorage(featureValues);

                    // If the load was successful, load the instance properties to instanceProperties
                    if (couldLoad)
                    {
                        float label = _cursor.Label;
                        InstanceProperties prop = new InstanceProperties(featureCount, label, _cursor.Features.IsDense);
                        _instanceProperties.Add(prop);

                        _cursorMoveNext = _cursor.MoveNext();

                        if (_instanceProperties.Count > (1 << 30))
                        {
                            // If it happened to be the case that we have so much memory that we were able to load (1<<30) instances,
                            // break. This is because in such a case _instanceProperties can only be addressed by int32 and (1<<30) is
                            // getting close to the limits. This should rarely happen!
                            _isFullyLoaded = false;
                            break;
                        }
                    }
                    else
                    {
                        // If couldLoad fails at any point (which is becuase of memorySize), isFullyLoaded becomes false forever
                        _isFullyLoaded = false;
                        break;
                    }
                }
            }

            public void PrepareCursoring()
            {
                _instanceIndex = 0;
                _instIndices.ResetIndexing();
                _instValues.ResetIndexing();
            }

            /// <summary>
            /// This method provides instances stored in the buffer in a sequential order. Note that method PrepareCursoring should be called before using this method.
            /// </summary>
            /// <param name="prop">The property of the given instance. It is set to null in case there are no more instance.</param>
            /// <param name="indicesGcHandle"></param>
            /// <param name="indicesStartIndex">The offset for the indices array.</param>
            /// <param name="valuesGcHandle"></param>
            /// <param name="valuesStartIndex">The offset for the values array.</param>
            /// <returns>Retruns whether output is valid. Otherwise we have gone through the entire loaded instances.</returns>
            public bool GiveNextInstance(out InstanceProperties? prop, out GCHandle? indicesGcHandle, out int indicesStartIndex,
                out GCHandle? valuesGcHandle, out int valuesStartIndex)
            {
                if (_instanceIndex == _instanceProperties.Count)
                {
                    // We hit the end.
                    prop = null;
                    indicesGcHandle = null;
                    indicesStartIndex = 0;
                    valuesGcHandle = null;
                    valuesStartIndex = 0;
                    return false;
                }
                prop = _instanceProperties[_instanceIndex];
                if (!prop.Value.IsDense)
                {
                    // If sparse, set indices array accordingly.
                    _instIndices.GiveArrayOfSize(prop.Value.FeatureCount, out indicesGcHandle, out indicesStartIndex);
                }
                else
                {
                    indicesGcHandle = null;
                    indicesStartIndex = 0;
                }
                // Load values here.
                _instValues.GiveArrayOfSize(prop.Value.FeatureCount, out valuesGcHandle, out valuesStartIndex);
                _instanceIndex++;
                return true;
            }

            public void Dispose()
            {
                _cursor.Dispose();
                _instIndices.Dispose();
                _instValues.Dispose();
            }
        }

        private TPredictor TrainCore(IChannel ch, RoleMappedData data, LinearModelParameters predictor, int weightSetCount)
        {
            int numFeatures = data.Schema.Feature.Value.Type.GetVectorSize();
            var cursorFactory = new FloatLabelCursor.Factory(data, CursOpt.Label | CursOpt.Features);
            int numThreads = _options.NumberOfThreads ?? Environment.ProcessorCount;

            ch.CheckUserArg(numThreads > 0, nameof(_options.NumberOfThreads),
                "The number of threads must be either null or a positive integer.");

            var positiveInstanceWeight = _options.PositiveInstanceWeight;
            VBuffer<float> weights = default;
            float bias = 0.0f;
            if (predictor != null)
            {
                ((IHaveFeatureWeights)predictor).GetFeatureWeights(ref weights);
                VBufferUtils.Densify(ref weights);
                bias = predictor.Bias;
            }
            else
                weights = VBufferUtils.CreateDense<float>(numFeatures);

            var weightsEditor = VBufferEditor.CreateFromBuffer(ref weights);

            // Reference: Parasail. SymSGD.
            bool tuneLR = _options.LearningRate == null;
            var lr = _options.LearningRate ?? 1.0f;

            bool tuneNumLocIter = (_options.UpdateFrequency == null);
            var numLocIter = _options.UpdateFrequency ?? 1;

            var l2Const = _options.L2Regularization;
            var piw = _options.PositiveInstanceWeight;

            // This is state of the learner that is shared with the native code.
            State state = new State();
            GCHandle stateGCHandle = default;
            try
            {
                stateGCHandle = GCHandle.Alloc(state, GCHandleType.Pinned);

                state.TotalInstancesProcessed = 0;
                using (InputDataManager inputDataManager = new InputDataManager(this, cursorFactory, ch))
                {
                    bool shouldInitialize = true;
                    using (var pch = Host.StartProgressChannel("Preprocessing"))
                        inputDataManager.LoadAsMuchAsPossible();

                    int iter = 0;
                    if (inputDataManager.IsFullyLoaded)
                        ch.Info("Data fully loaded into memory.");
                    using (var pch = Host.StartProgressChannel("Training"))
                    {
                        if (inputDataManager.IsFullyLoaded)
                        {
                            pch.SetHeader(new ProgressHeader(new[] { "iterations" }),
                                entry => entry.SetProgress(0, state.PassIteration, _options.NumberOfIterations));
                            // If fully loaded, call the SymSGDNative and do not come back until learned for all iterations.
                            Native.LearnAll(inputDataManager, tuneLR, ref lr, l2Const, piw, weightsEditor.Values, ref bias, numFeatures,
                                _options.NumberOfIterations, numThreads, tuneNumLocIter, ref numLocIter, _options.Tolerance, _options.Shuffle, shouldInitialize,
                                stateGCHandle, ch.Info);
                            shouldInitialize = false;
                        }
                        else
                        {
                            pch.SetHeader(new ProgressHeader(new[] { "iterations" }),
                                entry => entry.SetProgress(0, iter, _options.NumberOfIterations));

                            // Since we loaded data in batch sizes, multiple passes over the loaded data is feasible.
                            int numPassesForABatch = inputDataManager.Count / 10000;
                            while (iter < _options.NumberOfIterations)
                            {
                                // We want to train on the final passes thoroughly (without learning on the same batch multiple times)
                                // This is for fine tuning the AUC. Experimentally, we found that 1 or 2 passes is enough
                                int numFinalPassesToTrainThoroughly = 2;
                                // We also do not want to learn for more passes than what the user asked
                                int numPassesForThisBatch = Math.Min(numPassesForABatch, _options.NumberOfIterations - iter - numFinalPassesToTrainThoroughly);
                                // If all of this leaves us with 0 passes, then set numPassesForThisBatch to 1
                                numPassesForThisBatch = Math.Max(1, numPassesForThisBatch);
                                state.PassIteration = iter;
                                Native.LearnAll(inputDataManager, tuneLR, ref lr, l2Const, piw, weightsEditor.Values, ref bias, numFeatures,
                                    numPassesForThisBatch, numThreads, tuneNumLocIter, ref numLocIter, _options.Tolerance, _options.Shuffle, shouldInitialize,
                                    stateGCHandle, ch.Info);
                                shouldInitialize = false;

                                // Check if we are done with going through the data
                                if (inputDataManager.FinishedTheLoad)
                                {
                                    iter += numPassesForThisBatch;
                                    // Check if more passes are left
                                    if (iter < _options.NumberOfIterations)
                                        inputDataManager.RestartLoading(_options.Shuffle, Host);
                                }

                                // If more passes are left, load as much as possible
                                if (iter < _options.NumberOfIterations)
                                    inputDataManager.LoadAsMuchAsPossible();
                            }
                        }

                        // Maps back the dense features that are mislocated
                        if (numThreads > 1)
                            Native.MapBackWeightVector(weightsEditor.Values, stateGCHandle);
                        Native.DeallocateSequentially(stateGCHandle);
                    }
                }
            }
            finally
            {
                if (stateGCHandle.IsAllocated)
                    stateGCHandle.Free();
            }
            return CreatePredictor(weights, bias);
        }

        private void CheckLabel(RoleMappedData examples, out int weightSetCount)
        {
            examples.CheckBinaryLabel();
            weightSetCount = 1;
        }

        private long AcceleratedMemoryBudgetBytes => _options.MemorySize * 1024 * 1024;
        private long UsedMemory { get; set; }

        private static unsafe class Native
        {
            //To triger the loading of MKL library since SymSGD native library depends on it.
            static Native() => ErrorMessage(0);

            internal const string NativePath = "SymSgdNative";
            internal const string MklPath = "MklImports";

            public delegate void ChannelCallBack(string message);

            [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
            private static extern void LearnAll(int totalNumInstances, int* instSizes, int** instIndices,
                float** instValues, float* labels, bool tuneLR, ref float lr, float l2Const, float piw, float* weightVector, ref float bias,
                int numFeatres, int numPasses, int numThreads, bool tuneNumLocIter, ref int numLocIter, float tolerance, bool needShuffle, bool shouldInitialize,
                State* state, ChannelCallBack info);

            /// <summary>
            /// This method puts all of the buffered instances in array of pointers to pass it to SymSGDNative.
            /// </summary>
            /// <param name="inputDataManager">The buffered data</param>
            /// <param name="tuneLR">Specifies if SymSGD should tune alpha automatically</param>
            /// <param name="lr">Initial learning rate</param>
            /// <param name="l2Const"></param>
            /// <param name="piw"></param>
            /// <param name="weightVector">The storage for the weight vector</param>
            /// <param name="bias">bias</param>
            /// <param name="numFeatres">Number of features</param>
            /// <param name="numPasses">Number of passes</param>
            /// <param name="numThreads">Number of threads</param>
            /// <param name="tuneNumLocIter">Specifies if SymSGD should tune numLocIter automatically</param>
            /// <param name="numLocIter">Number of thread local iterations of SGD before combining with the global model</param>
            /// <param name="tolerance">Tolerance for the amount of decrease in the total loss in consecutive passes</param>
            /// <param name="needShuffle">Specifies if data needs to be shuffled</param>
            /// <param name="shouldInitialize">Specifies if this is the first time to run SymSGD</param>
            /// <param name="stateGCHandle"></param>
            /// <param name="info"></param>
            public static void LearnAll(InputDataManager inputDataManager, bool tuneLR,
                ref float lr, float l2Const, float piw, Span<float> weightVector, ref float bias, int numFeatres, int numPasses,
                int numThreads, bool tuneNumLocIter, ref int numLocIter, float tolerance, bool needShuffle, bool shouldInitialize, GCHandle stateGCHandle, ChannelCallBack info)
            {
                inputDataManager.PrepareCursoring();

                int totalNumInstances = inputDataManager.Count;
                // Each instance has a pointer to indices array and a pointer to values array
                int*[] arrayIndicesPointers = new int*[totalNumInstances];
                float*[] arrayValuesPointers = new float*[totalNumInstances];
                // Labels of the instances
                float[] instLabels = new float[totalNumInstances];
                // Sizes of each inst
                int[] instSizes = new int[totalNumInstances];

                int instanceIndex = 0;
                // Going through the buffer to set the properties and the pointers
                while (inputDataManager.GiveNextInstance(out InstanceProperties? prop, out GCHandle? indicesGcHandle, out int indicesStartIndex, out GCHandle? valuesGcHandle, out int valuesStartIndex))
                {
                    if (prop.Value.IsDense)
                    {
                        arrayIndicesPointers[instanceIndex] = null;
                    }
                    else
                    {
                        int* pIndicesArray = (int*)indicesGcHandle.Value.AddrOfPinnedObject();
                        arrayIndicesPointers[instanceIndex] = &pIndicesArray[indicesStartIndex];
                    }
                    float* pValuesArray = (float*)valuesGcHandle.Value.AddrOfPinnedObject();
                    arrayValuesPointers[instanceIndex] = &pValuesArray[valuesStartIndex];

                    instLabels[instanceIndex] = prop.Value.Label;
                    instSizes[instanceIndex] = prop.Value.FeatureCount;
                    instanceIndex++;
                }

                fixed (float* pweightVector = &weightVector[0])
                fixed (int** pIndicesPointer = &arrayIndicesPointers[0])
                fixed (float** pValuesPointer = &arrayValuesPointers[0])
                fixed (int* pInstSizes = &instSizes[0])
                fixed (float* pInstLabels = &instLabels[0])
                {
                    LearnAll(totalNumInstances, pInstSizes, pIndicesPointer, pValuesPointer, pInstLabels, tuneLR, ref lr, l2Const, piw,
                            pweightVector, ref bias, numFeatres, numPasses, numThreads, tuneNumLocIter, ref numLocIter, tolerance, needShuffle,
                            shouldInitialize, (State*)stateGCHandle.AddrOfPinnedObject(), info);
                }
            }

            [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
            private static extern void MapBackWeightVector(float* weightVector, State* state);

            /// <summary>
            /// Maps back the dense feature to the correct position
            /// </summary>
            /// <param name="weightVector">The weight vector</param>
            /// <param name="stateGCHandle"></param>
            public static void MapBackWeightVector(Span<float> weightVector, GCHandle stateGCHandle)
            {
                fixed (float* pweightVector = &weightVector[0])
                    MapBackWeightVector(pweightVector, (State*)stateGCHandle.AddrOfPinnedObject());
            }

            [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
            private static extern void DeallocateSequentially(State* state);

            public static void DeallocateSequentially(GCHandle stateGCHandle)
            {
                DeallocateSequentially((State*)stateGCHandle.AddrOfPinnedObject());
            }

            // See: https://software.intel.com/en-us/node/521990
            [DllImport(MklPath, EntryPoint = "DftiErrorMessage", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto), SuppressUnmanagedCodeSecurity]
            private static extern IntPtr ErrorMessage(int status);
        }

        /// <summary>
        /// This is the state of a SymSGD learner that is shared between the managed and native code.
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        internal unsafe struct State
        {
#pragma warning disable 649 // never assigned
            [FieldOffset(0x00)]
            public readonly int NumLearners;
            [FieldOffset(0x04)]
            public int TotalInstancesProcessed;
            [FieldOffset(0x08)]
            public readonly void* Learners;
            [FieldOffset(0x10)]
            public readonly void* FreqFeatUnorderedMap;
            [FieldOffset(0x18)]
            public readonly int* FreqFeatDirectMap;
            [FieldOffset(0x20)]
            public readonly int NumFrequentFeatures;
            [FieldOffset(0x24)]
            public int PassIteration;
            [FieldOffset(0x28)]
            public readonly float WeightScaling;
#pragma warning restore 649 // never assigned
        }
    }

}
