// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers.SymSgd;
using Microsoft.ML.Runtime.Training;

[assembly: LoadableClass(typeof(SymSgdClassificationTrainer), typeof(SymSgdClassificationTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    SymSgdClassificationTrainer.UserNameValue,
    SymSgdClassificationTrainer.LoadNameValue,
    SymSgdClassificationTrainer.ShortName)]

[assembly: LoadableClass(typeof(void), typeof(SymSgdClassificationTrainer), null, typeof(SignatureEntryPointModule), SymSgdClassificationTrainer.LoadNameValue)]

namespace Microsoft.ML.Trainers.SymSgd
{
    using TPredictor = IPredictorWithFeatureWeights<float>;

    /// <include file='doc.xml' path='doc/members/member[@name="SymSGD"]/*' />
    public sealed class SymSgdClassificationTrainer : TrainerEstimatorBase<BinaryPredictionTransformer<TPredictor>, TPredictor>
    {
        internal const string LoadNameValue = "SymbolicSGD";
        internal const string UserNameValue = "Symbolic SGD (binary)";
        internal const string ShortName = "SymSGD";

        public sealed class Arguments : LearnerInputBaseWithLabel
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Degree of lock-free parallelism. Determinism not guaranteed. " +
                "Multi-threading is not supported currently.", ShortName = "nt")]
            public int? NumberOfThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of passes over the data.", ShortName = "iter", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "1,5,10,20,30,40,50")]
            [TlcModule.SweepableDiscreteParam("NumberOfIterations", new object[] { 1, 5, 10, 20, 30, 40, 50 })]
            public int NumberOfIterations = 50;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Tolerance for difference in average loss in consecutive passes.", ShortName = "tol")]
            public float Tolerance = 1e-4f;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Learning rate", ShortName = "lr", NullName = "<Auto>", SortOrder = 51)]
            [TGUI(SuggestedSweeps = "<Auto>,1e1,1e0,1e-1,1e-2,1e-3")]
            [TlcModule.SweepableDiscreteParam("LearningRate", new object[] { "<Auto>", 1e1f, 1e0f, 1e-1f, 1e-2f, 1e-3f })]
            public float? LearningRate;

            [Argument(ArgumentType.AtMostOnce, HelpText = "L2 regularization", ShortName = "l2", SortOrder = 52)]
            [TGUI(SuggestedSweeps = "0.0,1e-5,1e-5,1e-6,1e-7")]
            [TlcModule.SweepableDiscreteParam("L2Regularization", new object[] { 0.0f, 1e-5f, 1e-5f, 1e-6f, 1e-7f })]
            public float L2Regularization;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of iterations each thread learns a local model until combining it with the " +
                "global model. Low value means more updated global model and high value means less cache traffic.", ShortName = "freq", NullName = "<Auto>")]
            [TGUI(SuggestedSweeps = "<Auto>,5,20")]
            [TlcModule.SweepableDiscreteParam("UpdateFrequency", new object[] { "<Auto>", 5, 20 })]
            public int? UpdateFrequency;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The acceleration memory budget in MB", ShortName = "accelMemBudget")]
            public long MemorySize = 1024;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Shuffle data?", ShortName = "shuf")]
            public bool Shuffle = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Apply weight to the positive class, for imbalanced data", ShortName = "piw")]
            public float PositiveInstanceWeight = 1;

            public void Check(IExceptionContext ectx)
            {
                ectx.CheckUserArg(LearningRate == null || LearningRate.Value > 0, nameof(LearningRate), "Must be positive.");
                ectx.CheckUserArg(NumberOfIterations > 0, nameof(NumberOfIterations), "Must be positive.");
                ectx.CheckUserArg(PositiveInstanceWeight > 0, nameof(PositiveInstanceWeight), "Must be positive");
                ectx.CheckUserArg(UpdateFrequency == null || UpdateFrequency > 0, nameof(UpdateFrequency), "Must be positive");
            }
        }

        public override TrainerInfo Info { get; }
        private readonly Arguments _args;

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
                var shuffleArgs = new ShuffleTransform.Arguments
                {
                    PoolOnly = false,
                    ForceShuffle = _args.Shuffle
                };
                idvToFeedTrain = new ShuffleTransform(Host, shuffleArgs, idvToShuffle);
            }

            ch.Assert(idvToFeedTrain.CanShuffle);

            var roles = examples.Schema.GetColumnRoleNames();
            var examplesToFeedTrain = new RoleMappedData(idvToFeedTrain, roles);

            ch.AssertValue(examplesToFeedTrain.Schema.Label);
            ch.AssertValue(examplesToFeedTrain.Schema.Feature);
            if (examples.Schema.Weight != null)
                ch.AssertValue(examplesToFeedTrain.Schema.Weight);

            int numFeatures = examplesToFeedTrain.Schema.Feature.Type.VectorSize;
            ch.Check(numFeatures > 0, "Training set has no features, aborting training.");
            return examplesToFeedTrain;
        }

        protected override TPredictor TrainModelCore(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            using (var ch = Host.Start("Training"))
            {
                var preparedData = PrepareDataFromTrainingExamples(ch, context.TrainingSet, out int weightSetCount);
                var initPred = context.InitialPredictor;
                var linInitPred = (initPred as CalibratedPredictorBase)?.SubPredictor as LinearPredictor;
                linInitPred = linInitPred ?? initPred as LinearPredictor;
                Host.CheckParam(context.InitialPredictor == null || linInitPred != null, nameof(context),
                    "Initial predictor was not a linear predictor.");
                return TrainCore(ch, preparedData, linInitPred, weightSetCount);
            }
        }

        public override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        /// <summary>
        /// Initializes a new instance of <see cref="SymSgdClassificationTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumn">The name of the label column.</param>
        /// <param name="featureColumn">The name of the feature column.</param>
        /// <param name="advancedSettings">A delegate to apply all the advanced arguments to the algorithm.</param>
        public SymSgdClassificationTrainer(IHostEnvironment env, string featureColumn, string labelColumn, Action<Arguments> advancedSettings = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadNameValue), TrainerUtils.MakeR4VecFeature(featureColumn),
                  TrainerUtils.MakeBoolScalarLabel(labelColumn))
        {
            _args = new Arguments();

            // Apply the advanced args, if the user supplied any.
            _args.Check(Host);
            advancedSettings?.Invoke(_args);
            _args.FeatureColumn = featureColumn;
            _args.LabelColumn = labelColumn;

            Info = new TrainerInfo();
        }

        /// <summary>
        /// Initializes a new instance of <see cref="SymSgdClassificationTrainer"/>
        /// </summary>
        internal SymSgdClassificationTrainer(IHostEnvironment env, Arguments args)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadNameValue), TrainerUtils.MakeR4VecFeature(args.FeatureColumn),
                  TrainerUtils.MakeBoolScalarLabel(args.LabelColumn))
        {
            args.Check(Host);
            _args = args;
            Info = new TrainerInfo();
        }

        private TPredictor CreatePredictor(VBuffer<float> weights, float bias)
        {
            Host.CheckParam(weights.Length > 0, nameof(weights));

            VBuffer<float> maybeSparseWeights = default;
            VBufferUtils.CreateMaybeSparseCopy(in weights, ref maybeSparseWeights,
                Conversions.Instance.GetIsDefaultPredicate<float>(NumberType.R4));
            var predictor = new LinearBinaryPredictor(Host, ref maybeSparseWeights, bias);
            return new ParameterMixingCalibratedPredictor(Host, predictor, new PlattCalibrator(Host, -1, 0));
        }

        protected override BinaryPredictionTransformer<TPredictor> MakeTransformer(TPredictor model, Schema trainSchema)
             => new BinaryPredictionTransformer<TPredictor>(Host, model, trainSchema, FeatureColumn.Name);

        public BinaryPredictionTransformer<TPredictor> Train(IDataView trainData, IDataView validationData = null, TPredictor initialPredictor = null)
            => TrainTransformer(trainData, validationData, initialPredictor);

        protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberType.R4, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BoolType.Instance, false, new SchemaShape(MetadataUtils.GetTrainerOutputMetadata()))
            };
        }

        [TlcModule.EntryPoint(Name = "Trainers.SymSgdBinaryClassifier",
            Desc = "Train a symbolic SGD.",
            UserName = SymSgdClassificationTrainer.UserNameValue,
            ShortName = SymSgdClassificationTrainer.ShortName,
            XmlInclude = new[] { @"<include file='../Microsoft.ML.HalLearners/doc.xml' path='doc/members/member[@name=""SymSGD""]/*' />" })]
        public static CommonOutputs.BinaryClassificationOutput TrainSymSgd(IHostEnvironment env, Arguments input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainSymSGD");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return LearnerEntryPointsUtils.Train<Arguments, CommonOutputs.BinaryClassificationOutput>(host, input,
                () => new SymSgdClassificationTrainer(host, input),
                () => LearnerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumn));
        }

        // We buffer instances from the cursor (limited to memorySize) and passes that buffer to
        // the native code to learn for multiple instances by one interop call.

        /// <summary>
        /// This struct holds the information about the size, label and isDense of each instance
        /// to be able to pass it to the native code.
        /// </summary>
        private struct InstanceProperties
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
            private readonly SymSgdClassificationTrainer _trainer;

            private readonly IChannel _ch;

            // Size of type T
            private readonly int _sizeofT;

            /// <summary>
            /// Constructor for initializing _storage and other indices.
            /// </summary>
            /// <param name="trainer"></param>
            /// <param name="ch"></param>
            public ArrayManager(SymSgdClassificationTrainer trainer, IChannel ch)
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
            /// Tries to add array <paramref name="instArray"/> to the storage without violating the restriction of memorySize.
            /// </summary>
            /// <param name="instArray">The array to be added</param>
            /// <param name="instArrayLength">Length of the array. <paramref name="instArray"/>.Length is unreliable since TLC cursoring
            /// has its own allocation mechanism.</param>
            /// <returns>Return if the allocation was successful</returns>
            public bool AddToStorage(T[] instArray, int instArrayLength)
            {
                _ch.Assert(0 < instArrayLength && instArrayLength <= Utils.Size(instArray));
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
                Array.Copy(instArray, 0, _storage[_storageIndex].Buffer, _indexInCurArray, instArrayLength);
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
            private readonly SymSgdClassificationTrainer _trainer;
            private readonly IChannel _ch;

            // Whether memorySize was big enough to load the entire instances into the buffer
            private bool _isFullyLoaded;
            public bool IsFullyLoaded => _isFullyLoaded;
            public int Count => _instanceProperties.Count;

            // Tells if we have gone through the dataset entirely.
            public bool FinishedTheLoad => !_cursorMoveNext;

            public InputDataManager(SymSgdClassificationTrainer trainer, FloatLabelCursor.Factory cursorFactory, IChannel ch)
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
                    int featureCount = _cursor.Features.Count;
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
                        couldLoad = _instIndices.AddToStorage(_cursor.Features.Indices, featureCount);
                    // Load values of an instance into instValues
                    if (couldLoad)
                        couldLoad = _instValues.AddToStorage(_cursor.Features.Values, featureCount);

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

        private TPredictor TrainCore(IChannel ch, RoleMappedData data, LinearPredictor predictor, int weightSetCount)
        {
            int numFeatures = data.Schema.Feature.Type.VectorSize;
            var cursorFactory = new FloatLabelCursor.Factory(data, CursOpt.Label | CursOpt.Features | CursOpt.Weight);
            int numThreads = 1;
            ch.CheckUserArg(numThreads > 0, nameof(_args.NumberOfThreads),
                "The number of threads must be either null or a positive integer.");

            var positiveInstanceWeight = _args.PositiveInstanceWeight;
            VBuffer<float> weights = default;
            float bias = 0.0f;
            if (predictor != null)
            {
                predictor.GetFeatureWeights(ref weights);
                VBufferUtils.Densify(ref weights);
                bias = predictor.Bias;
            }
            else
                weights = VBufferUtils.CreateDense<float>(numFeatures);

            // Reference: Parasail. SymSGD.
            bool tuneLR = _args.LearningRate == null;
            var lr = _args.LearningRate ?? 1.0f;

            bool tuneNumLocIter = (_args.UpdateFrequency == null);
            var numLocIter = _args.UpdateFrequency ?? 1;

            var l2Const = _args.L2Regularization;
            var piw = _args.PositiveInstanceWeight;

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
                                entry => entry.SetProgress(0, state.PassIteration, _args.NumberOfIterations));
                            // If fully loaded, call the SymSGDNative and do not come back until learned for all iterations.
                            Native.LearnAll(inputDataManager, tuneLR, ref lr, l2Const, piw, weights.Values, ref bias, numFeatures,
                                _args.NumberOfIterations, numThreads, tuneNumLocIter, ref numLocIter, _args.Tolerance, _args.Shuffle, shouldInitialize, stateGCHandle);
                            shouldInitialize = false;
                        }
                        else
                        {
                            pch.SetHeader(new ProgressHeader(new[] { "iterations" }),
                                entry => entry.SetProgress(0, iter, _args.NumberOfIterations));

                            // Since we loaded data in batch sizes, multiple passes over the loaded data is feasible.
                            int numPassesForABatch = inputDataManager.Count / 10000;
                            while (iter < _args.NumberOfIterations)
                            {
                                // We want to train on the final passes thoroughly (without learning on the same batch multiple times)
                                // This is for fine tuning the AUC. Experimentally, we found that 1 or 2 passes is enough
                                int numFinalPassesToTrainThoroughly = 2;
                                // We also do not want to learn for more passes than what the user asked
                                int numPassesForThisBatch = Math.Min(numPassesForABatch, _args.NumberOfIterations - iter - numFinalPassesToTrainThoroughly);
                                // If all of this leaves us with 0 passes, then set numPassesForThisBatch to 1
                                numPassesForThisBatch = Math.Max(1, numPassesForThisBatch);
                                state.PassIteration = iter;
                                Native.LearnAll(inputDataManager, tuneLR, ref lr, l2Const, piw, weights.Values, ref bias, numFeatures,
                                    numPassesForThisBatch, numThreads, tuneNumLocIter, ref numLocIter, _args.Tolerance, _args.Shuffle, shouldInitialize, stateGCHandle);
                                shouldInitialize = false;

                                // Check if we are done with going through the data
                                if (inputDataManager.FinishedTheLoad)
                                {
                                    iter += numPassesForThisBatch;
                                    // Check if more passes are left
                                    if (iter < _args.NumberOfIterations)
                                        inputDataManager.RestartLoading(_args.Shuffle, Host);
                                }

                                // If more passes are left, load as much as possible
                                if (iter < _args.NumberOfIterations)
                                    inputDataManager.LoadAsMuchAsPossible();
                            }
                        }

                        // Maps back the dense features that are mislocated
                        if (numThreads > 1)
                            Native.MapBackWeightVector(weights.Values, stateGCHandle);
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

        private long AcceleratedMemoryBudgetBytes => _args.MemorySize * 1024 * 1024;
        private long UsedMemory { get; set; }

        private static unsafe class Native
        {
            //To triger the loading of MKL library since SymSGD native library depends on it.
            static Native() => ErrorMessage(0);

            internal const string DllName = "SymSgdNative";

            [DllImport(DllName), SuppressUnmanagedCodeSecurity]
            private static extern void LearnAll(int totalNumInstances, int* instSizes, int** instIndices,
                float** instValues, float* labels, bool tuneLR, ref float lr, float l2Const, float piw, float* weightVector, ref float bias,
                int numFeatres, int numPasses, int numThreads, bool tuneNumLocIter, ref int numLocIter, float tolerance, bool needShuffle, bool shouldInitialize, State* state);

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
            public static void LearnAll(InputDataManager inputDataManager, bool tuneLR,
                ref float lr, float l2Const, float piw, float[] weightVector, ref float bias, int numFeatres, int numPasses,
                int numThreads, bool tuneNumLocIter, ref int numLocIter, float tolerance, bool needShuffle, bool shouldInitialize, GCHandle stateGCHandle)
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
                            pweightVector, ref bias, numFeatres, numPasses, numThreads, tuneNumLocIter, ref numLocIter, tolerance, needShuffle, shouldInitialize, (State*)stateGCHandle.AddrOfPinnedObject());
                }
            }

            [DllImport(DllName), SuppressUnmanagedCodeSecurity]
            private static extern void MapBackWeightVector(float* weightVector, State* state);

            /// <summary>
            /// Maps back the dense feature to the correct position
            /// </summary>
            /// <param name="weightVector">The weight vector</param>
            /// <param name="stateGCHandle"></param>
            public static void MapBackWeightVector(float[] weightVector, GCHandle stateGCHandle)
            {
                fixed (float* pweightVector = &weightVector[0])
                    MapBackWeightVector(pweightVector, (State*)stateGCHandle.AddrOfPinnedObject());
            }

            [DllImport(DllName), SuppressUnmanagedCodeSecurity]
            private static extern void DeallocateSequentially(State* state);

            public static void DeallocateSequentially(GCHandle stateGCHandle)
            {
                DeallocateSequentially((State*)stateGCHandle.AddrOfPinnedObject());
            }

            // See: https://software.intel.com/en-us/node/521990
            [System.Security.SuppressUnmanagedCodeSecurity]
            [DllImport("MklImports", EntryPoint = "DftiErrorMessage", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
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
