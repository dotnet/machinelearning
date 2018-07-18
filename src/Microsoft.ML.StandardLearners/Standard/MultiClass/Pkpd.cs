// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(Pkpd.Summary, typeof(Pkpd), typeof(Pkpd.Arguments),
    new[] { typeof(SignatureMultiClassClassifierTrainer), typeof(SignatureTrainer) },
    Pkpd.UserNameValue, Pkpd.LoadNameValue, DocName = "trainer/OvaPkpd.md")]

[assembly: LoadableClass(typeof(PkpdPredictor), null, typeof(SignatureLoadModel),
    "PKPD Executor",
    PkpdPredictor.LoaderSignature)]

namespace Microsoft.ML.Runtime.Learners
{
    using TScalarTrainer = ITrainer<RoleMappedData, IPredictorProducing<Float>>;
    using TScalarPredictor = IPredictorProducing<Float>;
    using TDistPredictor = IDistPredictorProducing<Float, Float>;
    using CR = RoleMappedSchema.ColumnRole;

    /// <summary>
    /// In this strategy, a binary classification algorithm is trained on each pair of classes. 
    /// The pairs are unordered but created with replacement: so, if there were three classes, 0, 1,
    /// 2, we would train classifiers for the pairs (0,0), (0,1), (0,2), (1,1), (1,2),
    /// and(2,2). For each binary classifier, an input data point is considered a
    /// positive example if it is in either of the two classes in the pair, and a
    /// negative example otherwise. At prediction time, the probabilities for each
    /// pair of classes is considered as the probability of being in either class of
    /// the pair given the data, and the final predictive probabilities out of that
    /// per class are calculated given the probability that an example is in any given
    /// pair.
    ///
    /// These two can allow you to exploit trainers that do not naturally have a
    /// multiclass option, e.g., using the Runtime.FastTree.FastTreeBinaryClassificationTrainer 
    /// to solve a multiclass problem.
    /// Alternately, it can allow ML.NET to solve a "simpler" problem even in the cases
    /// where the trainer has a multiclass option, but using it directly is not
    /// practical due to, usually, memory constraints.For example, while a multiclass
    /// logistic regression is a more principled way to solve a multiclass problem, it
    /// requires that the learner store a lot more intermediate state in the form of
    /// L-BFGS history for all classes *simultaneously*, rather than just one-by-one
    /// as would be needed for OVA.
    /// </summary>
    public sealed class Pkpd : MetaMulticlassTrainer<PkpdPredictor, Pkpd.Arguments>
    {
        internal const string LoadNameValue = "PKPD";
        internal const string UserNameValue = "Pairwise coupling (PKPD)";
        internal const string Summary = "In this strategy, a binary classification algorithm is used to train one classifier for each pair of classes. "
            + "Prediction is then performed by running these binary classifiers, and computing a score for each class by counting how many of the binary "
            + "classifiers predicted it. The prediction is the class with the highest score.";

        /// <summary>
        /// Arguments passed to PKPD.
        /// </summary>
        public sealed class Arguments : ArgumentsBase
        {
        }

        public Pkpd(IHostEnvironment env, Arguments args)
            : base(env, args, LoadNameValue)
        {
        }

        protected override PkpdPredictor TrainCore(IChannel ch, RoleMappedData data, int count)
        {
            // Train M * (M+1) / 2 models arranged as a lower triangular matrix.
            TDistPredictor[][] predictors;
            predictors = new TDistPredictor[count][];
            for (int i = 0; i < predictors.Length; i++)
            {
                predictors[i] = new TDistPredictor[i + 1];
                for (int j = 0; j <= i; j++)
                {
                    ch.Info($"Training learner ({i},{j})");
                    predictors[i][j] = TrainOne(ch, GetTrainer(), data, i, j);
                }
            }
            return new PkpdPredictor(Host, predictors);
        }

        private TDistPredictor TrainOne(IChannel ch, TScalarTrainer trainer, RoleMappedData data, int cls1, int cls2)
        {
            string dstName;
            var view = MapLabels(data, cls1, cls2, out dstName);

            var roles = data.Schema.GetColumnRoleNames()
                .Where(kvp => kvp.Key.Value != CR.Label.Value)
                .Prepend(CR.Label.Bind(dstName));
            var td = new RoleMappedData(view, roles);

            trainer.Train(td);

            ICalibratorTrainer calibrator;
            if (!Args.Calibrator.IsGood())
                calibrator = null;
            else
                calibrator = Args.Calibrator.CreateInstance(Host);
            TScalarPredictor predictor = trainer.CreatePredictor();
            var res = CalibratorUtils.TrainCalibratorIfNeeded(Host, ch, calibrator, Args.MaxCalibrationExamples,
                trainer, predictor, td);
            var dist = res as TDistPredictor;
            Host.Check(dist != null, "Calibrated predictor does not implement the expected interface");
            Host.Check(dist is IValueMapperDist, "Calibrated predictor does not implement the IValueMapperDist interface");
            return dist;
        }

        private IDataView MapLabels(RoleMappedData data, int cls1, int cls2, out string dstName)
        {
            var lab = data.Schema.Label;
            Host.Assert(!data.Schema.Schema.IsHidden(lab.Index));
            Host.Assert(lab.Type.KeyCount > 0 || lab.Type == NumberType.R4 || lab.Type == NumberType.R8);

            // Get the destination label column name.
            dstName = data.Schema.Schema.GetTempColumnName();

            if (lab.Type.KeyCount > 0)
            {
                // Key values are 1-based.
                uint key1 = (uint)(cls1 + 1);
                uint key2 = (uint)(cls2 + 1);
                return MapLabelsCore(NumberType.U4, (ref uint val) => val == key1 || val == key2, data, dstName);
            }
            if (lab.Type == NumberType.R4)
            {
                float key1 = cls1;
                float key2 = cls2;
                return MapLabelsCore(NumberType.R4, (ref float val) => val == key1 || val == key2, data, dstName);
            }
            if (lab.Type == NumberType.R8)
            {
                double key1 = cls1;
                double key2 = cls2;
                return MapLabelsCore(NumberType.R8, (ref double val) => val == key1 || val == key2, data, dstName);
            }

            throw Host.ExceptNotSupp($"Label column type is not supported by PKPD: {lab.Type}");
        }
    }

    public sealed class PkpdPredictor :
        PredictorBase<VBuffer<Float>>,
        IValueMapper,
        ICanSaveModel
    {
        public const string LoaderSignature = "PKPDExec";
        public const string RegistrationName = "PKPDPredictor";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TLC PKPD",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string SubPredictorFmt = "SubPredictor_{0:000}";
        private const string SubPredictorFmt2 = "SubPredictor_{0:000}_{1:000}";

        private readonly int _numClasses;
        // There are M * (M + 1) / 2 predictors. These are really organized as a lower triangular
        // matrix of predictors, with indices (0,0) (1,0) (1,1) (2,0) (2,1) (2,2), etc. We store them
        // in a 1-d array mostly to make it easy to parallelize.
        private readonly TDistPredictor[] _predictors;
        private readonly IValueMapperDist[] _mappers;

        public override PredictionKind PredictionKind => PredictionKind.MultiClassClassification;
        public ColumnType InputType { get; }
        public ColumnType OutputType { get; }

        internal PkpdPredictor(IHostEnvironment env, TDistPredictor[][] predictors) :
            base(env, RegistrationName)
        {
            Host.Assert(Utils.Size(predictors) > 0);

            // We store the predictors in a 1-d array to facilitate parallelizing the Predict calls.
            _numClasses = predictors.Length;
            _predictors = new TDistPredictor[checked(_numClasses * (_numClasses + 1) / 2)];
            int index = 0;
            for (int i = 0; i < _numClasses; i++)
            {
                Host.Assert(Utils.Size(predictors[i]) == i + 1);
                for (int j = 0; j <= i; j++)
                {
                    Host.Assert(index == GetIndex(i, j));
                    _predictors[index++] = predictors[i][j];
                }
            }
            Host.Assert(index == _predictors.Length);

            InputType = InitializeMappers(out _mappers);
            OutputType = new VectorType(NumberType.Float, _numClasses);
        }

        private PkpdPredictor(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx)
        {
            // *** Binary format ***
            // int: number of classes
            _numClasses = ctx.Reader.ReadInt32();
            Host.CheckDecode(_numClasses > 0);

            long count = (long)_numClasses * (_numClasses + 1) / 2;
            Host.CheckDecode(count <= int.MaxValue);

            // Load the predictors.
            _predictors = new TDistPredictor[(int)count];
            int index = 0;
            for (int i = 0; i < _numClasses; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    Host.Assert(index == GetIndex(i, j));
                    ctx.LoadModel<TDistPredictor, SignatureLoadModel>(Host, out _predictors[index++], string.Format(SubPredictorFmt2, i, j));
                }
                Host.Assert(index == GetIndex(i, i));
                ctx.LoadModel<TDistPredictor, SignatureLoadModel>(Host, out _predictors[index++], string.Format(SubPredictorFmt, i));
            }
            InputType = InitializeMappers(out _mappers);
            OutputType = new VectorType(NumberType.Float, _numClasses);
        }

        private ColumnType InitializeMappers(out IValueMapperDist[] mappers)
        {
            mappers = new IValueMapperDist[_predictors.Length];
            ColumnType inputType = null;
            for (int i = 0; i < _predictors.Length; i++)
            {
                var vmd = _predictors[i] as IValueMapperDist;
                Host.Check(IsValid(vmd, ref inputType), "Predictor doesn't implement the expected interface");
                mappers[i] = vmd;
            }
            return inputType;
        }

        private bool IsValid(IValueMapperDist mapper, ref ColumnType inputType)
        {
            if (mapper == null)
                return false;
            if (!mapper.InputType.IsKnownSizeVector || mapper.InputType.ItemType != NumberType.Float)
                return false;
            if (inputType == null)
                inputType = mapper.InputType;
            else if (inputType.VectorSize != mapper.InputType.VectorSize)
                return false;
            if (mapper.OutputType != NumberType.Float)
                return false;
            if (mapper.DistType != NumberType.Float)
                return false;
            return true;
        }

        public static PkpdPredictor Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new PkpdPredictor(env, ctx);
        }

        protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of classes
            ctx.Writer.Write(_numClasses);

            // Save other streams.
            for (int i = 0; i < _numClasses; i++)
            {
                int index = GetIndex(i, 0);
                ctx.SaveModel(_predictors[index + i], string.Format(SubPredictorFmt, i));
                for (int j = 0; j < i; j++)
                    ctx.SaveModel(_predictors[index + j], string.Format(SubPredictorFmt2, i, j));
            }
        }

        private void ComputeProbabilities(Double[] buffer, ref Float[] output)
        {
            // Compute the probabilities and store them in the beginning of buffer. Note that this is safe to do since
            // once we've computed the ith probability, we are totally done with the ith row and all previous rows
            // (in the lower triangular matrix of pairwise probabilities).
            Double sum = 0;
            for (int i = 0; i < _numClasses; i++)
            {
                var value = buffer[i] = Pi(i, buffer);
                Host.Assert(0 <= value && value <= 1);
                sum += value;
            }

            if (Utils.Size(output) < _numClasses)
                output = new Float[_numClasses];

            // Normalize.
            if (sum <= 0)
                sum = 1;

            for (int i = 0; i < _numClasses; i++)
                output[i] = (Float)(buffer[i] / sum);
        }

        // Reconcile the predictions - ensure that pij >= pii and pji >= pii (when pii > 0).
        private void ReconcilePredictions(Double[] buffer)
        {
            for (int i = 0; i < _numClasses; i++)
            {
                int index = GetIndex(i, 0);
                var pii = buffer[index + i];
                if (!(pii > 0))
                    continue;

                for (int j = 0; j < i; j++)
                {
                    if (!(buffer[index + j] >= pii))
                        buffer[index + j] = pii;
                }
                index += i;
                for (int j = i + 1; j < _numClasses; j++)
                {
                    index += j; // Move past previous row.
                    Host.Assert(index == GetIndex(j, i));
                    if (!(buffer[index] >= pii))
                        buffer[index] = pii;
                }
            }
        }

        private Double Pi(int i, Double[] values)
        {
            // values is the lower triangular matrix of pairwise probabilities pij = P(y=i or y=j | x).
            // Get pii = P(y=i | x)
            int index = GetIndex(i, 0);
            Double pii = values[index + i];

            if (!(pii > 0))
                return 0;

            // Compute sum { pij | j != i }
            Double sum = 0;
            for (int j = 0; j < i; j++)
            {
                Host.Assert(values[index + j] >= pii);
                sum += values[index + j];
            }
            index += i;
            for (int j = i + 1; j < _numClasses; j++)
            {
                index += j; // Move past previous row.
                Host.Assert(index == GetIndex(j, i));
                Host.Assert(values[index] >= pii);
                sum += values[index];
            }

            // Return pii / (sum - (k - 2) * pii).
            // Taking max simply protects against round-off error - in exact math it wouldn't be needed.
            return pii / Math.Max(pii, sum - (_numClasses - 2) * pii);
        }

        private int GetIndex(int i, int j)
        {
            Host.Assert(0 <= j && j <= i && i < _numClasses);
            return i * (i + 1) / 2 + j;
        }

        public ValueMapper<TIn, TOut> GetMapper<TIn, TOut>()
        {
            Host.Check(typeof(TIn) == typeof(VBuffer<Float>));
            Host.Check(typeof(TOut) == typeof(VBuffer<Float>));

            var maps = new ValueMapper<VBuffer<Float>, Float, Float>[_mappers.Length];
            for (int i = 0; i < _mappers.Length; i++)
                maps[i] = _mappers[i].GetMapper<VBuffer<Float>, Float, Float>();

            var buffer = new Double[_predictors.Length];
            ValueMapper<VBuffer<Float>, VBuffer<Float>> del =
                (ref VBuffer<Float> src, ref VBuffer<Float> dst) =>
                {
                    if (InputType.VectorSize > 0)
                        Host.Check(src.Length == InputType.VectorSize);

                    var values = dst.Values;
                    var tmp = src;
                    Parallel.For(0, maps.Length, i =>
                    {
                        Float score = 0;
                        Float prob = 0;
                        maps[i](ref tmp, ref score, ref prob);
                        buffer[i] = prob;
                    });

                    ReconcilePredictions(buffer);
                    ComputeProbabilities(buffer, ref values);

                    dst = new VBuffer<Float>(_numClasses, values, dst.Indices);
                };
            return (ValueMapper<TIn, TOut>)(Delegate)del;
        }
    }
}
