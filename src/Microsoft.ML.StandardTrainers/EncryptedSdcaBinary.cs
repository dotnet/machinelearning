// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.Research.SEAL;

namespace Microsoft.ML.SEAL
{
    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training an encrypted binary logistic regression classification model using the stochastic dual coordinate ascent method.
    /// The trained model is <a href='https://en.wikipedia.org/wiki/Calibration_(statistics)'>calibrated</a> and can produce probability by feeding the output value of the
    /// linear function to a <see cref="PlattCalibrator"/>.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [EncryptedSdcaLogisticRegression](xref:Microsoft.ML.EncryptedStandardTrainersCatalog.EncryptedSdcaLogisticRegression(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,System.String,System.String,System.String,System.Nullable{System.Single},System.Nullable{System.Single},System.Nullable{System.Int32}))
    /// or [EncryptedSdcaLogisticRegression(Options)](xref:Microsoft.ML.EncryptedStandardTrainersCatalog.EncryptedSdcaLogisticRegression(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,Microsoft.ML.Trainers.EncryptedSdcaLogisticRegressionBinaryTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-binary-classification.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Binary classification |
    /// | Is normalization required? | Yes |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | None |
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/algo-details-sdca.md)]
    ///
    /// [!include[regularization](~/../docs/samples/docs/api-reference/regularization-l1-l2.md)]
    ///
    /// [!include[references](~/../docs/samples/docs/api-reference/algo-details-sdca-refs.md)]
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="EncryptedStandardTrainersCatalog.EncryptedSdcaLogisticRegression(BinaryClassificationCatalog.BinaryClassificationTrainers, ulong, IEnumerable&lt;SmallModulus&gt;, double, string, string, string, string, float?, float?, int?)"/>
    /// <seealso cref="EncryptedStandardTrainersCatalog.EncryptedSdcaLogisticRegression(BinaryClassificationCatalog.BinaryClassificationTrainers, EncryptedSdcaLogisticRegressionBinaryTrainer.Options)"/>
    /// <seealso cref="Options"/>
    public sealed class EncryptedSdcaLogisticRegressionBinaryTrainer :
        SdcaBinaryTrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        private readonly ulong _polyModulusDegree;
        private readonly IEnumerable<SmallModulus> _coeffModuli;
        private readonly double _scale;
        private readonly String _encryptedFeatureColumnName;

        /// <summary>
        /// Options for the <see cref="EncryptedSdcaLogisticRegressionBinaryTrainer"/> as used in
        /// [EncryptedSdcaLogisticRegression(Options)](xref:Microsoft.ML.EncryptedStandardTrainersCatalog.EncryptedSdcaLogisticRegression(Microsoft.ML.BinaryClassificationCatalog.BinaryClassificationTrainers,Microsoft.ML.Trainers.EncryptedSdcaLogisticRegressionBinaryTrainer.Options)).
        /// </summary>
        public sealed class Options : BinaryOptionsBase
        {
            /// <summary>
            /// The value of the PolyModulusDegree encryption parameter.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "Polynomial Modulus Degree", ShortName = "pmd")]
            public ulong PolyModDegree;

            /// <summary>
            /// The bit-lengths of the primes to be generated.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "The bit-lengths of the primes to be generated", ShortName = "bs")]
            public IEnumerable<SmallModulus> CoeffModuli;

            /// <summary>
            /// Scaling parameter defining encoding precision.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "Scaling parameter defining encoding precision", ShortName = "s")]
            public double Scale;

            /// <summary>
            /// Name of column containing the encrypted data.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "Name of column containing the encrypted data", ShortName = "encCol")]
            public String EncryptedFeatureColumnName;

            internal override void Check(IHostEnvironment env)
            {
                base.Check(env);
                env.CheckUserArg(PolyModDegree > 0, nameof(PolyModDegree), "Polynomial modulus degree must be positive");
                env.CheckUserArg(Scale > 0, nameof(Scale), "Scale must be positive");

                foreach(var coeffModulus in CoeffModuli)
                {
                    env.CheckUserArg(coeffModulus != null, nameof(CoeffModuli), "Null coefficient moduli are not allowed");
                }
            }
        }

        internal EncryptedSdcaLogisticRegressionBinaryTrainer(IHostEnvironment env,
            ulong polyModulusDegree,
            IEnumerable<SmallModulus> coeffModuli,
            double scale,
            string encryptedFeatureColumnName,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string weightColumnName = null,
            float? l2Const = null,
            float? l1Threshold = null,
            int? maxIterations = null)
             : base(env, labelColumnName, featureColumnName, weightColumnName, new LogLoss(), l2Const, l1Threshold, maxIterations)
        {
            _polyModulusDegree = polyModulusDegree;
            _coeffModuli = coeffModuli;
            _scale = scale;
            _encryptedFeatureColumnName = encryptedFeatureColumnName;
        }

        internal EncryptedSdcaLogisticRegressionBinaryTrainer(IHostEnvironment env, Options options)
            : base(env, options, new LogLoss())
        {
            _polyModulusDegree = options.PolyModDegree;
            _coeffModuli = options.CoeffModuli;
            _scale = options.Scale;
            _encryptedFeatureColumnName = options.EncryptedFeatureColumnName;
        }

        private protected override CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator> CreatePredictor(VBuffer<float>[] weights, float[] bias)
        {
            Host.CheckParam(Utils.Size(weights) == 1, nameof(weights));
            Host.CheckParam(Utils.Size(bias) == 1, nameof(bias));
            Host.CheckParam(weights[0].Length > 0, nameof(weights));

            VBuffer<float> maybeSparseWeights = default;
            // below should be `in weights[0]`, but can't because of https://github.com/dotnet/roslyn/issues/29371
            VBufferUtils.CreateMaybeSparseCopy(weights[0], ref maybeSparseWeights,
                Conversions.Instance.GetIsDefaultPredicate<float>(NumberDataViewType.Single));
            System.Console.WriteLine("CreatePredictor");

            var linearModel = new LinearBinaryModelParameters(Host, in maybeSparseWeights, bias[0]);
            var encryptedLinearModel = new EncryptedLinearBinaryModelParameters(Host, in maybeSparseWeights, bias[0], _polyModulusDegree, _coeffModuli, _scale);
            var calibrator = new PlattCalibrator(Host, -1, 0);
            return new EncryptedParameterMixingCalibratedModelParameters<LinearBinaryModelParameters, PlattCalibrator, EncryptedLinearBinaryModelParameters>(Host, linearModel, calibrator, encryptedLinearModel);
        }

        [BestFriend]
        private protected override SchemaShape.Column[] ComputeSdcaBinaryClassifierSchemaShape()
        {
            return new SchemaShape.Column[]
            {
                    new SchemaShape.Column(
                        DefaultColumnNames.Score,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.Single,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                    new SchemaShape.Column(
                        DefaultColumnNames.Probability,
                        SchemaShape.Column.VectorKind.Scalar,
                        NumberDataViewType.Single,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                    new SchemaShape.Column(
                        DefaultColumnNames.PredictedLabel,
                        SchemaShape.Column.VectorKind.Scalar,
                        BooleanDataViewType.Instance,
                        false,
                        new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))

            };
        }
    }
}
