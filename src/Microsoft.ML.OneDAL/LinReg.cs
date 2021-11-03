using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Security;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

[assembly: LoadableClass(LinRegTrainer.Summary, typeof(LinRegTrainer), typeof(LinRegTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureFeatureScorerTrainer) },
    LinRegTrainer.UserNameValue,
    LinRegTrainer.LoadNameValue,
    LinRegTrainer.ShortName)]
[assembly: LoadableClass(typeof(LinearRegressionModelParameters), null, typeof(SignatureLoadModel),
    "oneDAL Linear Regression Executor",
    LinearRegressionModelParameters.LoaderSignature)]
[assembly: LoadableClass(typeof(void), typeof(LinRegTrainer), null, typeof(SignatureEntryPointModule), LinRegTrainer.LoadNameValue)]
namespace Microsoft.ML.Trainers
{
    public sealed class LinRegTrainer : TrainerEstimatorBase<RegressionPredictionTransformer<LinearRegressionModelParameters>, LinearRegressionModelParameters>
    {
        public sealed class Options : TrainerInputBaseWithWeight
        {
        }

        internal const string LoadNameValue = "LinearRegression";
        internal const string UserNameValue = "Linear Regression";
        internal const string ShortName = "linreg";
        internal const string Summary = "oneDAL Linear Regression";

	private protected override PredictionKind PredictionKind => PredictionKind.Regression;

	private static readonly TrainerInfo _info = new TrainerInfo(caching: false);

	public override TrainerInfo Info => _info;

	internal LinRegTrainer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(LoadNameValue), TrainerUtils.MakeR4VecFeature(options.FeatureColumnName),
                  TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName), TrainerUtils.MakeR4ScalarWeightColumn(options.ExampleWeightColumnName))
        {
            Host.CheckValue(options, nameof(options));
        }

	private protected override RegressionPredictionTransformer<LinearRegressionModelParameters> MakeTransformer(LinearRegressionModelParameters model, DataViewSchema trainSchema)
             => new RegressionPredictionTransformer<LinearRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

	private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
	}

        private protected override LinearRegressionModelParameters TrainModelCore(TrainContext context)
        {
            using (var ch = Host.Start("Training"))
            {
                ch.CheckValue(context, nameof(context));
                var examples = context.TrainingSet;
                ch.CheckParam(examples.Schema.Feature.HasValue, nameof(examples), "Need a feature column");
                ch.CheckParam(examples.Schema.Label.HasValue, nameof(examples), "Need a labelColumn column");
 
                // The labelColumn type must be either Float or a key type based on int (if allowKeyLabels is true).
                var typeLab = examples.Schema.Label.Value.Type;
                if (typeLab != NumberDataViewType.Single)
                    throw ch.Except("Incompatible labelColumn column type {0}, must be {1}", typeLab, NumberDataViewType.Single);
 
                // The feature type must be a vector of Float.
                var typeFeat = examples.Schema.Feature.Value.Type as VectorDataViewType;
                if (typeFeat == null || !typeFeat.IsKnownSize)
                    throw ch.Except("Incompatible feature column type {0}, must be known sized vector of {1}", typeFeat, NumberDataViewType.Single);
                if (typeFeat.ItemType != NumberDataViewType.Single)
                    throw ch.Except("Incompatible feature column type {0}, must be vector of {1}", typeFeat, NumberDataViewType.Single);
 
                CursOpt cursorOpt = CursOpt.Label | CursOpt.Features;
                if (examples.Schema.Weight.HasValue)
                    cursorOpt |= CursOpt.Weight;
 
                var cursorFactory = new FloatLabelCursor.Factory(examples, cursorOpt);
 
                return TrainCore(ch, cursorFactory, typeFeat.Size);
            }
	}

	private LinearRegressionModelParameters TrainCore(IChannel ch, FloatLabelCursor.Factory cursorFactory, int featureCount)
        {
            Host.AssertValue(ch);
            ch.AssertValue(cursorFactory);
 
            int m = featureCount + 1;
            var betasArray = new float[m];
            List<float> labelsList = new List<float>();
            List<float> featuresList = new List<float>();
 
            int n = 0;
            using (var cursor = cursorFactory.Create())
            {
                while (cursor.MoveNext())
                {
                    labelsList.Add(cursor.Label);
                    var values = cursor.Features.GetValues();
 
                    ch.Assert(cursor.Features.IsDense);
                    ch.Assert(values.Length + 1 == m);
 
                    for (int j = 0; j < m - 1; ++j)
                    {
                        featuresList.Add(values[j]);
                    }
                    n++;
                }
                if (cursor.BadFeaturesRowCount > 0)
                    ch.Warning("Skipped {0} instances with missing features/labelColumn during training", cursor.SkippedRowCount);
 
            }
 
            float[] labels = labelsList.ToArray();
            float[] features = featuresList.ToArray();
 
            // oneDAL part
            unsafe
            {
                fixed (void* featuresPtr = &features[0], labelsPtr = &labels[0], betasPtr = &betasArray[0])
                {
                    OneDAL.LinearRegressionSingle(featuresPtr, labelsPtr, betasPtr, n, m - 1);
                }
            }
 
            float bias = betasArray[0];
            var weights = new float[m - 1];
            for (int i = 1; i < m; ++i)
            {
                weights[i - 1] = betasArray[i];
            }
 
            var weightsBuffer = new VBuffer<float>(m - 1, weights);
	     
            return new LinearRegressionModelParameters(Host, in weightsBuffer, bias);
        }

	internal static class OneDAL
        {
            [DllImport("OneDALNative", EntryPoint = "linearRegressionSingle")]
            public unsafe static extern void LinearRegressionSingle(void* features, void* labels, void* betas, int nRows, int nColumns);
        }
    }
}

 