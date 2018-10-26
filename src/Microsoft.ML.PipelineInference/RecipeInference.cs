// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Sweeper;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Online;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Microsoft.ML.Runtime.PipelineInference
{
    public static class RecipeInference
    {
        public struct SuggestedRecipe
        {
            public readonly string Description;
            public readonly TransformInference.SuggestedTransform[] Transforms;
            public struct SuggestedLearner
            {
                public ComponentCatalog.LoadableClassInfo LoadableClassInfo;
                public string Settings;
                public TrainerPipelineNode PipelineNode;
                public string LearnerName;

                public SuggestedLearner Clone()
                {
                    return new SuggestedLearner
                    {
                        LoadableClassInfo = LoadableClassInfo,
                        Settings = Settings,
                        PipelineNode = PipelineNode.Clone(),
                        LearnerName = LearnerName
                    };
                }

                public override string ToString() => PipelineNode.ToString();
            }

            public readonly SuggestedLearner[] Learners;
            public readonly int PreferenceIndex;

            public SuggestedRecipe(string description,
                TransformInference.SuggestedTransform[] transforms,
                SuggestedLearner[] learners,
                int preferenceIndex = -1)
            {
                Contracts.Check(transforms != null, "Transforms cannot be null");
                Contracts.Check(learners != null, "Learners cannot be null");
                Description = description;
                Transforms = transforms;
                Learners = FillLearnerNames(learners);
                PreferenceIndex = preferenceIndex;
            }

            private static SuggestedLearner[] FillLearnerNames(SuggestedLearner[] learners)
            {
                for (int i = 0; i < learners.Length; i++)
                    learners[i].LearnerName = learners[i].LoadableClassInfo.LoadNames[0];
                return learners;
            }

            public AutoInference.EntryPointGraphDef ToEntryPointGraph(IHostEnvironment env)
            {
                // All transforms must have associated PipelineNode objects
                var unsupportedTransform = Transforms.Where(transform => transform.PipelineNode == null).Cast<TransformInference.SuggestedTransform?>().FirstOrDefault();
                if (unsupportedTransform != null)
                    throw env.ExceptNotSupp($"All transforms in recipe must have entrypoint support. {unsupportedTransform} is not yet supported.");
                var subGraph = env.CreateExperiment();

                Var<IDataView> lastOutput = new Var<IDataView>();

                // Chain transforms
                var transformsModels = new List<Var<ITransformModel>>();
                foreach (var transform in Transforms)
                {
                    transform.PipelineNode.SetInputData(lastOutput);
                    var transformAddResult = transform.PipelineNode.Add(subGraph);
                    transformsModels.Add(transformAddResult.Model);
                    lastOutput = transformAddResult.OutData;
                }

                // Add learner, if present. If not, just return transforms graph object.
                if (Learners.Length > 0 && Learners[0].PipelineNode != null)
                {
                    // Add learner
                    var learner = Learners[0];
                    learner.PipelineNode.SetInputData(lastOutput);
                    var learnerAddResult = learner.PipelineNode.Add(subGraph);

                    // Create single model for featurizing and scoring data,
                    // if transforms present.
                    if (Transforms.Length > 0)
                    {
                        var modelCombine = new ML.Legacy.Transforms.ManyHeterogeneousModelCombiner
                        {
                            TransformModels = new ArrayVar<ITransformModel>(transformsModels.ToArray()),
                            PredictorModel = learnerAddResult.Model
                        };
                        var modelCombineOutput = subGraph.Add(modelCombine);

                        return new AutoInference.EntryPointGraphDef(subGraph, modelCombineOutput.PredictorModel, lastOutput);
                    }

                    // No transforms present, so just return predictor's model.
                    return new AutoInference.EntryPointGraphDef(subGraph, learnerAddResult.Model, lastOutput);
                }

                return new AutoInference.EntryPointGraphDef(subGraph, null, lastOutput);
            }

            public override string ToString() => Description;
        }

        public struct InferenceResult
        {
            public readonly SuggestedRecipe[] SuggestedRecipes;
            public InferenceResult(SuggestedRecipe[] suggestedRecipes)
            {
                SuggestedRecipes = suggestedRecipes;
            }
        }

        private static IEnumerable<Recipe> GetRecipes(IHostEnvironment env)
        {
            yield return new DefaultRecipe();
            yield return new BalancedTextClassificationRecipe(env);
            yield return new AccuracyFocusedRecipe(env);
            yield return new ExplorationComboRecipe(env);
            yield return new TreeLeafRecipe(env);
        }

        public abstract class Recipe
        {
            public virtual List<Type> AllowedTransforms() => new List<Type>()
            {
                typeof (TransformInference.Experts.AutoLabel),
                typeof (TransformInference.Experts.GroupIdHashRename),
                typeof (TransformInference.Experts.NameColumnConcatRename),
                typeof (TransformInference.Experts.LabelAdvisory),
                typeof (TransformInference.Experts.Boolean),
                typeof (TransformInference.Experts.Categorical),
                typeof (TransformInference.Experts.Text),
                typeof (TransformInference.Experts.NumericMissing),
                typeof (TransformInference.Experts.FeaturesColumnConcatRename),
            };

            public virtual List<Type> QualifierTransforms() => AllowedTransforms();

            public virtual List<Type> AllowedPredictorTypes() => MacroUtils.PredictorTypes.ToList();

            protected virtual TransformInference.SuggestedTransform[] GetSuggestedTransforms(
                TransformInference.InferenceResult transformInferenceResult, Type predictorType)
            {
                List<Type> allowedTransforms = AllowedTransforms();
                List<Type> qualifierTransforms = QualifierTransforms();

                if (AllowedPredictorTypes().Any(type => type == predictorType) &&
                    transformInferenceResult.SuggestedTransforms.Any(transform => qualifierTransforms.Contains(transform.ExpertType)))
                {
                    return transformInferenceResult.SuggestedTransforms
                        .Where(transform => allowedTransforms.Contains(transform.ExpertType) || qualifierTransforms.Contains(transform.ExpertType))
                        .ToArray();
                }

                return null;
            }

            public virtual IEnumerable<SuggestedRecipe> Apply(
                TransformInference.InferenceResult transformInferenceResult, Type predictorType, IChannel ch)
            {
                TransformInference.SuggestedTransform[] transforms = GetSuggestedTransforms(
                    transformInferenceResult, predictorType);

                if (transforms?.Length > 0)
                {
                    foreach (var recipe in ApplyCore(predictorType, transforms))
                        yield return recipe;
                }
            }

            protected abstract IEnumerable<SuggestedRecipe> ApplyCore(Type predictorType, TransformInference.SuggestedTransform[] transforms);
        }

        public sealed class DefaultRecipe : Recipe
        {
            public override List<Type> AllowedTransforms() => base.AllowedTransforms().Where(
                expert =>
                    expert != typeof(TransformInference.Experts.FeaturesColumnConcatRename))
                .Concat(new List<Type> {
                    typeof(TransformInference.Experts.FeaturesColumnConcatRenameNumericOnly)})
                .ToList();

            protected override IEnumerable<SuggestedRecipe> ApplyCore(Type predictorType,
                TransformInference.SuggestedTransform[] transforms)
            {
                yield return
                    new SuggestedRecipe(ToString(), transforms, new SuggestedRecipe.SuggestedLearner[0], int.MinValue + 1);
            }

            public override string ToString() => "Default transforms";
        }

        public abstract class MultiClassRecipies : Recipe
        {
            protected IHostEnvironment Host { get; }

            protected MultiClassRecipies(IHostEnvironment host)
            {
                Contracts.CheckValue(host, nameof(host));
                Host = host;
            }

            public override List<Type> AllowedTransforms() => base.AllowedTransforms().Where(
                expert =>
                    expert != typeof(TransformInference.Experts.Text) &&
                    expert != typeof(TransformInference.Experts.FeaturesColumnConcatRename))
                .Concat(new List<Type> {
                    typeof(TransformInference.Experts.FeaturesColumnConcatRenameNumericOnly) })
                .ToList();

            public override List<Type> AllowedPredictorTypes() => new List<Type>()
            {
                typeof (SignatureBinaryClassifierTrainer),
                typeof (SignatureMultiClassClassifierTrainer)
            };
        }

        public sealed class BalancedTextClassificationRecipe : MultiClassRecipies
        {
            public BalancedTextClassificationRecipe(IHostEnvironment host)
                : base(host)
            {
            }

            public override List<Type> QualifierTransforms()
                => new List<Type> { typeof(TransformInference.Experts.TextBiGramTriGram) };

            protected override IEnumerable<SuggestedRecipe> ApplyCore(Type predictorType,
                TransformInference.SuggestedTransform[] transforms)
            {
                SuggestedRecipe.SuggestedLearner learner = new SuggestedRecipe.SuggestedLearner();
                if (predictorType == typeof(SignatureMultiClassClassifierTrainer))
                {
                    learner.LoadableClassInfo = Host.ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>("OVA");
                    learner.Settings = "p=AveragedPerceptron{iter=10}";
                }
                else
                {
                    learner.LoadableClassInfo = Host.ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>(AveragedPerceptronTrainer.LoadNameValue);
                    learner.Settings = "iter=10";
                    var epInput = new Legacy.Trainers.AveragedPerceptronBinaryClassifier
                    {
                        NumIterations = 10
                    };
                    learner.PipelineNode = new TrainerPipelineNode(epInput);
                }

                yield return
                    new SuggestedRecipe(ToString(), transforms, new[] { learner }, int.MaxValue);
            }

            public override string ToString() => "Text classification optimized for speed and accuracy";
        }

        public sealed class AccuracyFocusedRecipe : MultiClassRecipies
        {
            public AccuracyFocusedRecipe(IHostEnvironment host)
                : base(host)
            {
            }

            public override List<Type> QualifierTransforms()
                => new List<Type> { typeof(TransformInference.Experts.TextUniGramTriGram) };

            protected override IEnumerable<SuggestedRecipe> ApplyCore(Type predictorType,
                TransformInference.SuggestedTransform[] transforms)
            {
                SuggestedRecipe.SuggestedLearner learner = new SuggestedRecipe.SuggestedLearner();
                if (predictorType == typeof(SignatureMultiClassClassifierTrainer))
                {
                    learner.LoadableClassInfo = Host.ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>("OVA");
                    learner.Settings = "p=FastTreeBinaryClassification";
                }
                else
                {
                    learner.LoadableClassInfo =
                        Host.ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>(FastTreeBinaryClassificationTrainer.LoadNameValue);
                    learner.Settings = "";
                    var epInput = new Legacy.Trainers.FastTreeBinaryClassifier();
                    learner.PipelineNode = new TrainerPipelineNode(epInput);
                }

                yield return new SuggestedRecipe(ToString(), transforms, new[] { learner });
            }

            public override string ToString() => "Text classification optimized for accuracy";
        }

        public sealed class ExplorationComboRecipe : MultiClassRecipies
        {
            public ExplorationComboRecipe(IHostEnvironment host)
                : base(host)
            {
            }

            public override List<Type> QualifierTransforms()
                => new List<Type> { typeof(TransformInference.Experts.SdcaTransform) };

            protected override IEnumerable<SuggestedRecipe> ApplyCore(Type predictorType,
                TransformInference.SuggestedTransform[] transforms)
            {
                SuggestedRecipe.SuggestedLearner learner = new SuggestedRecipe.SuggestedLearner();
                if (predictorType == typeof(SignatureMultiClassClassifierTrainer))
                {
                    learner.LoadableClassInfo =
                        Host.ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>(SdcaMultiClassTrainer.LoadNameValue);
                }
                else
                {
                    learner.LoadableClassInfo =
                        Host.ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>(LinearClassificationTrainer.LoadNameValue);
                    var epInput = new Legacy.Trainers.StochasticDualCoordinateAscentBinaryClassifier();
                    learner.PipelineNode = new TrainerPipelineNode(epInput);
                }

                learner.Settings = "";
                yield return new SuggestedRecipe(ToString(), transforms, new[] { learner });
            }

            public override string ToString() => "Text classification exploration combo";
        }

        public sealed class TreeLeafRecipe : MultiClassRecipies
        {
            public TreeLeafRecipe(IHostEnvironment host)
                : base(host)
            {
            }

            public override List<Type> QualifierTransforms()
                => new List<Type> { typeof(TransformInference.Experts.NaiveBayesTransform) };

            protected override IEnumerable<SuggestedRecipe> ApplyCore(Type predictorType,
                TransformInference.SuggestedTransform[] transforms)
            {
                SuggestedRecipe.SuggestedLearner learner = new SuggestedRecipe.SuggestedLearner();
                learner.LoadableClassInfo =
                    Host.ComponentCatalog.GetLoadableClassInfo<SignatureTrainer>(MultiClassNaiveBayesTrainer.LoadName);
                learner.Settings = "";
                var epInput = new Legacy.Trainers.NaiveBayesClassifier();
                learner.PipelineNode = new TrainerPipelineNode(epInput);
                yield return new SuggestedRecipe(ToString(), transforms, new[] { learner });
            }

            public override string ToString() => "Treeleaf multiclass";
        }

        public static SuggestedRecipe[] InferRecipesFromData(IHostEnvironment env, string dataFile, string schemaDefinitionFile,
            out Type predictorType, out string settingsString, out TransformInference.InferenceResult inferenceResult,
            bool excludeFeaturesConcatTransforms = false)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("InferRecipesFromData", seed: 0, verbose: false);

            using (var ch = h.Start("InferRecipesFromData"))
            {
                // Validate the schema file has content if provided.
                // Warn the user early if that is provided but beign skipped.
                string schemaJson = null;
                if (!string.IsNullOrEmpty(schemaDefinitionFile))
                {
                    try
                    {
                        schemaJson = File.ReadAllText(schemaDefinitionFile);
                    }
                    catch (Exception ex)
                    {
                        ch.Warning($"Unable to read the schema file. Proceeding to infer the schema :{ex.Message}");
                    }
                }

                ch.Info("Loading file sample into memory.");
                var sample = TextFileSample.CreateFromFullFile(h, dataFile);

                ch.Info("Detecting separator and columns");
                var splitResult = TextFileContents.TrySplitColumns(h, sample, TextFileContents.DefaultSeparators);

                // initialize to clustering if we're not successful?
                predictorType = typeof(SignatureClusteringTrainer);
                settingsString = "";
                if (!splitResult.IsSuccess)
                    throw ch.ExceptDecode("Couldn't detect separator.");

                ch.Info($"Separator detected as '{splitResult.Separator}', there's {splitResult.ColumnCount} columns.");

                ColumnGroupingInference.GroupingColumn[] columns;
                bool hasHeader = false;
                if (string.IsNullOrEmpty(schemaJson))
                {
                    ch.Warning("Empty schema file. Proceeding to infer the schema.");
                    columns = InferenceUtils.InferColumnPurposes(ch, h, sample, splitResult, out hasHeader);
                }
                else
                {
                    try
                    {
                        columns = JsonConvert.DeserializeObject<ColumnGroupingInference.GroupingColumn[]>(schemaJson);
                        ch.Info("Using the provided schema file.");
                    }
                    catch
                    {
                        ch.Warning("Invalid json in the schema file. Proceeding to infer the schema.");
                        columns = InferenceUtils.InferColumnPurposes(ch, h, sample, splitResult, out hasHeader);
                    }
                }

                var finalLoaderArgs = new TextLoader.Arguments
                {
                    Column = ColumnGroupingInference.GenerateLoaderColumns(columns),
                    HasHeader = hasHeader,
                    Separator = splitResult.Separator,
                    AllowSparse = splitResult.AllowSparse,
                    AllowQuoting = splitResult.AllowQuote
                };

                settingsString = CommandLine.CmdParser.GetSettings(h, finalLoaderArgs, new TextLoader.Arguments());
                ch.Info($"Loader options: {settingsString}");

                ch.Info("Inferring recipes");
                var finalData = TextLoader.ReadFile(h, finalLoaderArgs, sample);
                var cached = new CacheDataView(h, finalData,
                    Enumerable.Range(0, finalLoaderArgs.Column.Length).ToArray());

                var purposeColumns = columns.Select((x, i) => new PurposeInference.Column(i, x.Purpose, x.ItemKind)).ToArray();

                var fraction = sample.FullFileSize == null ? 1.0 : (double)sample.SampleSize / sample.FullFileSize.Value;
                var transformInferenceResult = TransformInference.InferTransforms(h, cached, purposeColumns,
                    new TransformInference.Arguments
                    {
                        EstimatedSampleFraction = fraction,
                        ExcludeFeaturesConcatTransforms = excludeFeaturesConcatTransforms
                    }
                );
                predictorType = InferenceUtils.InferPredictorCategoryType(cached, purposeColumns);
                var recipeInferenceResult = InferRecipes(h, transformInferenceResult, predictorType);

                inferenceResult = transformInferenceResult;
                return recipeInferenceResult.SuggestedRecipes;
            }
        }

        public static InferenceResult InferRecipes(IHostEnvironment env, TransformInference.InferenceResult transformInferenceResult,
            Type predictorType)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register("InferRecipes");

            using (var ch = h.Start("InferRecipes"))
            {
                var list = new List<SuggestedRecipe>();
                foreach (var recipe in GetRecipes(h))
                    list.AddRange(recipe.Apply(transformInferenceResult, predictorType, ch));

                if (list.Count == 0)
                    ch.Info("No recipes are needed for the data.");

                return new InferenceResult(list.ToArray());
            }
        }

        public static List<string> GetLearnerSettingsAndSweepParams(IHostEnvironment env, ComponentCatalog.LoadableClassInfo cl, out string settings)
        {
            List<string> sweepParams = new List<string>();
            var ci = cl.Constructor?.GetParameters();
            if (ci == null)
            {
                settings = "";
                return sweepParams;
            }

            var suggestedSweepsParser = new SuggestedSweepsParser();
            StringBuilder learnerSettings = new StringBuilder();

            foreach (var prop in ci)
            {
                var fieldInfo = prop.ParameterType?.GetFields(BindingFlags.Public | BindingFlags.Instance);

                foreach (var field in fieldInfo)
                {
                    TGUIAttribute[] tgui =
                        field.GetCustomAttributes(typeof(TGUIAttribute), true) as TGUIAttribute[];
                    if (tgui == null)
                        continue;
                    foreach (var attr in tgui)
                    {
                        if (attr.SuggestedSweeps != null)
                        {
                            // Build the learner setting.
                            learnerSettings.Append($" {field.Name}=${field.Name}$");

                            // Build the sweeper.
                            suggestedSweepsParser.TryParseParameter(attr.SuggestedSweeps, field.FieldType, field.Name, out var sweepValues, out var error);
                            sweepParams.Add(sweepValues?.ToStringParameter(env));
                        }
                    }
                }
            }
            settings = learnerSettings.ToString();
            return sweepParams;
        }

        /// <summary>
        /// Given a predictor type returns a set of all permissible learners (with their sweeper params, if defined).
        /// </summary>
        /// <returns>Array of viable learners.</returns>
        public static SuggestedRecipe.SuggestedLearner[] AllowedLearners(IHostEnvironment env, MacroUtils.TrainerKinds trainerKind)
        {
            //not all learners advertised in the API are available in CORE.
            var catalog = env.ComponentCatalog;
            var availableLearnersList = catalog.AllEntryPoints().Where(
                x => x.InputKinds?.FirstOrDefault(i => i == typeof(CommonInputs.ITrainerInput)) != null);

            var learners = new List<SuggestedRecipe.SuggestedLearner>();
            var type = typeof(CommonInputs.ITrainerInput);
            var trainerTypes = typeof(Experiment).Assembly.GetTypes()
                .Where(p => type.IsAssignableFrom(p) &&
                    MacroUtils.IsTrainerOfKind(p, trainerKind));

            foreach (var tt in trainerTypes)
            {
                var sweepParams = AutoMlUtils.GetSweepRanges(tt);
                var epInputObj = (CommonInputs.ITrainerInput)tt.GetConstructor(Type.EmptyTypes)?.Invoke(new object[] { });
                var sl = new SuggestedRecipe.SuggestedLearner
                {
                    PipelineNode = new TrainerPipelineNode(epInputObj, sweepParams),
                    LearnerName = tt.Name
                };

                if (sl.PipelineNode != null && availableLearnersList.FirstOrDefault(l => l.Name.Equals(sl.PipelineNode.GetEpName())) != null)
                    learners.Add(sl);
            }

            return learners.ToArray();
        }
    }
}
