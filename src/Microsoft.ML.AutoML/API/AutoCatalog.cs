// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// A catalog of all available AutoML tasks.
    /// </summary>
    public sealed class AutoCatalog
    {
        private readonly MLContext _context;

        internal AutoCatalog(MLContext context)
        {
            _context = context;
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a regression dataset.
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">Maximum number of seconds that experiment will run.</param>
        /// <returns>A new AutoML regression experiment.</returns>
        /// <remarks>
        /// <para>See <see cref="RegressionExperiment"/> for a more detailed code example of an AutoML regression experiment.</para>
        /// <para>An experiment may run for longer than <paramref name="maxExperimentTimeInSeconds"/>.
        /// This is because once AutoML starts training an ML.NET model, AutoML lets the
        /// model train to completion. For instance, if the first model
        /// AutoML trains takes 4 hours, and the second model trained takes 5 hours,
        /// but <paramref name="maxExperimentTimeInSeconds"/> was the number of seconds in 6 hours,
        /// the experiment will run for 4 + 5 = 9 hours (not 6 hours).</para>
        /// </remarks>
        public RegressionExperiment CreateRegressionExperiment(uint maxExperimentTimeInSeconds)
        {
            return new RegressionExperiment(_context, new RegressionExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a regression dataset.
        /// </summary>
        /// <param name="experimentSettings">Settings for the AutoML experiment.</param>
        /// <returns>A new AutoML regression experiment.</returns>
        /// <remarks>
        /// See <see cref="RegressionExperiment"/> for a more detailed code example of an AutoML regression experiment.
        /// </remarks>
        public RegressionExperiment CreateRegressionExperiment(RegressionExperimentSettings experimentSettings)
        {
            return new RegressionExperiment(_context, experimentSettings);
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a binary classification dataset.
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">Maximum number of seconds that experiment will run.</param>
        /// <returns>A new AutoML binary classification experiment.</returns>
        /// <remarks>
        /// <para>See <see cref="BinaryClassificationExperiment"/> for a more detailed code example of an AutoML binary classification experiment.</para>
        /// <para>An experiment may run for longer than <paramref name="maxExperimentTimeInSeconds"/>.
        /// This is because once AutoML starts training an ML.NET model, AutoML lets the
        /// model train to completion. For instance, if the first model
        /// AutoML trains takes 4 hours, and the second model trained takes 5 hours,
        /// but <paramref name="maxExperimentTimeInSeconds"/> was the number of seconds in 6 hours,
        /// the experiment will run for 4 + 5 = 9 hours (not 6 hours).</para>
        /// </remarks>
        public BinaryClassificationExperiment CreateBinaryClassificationExperiment(uint maxExperimentTimeInSeconds)
        {
            return new BinaryClassificationExperiment(_context, new BinaryExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a binary classification dataset.
        /// </summary>
        /// <param name="experimentSettings">Settings for the AutoML experiment.</param>
        /// <returns>A new AutoML binary classification experiment.</returns>
        /// <remarks>
        /// See <see cref="BinaryClassificationExperiment"/> for a more detailed code example of an AutoML binary classification experiment.
        /// </remarks>
        public BinaryClassificationExperiment CreateBinaryClassificationExperiment(BinaryExperimentSettings experimentSettings)
        {
            return new BinaryClassificationExperiment(_context, experimentSettings);
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a multiclass classification dataset.
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">Maximum number of seconds that experiment will run.</param>
        /// <returns>A new AutoML multiclass classification experiment.</returns>
        /// <remarks>
        /// <para>See <see cref="MulticlassClassificationExperiment"/> for a more detailed code example of an AutoML multiclass classification experiment.</para>
        /// <para>An experiment may run for longer than <paramref name="maxExperimentTimeInSeconds"/>.
        /// This is because once AutoML starts training an ML.NET model, AutoML lets the
        /// model train to completion. For instance, if the first model
        /// AutoML trains takes 4 hours, and the second model trained takes 5 hours,
        /// but <paramref name="maxExperimentTimeInSeconds"/> was the number of seconds in 6 hours,
        /// the experiment will run for 4 + 5 = 9 hours (not 6 hours).</para>
        /// </remarks>
        public MulticlassClassificationExperiment CreateMulticlassClassificationExperiment(uint maxExperimentTimeInSeconds)
        {
            return new MulticlassClassificationExperiment(_context, new MulticlassExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a multiclass classification dataset.
        /// </summary>
        /// <param name="experimentSettings">Settings for the AutoML experiment.</param>
        /// <returns>A new AutoML multiclass classification experiment.</returns>
        /// <remarks>
        /// See <see cref="MulticlassClassificationExperiment"/> for a more detailed code example of an AutoML multiclass classification experiment.
        /// </remarks>
        public MulticlassClassificationExperiment CreateMulticlassClassificationExperiment(MulticlassExperimentSettings experimentSettings)
        {
            return new MulticlassClassificationExperiment(_context, experimentSettings);
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a recommendation classification dataset.
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">Maximum number of seconds that experiment will run.</param>
        /// <returns>A new AutoML recommendation classification experiment.</returns>
        /// <remarks>
        /// <para>See <see cref="RecommendationExperiment"/> for a more detailed code example of an AutoML multiclass classification experiment.</para>
        /// <para>An experiment may run for longer than <paramref name="maxExperimentTimeInSeconds"/>.
        /// This is because once AutoML starts training an ML.NET model, AutoML lets the
        /// model train to completion. For instance, if the first model
        /// AutoML trains takes 4 hours, and the second model trained takes 5 hours,
        /// but <paramref name="maxExperimentTimeInSeconds"/> was the number of seconds in 6 hours,
        /// the experiment will run for 4 + 5 = 9 hours (not 6 hours).</para>
        /// </remarks>
        public RecommendationExperiment CreateRecommendationExperiment(uint maxExperimentTimeInSeconds)
        {
            return new RecommendationExperiment(_context, new RecommendationExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a recommendation dataset.
        /// </summary>
        /// <param name="experimentSettings">Settings for the AutoML experiment.</param>
        /// <returns>A new AutoML recommendation experiment.</returns>
        /// <remarks>
        /// See <see cref="RecommendationExperiment"/> for a more detailed code example of an AutoML recommendation experiment.
        /// </remarks>
        public RecommendationExperiment CreateRecommendationExperiment(RecommendationExperimentSettings experimentSettings)
        {
            return new RecommendationExperiment(_context, experimentSettings);
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a ranking dataset.
        /// </summary>
        /// <param name="maxExperimentTimeInSeconds">Maximum number of seconds that experiment will run.</param>
        /// <returns>A new AutoML ranking experiment.</returns>
        /// <remarks>
        /// <para>See <see cref="RankingExperiment"/> for a more detailed code example of an AutoML ranking experiment.</para>
        /// <para>An experiment may run for longer than <paramref name="maxExperimentTimeInSeconds"/>.
        /// This is because once AutoML starts training an ML.NET model, AutoML lets the
        /// model train to completion. For instance, if the first model
        /// AutoML trains takes 4 hours, and the second model trained takes 5 hours,
        /// but <paramref name="maxExperimentTimeInSeconds"/> was the number of seconds in 6 hours,
        /// the experiment will run for 4 + 5 = 9 hours (not 6 hours).</para>
        /// </remarks>
        public RankingExperiment CreateRankingExperiment(uint maxExperimentTimeInSeconds)
        {
            return new RankingExperiment(_context, new RankingExperimentSettings()
            {
                MaxExperimentTimeInSeconds = maxExperimentTimeInSeconds
            });
        }

        /// <summary>
        /// Creates a new AutoML experiment to run on a ranking dataset.
        /// </summary>
        /// <param name="experimentSettings">Settings for the AutoML experiment.</param>
        /// <returns>A new AutoML ranking experiment.</returns>
        /// <remarks>
        /// See <see cref="RankingExperiment"/> for a more detailed code example of an AutoML ranking experiment.
        /// </remarks>
        public RankingExperiment CreateRankingExperiment(RankingExperimentSettings experimentSettings)
        {
            return new RankingExperiment(_context, experimentSettings);
        }

        /// <summary>
        /// Infers information about the columns of a dataset in a file located at <paramref name="path"/>.
        /// </summary>
        /// <param name="path">Path to a dataset file.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="separatorChar">The character used as separator between data elements in a row. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="allowQuoting">Whether the file can contain columns defined by a quoted string. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="allowSparse">Whether the file can contain numerical vectors in sparse format. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="trimWhitespace">Whether trailing whitespace should be removed from dataset file lines.</param>
        /// <param name="groupColumns">Whether to group together (when possible) original columns in the dataset file into vector columns in the resulting data structures. See <see cref="TextLoader.Range"/> for more information.</param>
        /// <returns>Information inferred about the columns in the provided dataset.</returns>
        /// <remarks>
        /// Infers information about the name, data type, and purpose of each column.
        /// The returned <see cref="ColumnInferenceResults.TextLoaderOptions" /> can be used to
        /// instantiate a <see cref="TextLoader" />. The <see cref="TextLoader" /> can be used to
        /// obtain an <see cref="IDataView"/> that can be fed into an AutoML experiment,
        /// or used elsewhere in the ML.NET ecosystem (ie in <see cref="IEstimator{TTransformer}.Fit(IDataView)"/>.
        /// The <see cref="ColumnInformation"/> contains the inferred purpose of each column in the dataset.
        /// (For instance, is the column categorical, numeric, or text data? Should the column be ignored? Etc.)
        /// The <see cref="ColumnInformation"/> can be inspected and modified (or kept as is) and used by an AutoML experiment.
        /// </remarks>
        public ColumnInferenceResults InferColumns(string path, string labelColumnName = DefaultColumnNames.Label, char? separatorChar = null, bool? allowQuoting = null,
            bool? allowSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path, labelColumnName);
            return ColumnInferenceApi.InferColumns(_context, path, labelColumnName, separatorChar, allowQuoting, allowSparse, trimWhitespace, groupColumns);
        }

        /// <summary>
        /// Infers information about the columns of a dataset in a file located at <paramref name="path"/>.
        /// </summary>
        /// <param name="path">Path to a dataset file.</param>
        /// <param name="columnInformation">Column information for the dataset.</param>
        /// <param name="separatorChar">The character used as separator between data elements in a row. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="allowQuoting">Whether the file can contain columns defined by a quoted string. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="allowSparse">Whether the file can contain numerical vectors in sparse format. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="trimWhitespace">Whether trailing whitespace should be removed from dataset file lines.</param>
        /// <param name="groupColumns">Whether to group together (when possible) original columns in the dataset file into vector columns in the resulting data structures. See <see cref="TextLoader.Range"/> for more information.</param>
        /// <returns>Information inferred about the columns in the provided dataset.</returns>
        /// <remarks>
        /// Infers information about the name, data type, and purpose of each column.
        /// The returned <see cref="ColumnInferenceResults.TextLoaderOptions" /> can be used to
        /// instantiate a <see cref="TextLoader" />. The <see cref="TextLoader" /> can be used to
        /// obtain an <see cref="IDataView"/> that can be fed into an AutoML experiment,
        /// or used elsewhere in the ML.NET ecosystem (ie in <see cref="IEstimator{TTransformer}.Fit(IDataView)"/>.
        /// The <see cref="ColumnInformation"/> contains the inferred purpose of each column in the dataset.
        /// (For instance, is the column categorical, numeric, or text data? Should the column be ignored? Etc.)
        /// The <see cref="ColumnInformation"/> can be inspected and modified (or kept as is) and used by an AutoML experiment.
        /// </remarks>
        public ColumnInferenceResults InferColumns(string path, ColumnInformation columnInformation, char? separatorChar = null, bool? allowQuoting = null,
            bool? allowSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            columnInformation = columnInformation ?? new ColumnInformation();
            UserInputValidationUtil.ValidateInferColumnsArgs(path, columnInformation);
            return ColumnInferenceApi.InferColumns(_context, path, columnInformation, separatorChar, allowQuoting, allowSparse, trimWhitespace, groupColumns);
        }

        /// <summary>
        /// Infers information about the columns of a dataset in a file located at <paramref name="path"/>.
        /// </summary>
        /// <param name="path">Path to a dataset file.</param>
        /// <param name="labelColumnIndex">Column index of the label column in the dataset.</param>
        /// <param name="hasHeader">Whether or not the dataset file has a header row.</param>
        /// <param name="separatorChar">The character used as separator between data elements in a row. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="allowQuoting">Whether the file can contain columns defined by a quoted string. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="allowSparse">Whether the file can contain numerical vectors in sparse format. If <see langword="null"/>, AutoML will try to infer this value.</param>
        /// <param name="trimWhitespace">Whether trailing whitespace should be removed from dataset file lines.</param>
        /// <param name="groupColumns">Whether to group together (when possible) original columns in the dataset file into vector columns in the resulting data structures. See <see cref="TextLoader.Range"/> for more information.</param>
        /// <returns>Information inferred about the columns in the provided dataset.</returns>
        /// <remarks>
        /// Infers information about the name, data type, and purpose of each column.
        /// The returned <see cref="ColumnInferenceResults.TextLoaderOptions" /> can be used to
        /// instantiate a <see cref="TextLoader" />. The <see cref="TextLoader" /> can be used to
        /// obtain an <see cref="IDataView"/> that can be fed into an AutoML experiment,
        /// or used elsewhere in the ML.NET ecosystem (ie in <see cref="IEstimator{TTransformer}.Fit(IDataView)"/>.
        /// The <see cref="ColumnInformation"/> contains the inferred purpose of each column in the dataset.
        /// (For instance, is the column categorical, numeric, or text data? Should the column be ignored? Etc.)
        /// The <see cref="ColumnInformation"/> can be inspected and modified (or kept as is) and used by an AutoML experiment.
        /// </remarks>
        public ColumnInferenceResults InferColumns(string path, uint labelColumnIndex, bool hasHeader = false, char? separatorChar = null,
            bool? allowQuoting = null, bool? allowSparse = null, bool trimWhitespace = false, bool groupColumns = true)
        {
            UserInputValidationUtil.ValidateInferColumnsArgs(path);
            return ColumnInferenceApi.InferColumns(_context, path, labelColumnIndex, hasHeader, separatorChar, allowQuoting, allowSparse, trimWhitespace, groupColumns);
        }

        /// <summary>
        /// Create a sweepable estimator with a custom factory and search space.
        /// </summary>
        public SweepableEstimator CreateSweepableEstimator<T>(Func<MLContext, T, IEstimator<ITransformer>> factory, SearchSpace<T> ss = null)
            where T : class, new()
        {
            return new SweepableEstimator((MLContext context, Parameter param) => factory(context, param.AsType<T>()), ss);
        }

        /// <summary>
        /// Create an <see cref="AutoMLExperiment"/>.
        /// </summary>
        public AutoMLExperiment CreateExperiment(AutoMLExperiment.AutoMLExperimentSettings settings = null)
        {
            return new AutoMLExperiment(_context, settings ?? new AutoMLExperiment.AutoMLExperimentSettings());
        }

        /// <summary>
        /// Create a list of <see cref="SweepableEstimator"/> for binary classification.
        /// </summary>
        /// <param name="labelColumnName">label column name.</param>
        /// <param name="featureColumnName">feature column name.</param>
        /// <param name="exampleWeightColumnName">example weight column name.</param>
        /// <param name="useFastForest">true if use fast forest as available trainer.</param>
        /// <param name="useLgbm">true if use lgbm as available trainer.</param>
        /// <param name="useFastTree">true if use fast tree as available trainer.</param>
        /// <param name="useLbfgs">true if use lbfgs as available trainer.</param>
        /// <param name="useSdca">true if use sdca as available trainer.</param>
        /// <param name="fastTreeOption">if provided, use it as initial option for fast tree, otherwise the default option will be used.</param>
        /// <param name="lgbmOption">if provided, use it as initial option for lgbm, otherwise the default option will be used.</param>
        /// <param name="fastForestOption">if provided, use it as initial option for fast forest, otherwise the default option will be used.</param>
        /// <param name="lbfgsOption">if provided, use it as initial option for lbfgs, otherwise the default option will be used.</param>
        /// <param name="sdcaOption">if provided, use it as initial option for sdca, otherwise the default option will be used.</param>
        /// <param name="fastTreeSearchSpace">if provided, use it as search space for fast tree, otherwise the default search space will be used.</param>
        /// <param name="lgbmSearchSpace">if provided, use it as search space for lgbm, otherwise the default search space will be used.</param>
        /// <param name="fastForestSearchSpace">if provided, use it as search space for fast forest, otherwise the default search space will be used.</param>
        /// <param name="lbfgsSearchSpace">if provided, use it as search space for lbfgs, otherwise the default search space will be used.</param>
        /// <param name="sdcaSearchSpace">if provided, use it as search space for sdca, otherwise the default search space will be used.</param>
        /// <returns></returns>
        public SweepableEstimator[] BinaryClassification(string labelColumnName = DefaultColumnNames.Label, string featureColumnName = DefaultColumnNames.Features, string exampleWeightColumnName = null, bool useFastForest = true, bool useLgbm = true, bool useFastTree = true, bool useLbfgs = true, bool useSdca = true,
            FastTreeOption fastTreeOption = null, LgbmOption lgbmOption = null, FastForestOption fastForestOption = null, LbfgsOption lbfgsOption = null, SdcaOption sdcaOption = null,
            SearchSpace<FastTreeOption> fastTreeSearchSpace = null, SearchSpace<LgbmOption> lgbmSearchSpace = null, SearchSpace<FastForestOption> fastForestSearchSpace = null, SearchSpace<LbfgsOption> lbfgsSearchSpace = null, SearchSpace<SdcaOption> sdcaSearchSpace = null)
        {
            var res = new List<SweepableEstimator>();

            if (useFastTree)
            {
                fastTreeOption = fastTreeOption ?? new FastTreeOption();
                fastTreeOption.LabelColumnName = labelColumnName;
                fastTreeOption.FeatureColumnName = featureColumnName;
                fastTreeOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateFastTreeBinary(fastTreeOption, fastTreeSearchSpace ?? new SearchSpace<FastTreeOption>(fastTreeOption)));
            }

            if (useFastForest)
            {
                fastForestOption = fastForestOption ?? new FastForestOption();
                fastForestOption.LabelColumnName = labelColumnName;
                fastForestOption.FeatureColumnName = featureColumnName;
                fastForestOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateFastForestBinary(fastForestOption, fastForestSearchSpace ?? new SearchSpace<FastForestOption>(fastForestOption)));
            }

            if (useLgbm)
            {
                lgbmOption = lgbmOption ?? new LgbmOption();
                lgbmOption.LabelColumnName = labelColumnName;
                lgbmOption.FeatureColumnName = featureColumnName;
                lgbmOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateLightGbmBinary(lgbmOption, lgbmSearchSpace ?? new SearchSpace<LgbmOption>(lgbmOption)));
            }

            if (useLbfgs)
            {
                lbfgsOption = lbfgsOption ?? new LbfgsOption();
                lbfgsOption.LabelColumnName = labelColumnName;
                lbfgsOption.FeatureColumnName = featureColumnName;
                lbfgsOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateLbfgsLogisticRegressionBinary(lbfgsOption, lbfgsSearchSpace ?? new SearchSpace<LbfgsOption>(lbfgsOption)));
            }

            if (useSdca)
            {
                sdcaOption = sdcaOption ?? new SdcaOption();
                sdcaOption.LabelColumnName = labelColumnName;
                sdcaOption.FeatureColumnName = featureColumnName;
                sdcaOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateSdcaLogisticRegressionBinary(sdcaOption, sdcaSearchSpace ?? new SearchSpace<SdcaOption>(sdcaOption)));
            }

            return res.ToArray();
        }

        /// <summary>
        /// Create a list of <see cref="SweepableEstimator"/> for multiclass classification.
        /// </summary>
        /// <param name="labelColumnName">label column name.</param>
        /// <param name="featureColumnName">feature column name.</param>
        /// <param name="exampleWeightColumnName">example weight column name.</param>
        /// <param name="useFastForest">true if use fast forest as available trainer.</param>
        /// <param name="useLgbm">true if use lgbm as available trainer.</param>
        /// <param name="useFastTree">true if use fast tree as available trainer.</param>
        /// <param name="useLbfgs">true if use lbfgs as available trainer.</param>
        /// <param name="useSdca">true if use sdca as available trainer.</param>
        /// <param name="fastTreeOption">if provided, use it as initial option for fast tree, otherwise the default option will be used.</param>
        /// <param name="lgbmOption">if provided, use it as initial option for lgbm, otherwise the default option will be used.</param>
        /// <param name="fastForestOption">if provided, use it as initial option for fast forest, otherwise the default option will be used.</param>
        /// <param name="lbfgsOption">if provided, use it as initial option for lbfgs, otherwise the default option will be used.</param>
        /// <param name="sdcaOption">if provided, use it as initial option for sdca, otherwise the default option will be used.</param>
        /// <param name="fastTreeSearchSpace">if provided, use it as search space for fast tree, otherwise the default search space will be used.</param>
        /// <param name="lgbmSearchSpace">if provided, use it as search space for lgbm, otherwise the default search space will be used.</param>
        /// <param name="fastForestSearchSpace">if provided, use it as search space for fast forest, otherwise the default search space will be used.</param>
        /// <param name="lbfgsSearchSpace">if provided, use it as search space for lbfgs, otherwise the default search space will be used.</param>
        /// <param name="sdcaSearchSpace">if provided, use it as search space for sdca, otherwise the default search space will be used.</param>
        /// <returns></returns>
        public SweepableEstimator[] MultiClassification(string labelColumnName = DefaultColumnNames.Label, string featureColumnName = DefaultColumnNames.Features, string exampleWeightColumnName = null, bool useFastForest = true, bool useLgbm = true, bool useFastTree = true, bool useLbfgs = true, bool useSdca = true,
            FastTreeOption fastTreeOption = null, LgbmOption lgbmOption = null, FastForestOption fastForestOption = null, LbfgsOption lbfgsOption = null, SdcaOption sdcaOption = null,
            SearchSpace<FastTreeOption> fastTreeSearchSpace = null, SearchSpace<LgbmOption> lgbmSearchSpace = null, SearchSpace<FastForestOption> fastForestSearchSpace = null, SearchSpace<LbfgsOption> lbfgsSearchSpace = null, SearchSpace<SdcaOption> sdcaSearchSpace = null)
        {
            var res = new List<SweepableEstimator>();

            if (useFastTree)
            {
                fastTreeOption = fastTreeOption ?? new FastTreeOption();
                fastTreeOption.LabelColumnName = labelColumnName;
                fastTreeOption.FeatureColumnName = featureColumnName;
                fastTreeOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateFastTreeOva(fastTreeOption, fastTreeSearchSpace ?? new SearchSpace<FastTreeOption>(fastTreeOption)));
            }

            if (useFastForest)
            {
                fastForestOption = fastForestOption ?? new FastForestOption();
                fastForestOption.LabelColumnName = labelColumnName;
                fastForestOption.FeatureColumnName = featureColumnName;
                fastForestOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateFastForestOva(fastForestOption, fastForestSearchSpace ?? new SearchSpace<FastForestOption>(fastForestOption)));
            }

            if (useLgbm)
            {
                lgbmOption = lgbmOption ?? new LgbmOption();
                lgbmOption.LabelColumnName = labelColumnName;
                lgbmOption.FeatureColumnName = featureColumnName;
                lgbmOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateLightGbmMulti(lgbmOption, lgbmSearchSpace ?? new SearchSpace<LgbmOption>(lgbmOption)));
            }

            if (useLbfgs)
            {
                lbfgsOption = lbfgsOption ?? new LbfgsOption();
                lbfgsOption.LabelColumnName = labelColumnName;
                lbfgsOption.FeatureColumnName = featureColumnName;
                lbfgsOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateLbfgsLogisticRegressionOva(lbfgsOption, lbfgsSearchSpace ?? new SearchSpace<LbfgsOption>(lbfgsOption)));
                res.Add(SweepableEstimatorFactory.CreateLbfgsMaximumEntropyMulti(lbfgsOption, lbfgsSearchSpace ?? new SearchSpace<LbfgsOption>(lbfgsOption)));
            }

            if (useSdca)
            {
                sdcaOption = sdcaOption ?? new SdcaOption();
                sdcaOption.LabelColumnName = labelColumnName;
                sdcaOption.FeatureColumnName = featureColumnName;
                sdcaOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateSdcaMaximumEntropyMulti(sdcaOption, sdcaSearchSpace ?? new SearchSpace<SdcaOption>(sdcaOption)));
                res.Add(SweepableEstimatorFactory.CreateSdcaLogisticRegressionOva(sdcaOption, sdcaSearchSpace ?? new SearchSpace<SdcaOption>(sdcaOption)));
            }

            return res.ToArray();
        }

        /// <summary>
        /// Create a list of <see cref="SweepableEstimator"/> for regression.
        /// </summary>
        /// <param name="labelColumnName">label column name.</param>
        /// <param name="featureColumnName">feature column name.</param>
        /// <param name="exampleWeightColumnName">example weight column name.</param>
        /// <param name="useFastForest">true if use fast forest as available trainer.</param>
        /// <param name="useLgbm">true if use lgbm as available trainer.</param>
        /// <param name="useFastTree">true if use fast tree as available trainer.</param>
        /// <param name="useLbfgs">true if use lbfgs as available trainer.</param>
        /// <param name="useSdca">true if use sdca as available trainer.</param>
        /// <param name="fastTreeOption">if provided, use it as initial option for fast tree, otherwise the default option will be used.</param>
        /// <param name="lgbmOption">if provided, use it as initial option for lgbm, otherwise the default option will be used.</param>
        /// <param name="fastForestOption">if provided, use it as initial option for fast forest, otherwise the default option will be used.</param>
        /// <param name="lbfgsOption">if provided, use it as initial option for lbfgs, otherwise the default option will be used.</param>
        /// <param name="sdcaOption">if provided, use it as initial option for sdca, otherwise the default option will be used.</param>
        /// <param name="fastTreeSearchSpace">if provided, use it as search space for fast tree, otherwise the default search space will be used.</param>
        /// <param name="lgbmSearchSpace">if provided, use it as search space for lgbm, otherwise the default search space will be used.</param>
        /// <param name="fastForestSearchSpace">if provided, use it as search space for fast forest, otherwise the default search space will be used.</param>
        /// <param name="lbfgsSearchSpace">if provided, use it as search space for lbfgs, otherwise the default search space will be used.</param>
        /// <param name="sdcaSearchSpace">if provided, use it as search space for sdca, otherwise the default search space will be used.</param>
        /// <returns></returns>
        public SweepableEstimator[] Regression(string labelColumnName = DefaultColumnNames.Label, string featureColumnName = DefaultColumnNames.Features, string exampleWeightColumnName = null, bool useFastForest = true, bool useLgbm = true, bool useFastTree = true, bool useLbfgs = true, bool useSdca = true,
            FastTreeOption fastTreeOption = null, LgbmOption lgbmOption = null, FastForestOption fastForestOption = null, LbfgsOption lbfgsOption = null, SdcaOption sdcaOption = null,
            SearchSpace<FastTreeOption> fastTreeSearchSpace = null, SearchSpace<LgbmOption> lgbmSearchSpace = null, SearchSpace<FastForestOption> fastForestSearchSpace = null, SearchSpace<LbfgsOption> lbfgsSearchSpace = null, SearchSpace<SdcaOption> sdcaSearchSpace = null)
        {
            var res = new List<SweepableEstimator>();

            if (useFastTree)
            {
                fastTreeOption = fastTreeOption ?? new FastTreeOption();
                fastTreeOption.LabelColumnName = labelColumnName;
                fastTreeOption.FeatureColumnName = featureColumnName;
                fastTreeOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateFastTreeRegression(fastTreeOption, fastTreeSearchSpace ?? new SearchSpace<FastTreeOption>(fastTreeOption)));
            }

            if (useFastForest)
            {
                fastForestOption = fastForestOption ?? new FastForestOption();
                fastForestOption.LabelColumnName = labelColumnName;
                fastForestOption.FeatureColumnName = featureColumnName;
                fastForestOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateFastForestRegression(fastForestOption, fastForestSearchSpace ?? new SearchSpace<FastForestOption>(fastForestOption)));
            }

            if (useLgbm)
            {
                lgbmOption = lgbmOption ?? new LgbmOption();
                lgbmOption.LabelColumnName = labelColumnName;
                lgbmOption.FeatureColumnName = featureColumnName;
                lgbmOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateLightGbmRegression(lgbmOption, lgbmSearchSpace ?? new SearchSpace<LgbmOption>(lgbmOption)));
            }

            if (useLbfgs)
            {
                lbfgsOption = lbfgsOption ?? new LbfgsOption();
                lbfgsOption.LabelColumnName = labelColumnName;
                lbfgsOption.FeatureColumnName = featureColumnName;
                lbfgsOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateLbfgsPoissonRegressionRegression(lbfgsOption, lbfgsSearchSpace ?? new SearchSpace<LbfgsOption>(lbfgsOption)));
            }

            if (useSdca)
            {
                sdcaOption = sdcaOption ?? new SdcaOption();
                sdcaOption.LabelColumnName = labelColumnName;
                sdcaOption.FeatureColumnName = featureColumnName;
                sdcaOption.ExampleWeightColumnName = exampleWeightColumnName;
                res.Add(SweepableEstimatorFactory.CreateSdcaRegression(sdcaOption, sdcaSearchSpace ?? new SearchSpace<SdcaOption>(sdcaOption)));
            }

            return res.ToArray();
        }

        /// <summary>
        /// Create a list of <see cref="SweepableEstimator"/> for featurizing text.
        /// </summary>
        /// <param name="outputColumnName">output column name.</param>
        /// <param name="inputColumnName">input column name.</param>
        internal SweepableEstimator[] TextFeaturizer(string outputColumnName, string inputColumnName)
        {
            var option = new FeaturizeTextOption
            {
                InputColumnName = inputColumnName,
                OutputColumnName = outputColumnName,
            };

            return new[] { SweepableEstimatorFactory.CreateFeaturizeText(option) };
        }

        /// <summary>
        /// Create a list of <see cref="SweepableEstimator"/> for featurizing numeric columns.
        /// </summary>
        /// <param name="outputColumnNames">output column names.</param>
        /// <param name="inputColumnNames">input column names.</param>
        internal SweepableEstimator[] NumericFeaturizer(string[] outputColumnNames, string[] inputColumnNames)
        {
            Contracts.CheckValue(inputColumnNames, nameof(inputColumnNames));
            Contracts.CheckValue(outputColumnNames, nameof(outputColumnNames));
            Contracts.Check(outputColumnNames.Count() == inputColumnNames.Count() && outputColumnNames.Count() > 0, "outputColumnNames and inputColumnNames must have the same length and greater than 0");
            var replaceMissingValueOption = new ReplaceMissingValueOption
            {
                InputColumnNames = inputColumnNames,
                OutputColumnNames = outputColumnNames,
            };

            return new[] { SweepableEstimatorFactory.CreateReplaceMissingValues(replaceMissingValueOption) };
        }

        /// <summary>
        /// Create a list of <see cref="SweepableEstimator"/> for featurizing catalog columns.
        /// </summary>
        /// <param name="outputColumnNames">output column names.</param>
        /// <param name="inputColumnNames">input column names.</param>
        internal SweepableEstimator[] CatalogFeaturizer(string[] outputColumnNames, string[] inputColumnNames)
        {
            Contracts.Check(outputColumnNames.Count() == inputColumnNames.Count() && outputColumnNames.Count() > 0, "outputColumnNames and inputColumnNames must have the same length and greater than 0");

            var option = new OneHotOption
            {
                InputColumnNames = inputColumnNames,
                OutputColumnNames = outputColumnNames,
            };

            return new SweepableEstimator[] { SweepableEstimatorFactory.CreateOneHotEncoding(option), SweepableEstimatorFactory.CreateOneHotHashEncoding(option) };
        }

        /// <summary>
        /// Create a single featurize pipeline according to <paramref name="data"/>. This function will collect all columns in <paramref name="data"/> and not in <paramref name="excludeColumns"/>,
        /// featurizing them using <see cref="CatalogFeaturizer(string[], string[])"/>, <see cref="NumericFeaturizer(string[], string[])"/> or <see cref="TextFeaturizer(string, string)"/>. And combine
        /// them into a single feature column as output.
        /// </summary>
        /// <param name="data">input data.</param>
        /// <param name="catalogColumns">columns that should be treated as catalog. If not specified, it will automatically infer if a column is catalog or not.</param>
        /// <param name="numericColumns">columns that should be treated as numeric. If not specified, it will automatically infer if a column is catalog or not.</param>
        /// <param name="textColumns">columns that should be treated as text. If not specified, it will automatically infer if a column is catalog or not.</param>
        /// <param name="outputColumnName">output feature column.</param>
        /// <param name="excludeColumns">columns that won't be included when featurizing, like label</param>
        public MultiModelPipeline Featurizer(IDataView data, string outputColumnName = "Features", string[] catalogColumns = null, string[] numericColumns = null, string[] textColumns = null, string[] excludeColumns = null)
        {
            Contracts.CheckValue(data, nameof(data));

            // validate if there's overlapping among catalogColumns, numericColumns, textColumns and excludeColumns
            var overallColumns = new string[][] { catalogColumns, numericColumns, textColumns, excludeColumns }
                                    .Where(c => c != null)
                                    .SelectMany(c => c);

            if (overallColumns != null)
            {
                Contracts.Assert(overallColumns.Count() == overallColumns.Distinct().Count(), "detect overlapping among catalogColumns, numericColumns, textColumns and excludedColumns");
            }

            var columnInfo = new ColumnInformation();

            if (excludeColumns != null)
            {
                foreach (var ignoreColumn in excludeColumns)
                {
                    columnInfo.IgnoredColumnNames.Add(ignoreColumn);
                }
            }

            if (catalogColumns != null)
            {
                foreach (var catalogColumn in catalogColumns)
                {
                    columnInfo.CategoricalColumnNames.Add(catalogColumn);
                }
            }

            if (numericColumns != null)
            {
                foreach (var column in numericColumns)
                {
                    columnInfo.NumericColumnNames.Add(column);
                }
            }

            if (textColumns != null)
            {
                foreach (var column in textColumns)
                {
                    columnInfo.TextColumnNames.Add(column);
                }
            }

            return this.Featurizer(data, columnInfo, outputColumnName);
        }

        /// <summary>
        /// Create a single featurize pipeline according to <paramref name="columnInformation"/>. This function will collect all columns in <paramref name="columnInformation"/>,
        /// featurizing them using <see cref="CatalogFeaturizer(string[], string[])"/>, <see cref="NumericFeaturizer(string[], string[])"/> or <see cref="TextFeaturizer(string, string)"/>. And combine
        /// them into a single feature column as output.
        /// </summary>
        /// <param name="data">input data.</param>
        /// <param name="columnInformation">column information.</param>
        /// <param name="outputColumnName">output feature column.</param>
        /// <returns>A <see cref="MultiModelPipeline"/> for featurization.</returns>
        public MultiModelPipeline Featurizer(IDataView data, ColumnInformation columnInformation, string outputColumnName = "Features")
        {
            Contracts.CheckValue(data, nameof(data));
            Contracts.CheckValue(columnInformation, nameof(columnInformation));

            var columnPurposes = PurposeInference.InferPurposes(this._context, data, columnInformation);
            var textFeatures = columnPurposes.Where(c => c.Purpose == ColumnPurpose.TextFeature);
            var numericFeatures = columnPurposes.Where(c => c.Purpose == ColumnPurpose.NumericFeature);
            var catalogFeatures = columnPurposes.Where(c => c.Purpose == ColumnPurpose.CategoricalFeature);
            var textFeatureColumnNames = textFeatures.Select(c => data.Schema[c.ColumnIndex].Name).ToArray();
            var numericFeatureColumnNames = numericFeatures.Select(c => data.Schema[c.ColumnIndex].Name).ToArray();
            var catalogFeatureColumnNames = catalogFeatures.Select(c => data.Schema[c.ColumnIndex].Name).ToArray();

            var pipeline = new MultiModelPipeline();
            if (numericFeatureColumnNames.Length > 0)
            {
                pipeline = pipeline.Append(this.NumericFeaturizer(numericFeatureColumnNames, numericFeatureColumnNames));
            }

            if (catalogFeatureColumnNames.Length > 0)
            {
                pipeline = pipeline.Append(this.CatalogFeaturizer(catalogFeatureColumnNames, catalogFeatureColumnNames));
            }

            foreach (var textColumn in textFeatureColumnNames)
            {
                pipeline = pipeline.Append(this.TextFeaturizer(textColumn, textColumn));
            }

            var option = new ConcatOption
            {
                InputColumnNames = textFeatureColumnNames.Concat(numericFeatureColumnNames).Concat(catalogFeatureColumnNames).ToArray(),
                OutputColumnName = outputColumnName,
            };

            if (option.InputColumnNames.Length > 0)
            {
                pipeline = pipeline.Append(SweepableEstimatorFactory.CreateConcatenate(option));
            }

            return pipeline;
        }
    }
}
