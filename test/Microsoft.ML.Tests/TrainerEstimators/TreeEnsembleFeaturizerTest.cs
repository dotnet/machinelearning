// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
        [Fact]
        public void TreeEnsembleFeaturizerOutputSchemaTest()
        {
            // Create data set
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(1000).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);

            // Define a tree model whose trees will be extracted to construct a tree featurizer.
            var trainer = ML.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options
                {
                    NumberOfThreads = 1,
                    NumberOfTrees = 10,
                    NumberOfLeaves = 5,
                });

            // Train the defined tree model.
            var model = trainer.Fit(dataView);

            // From the trained tree model, a mapper of tree featurizer is created.
            const string treesColumnName = "MyTrees";
            const string leavesColumnName = "MyLeaves";
            const string pathsColumnName = "MyPaths";
            var args = new TreeEnsembleFeaturizerBindableMapper.Arguments()
            {
                TreesColumnName = treesColumnName,
                LeavesColumnName = leavesColumnName,
                PathsColumnName = pathsColumnName
            };
            var treeFeaturizer = new TreeEnsembleFeaturizerBindableMapper(Env, args, model.Model);

            // To get output schema, we need to create RoleMappedSchema for calling Bind(...).
            var roleMappedSchema = new RoleMappedSchema(dataView.Schema,
                label: nameof(SamplesUtils.DatasetUtils.BinaryLabelFloatFeatureVectorFloatWeightSample.Label),
                feature: nameof(SamplesUtils.DatasetUtils.BinaryLabelFloatFeatureVectorFloatWeightSample.Features));

            // Retrieve output schema. 
            var boundMapper = (treeFeaturizer as ISchemaBindableMapper).Bind(Env, roleMappedSchema);
            var outputSchema = boundMapper.OutputSchema;

            {
                // Check if output schema is correct.
                var treeValuesColumn = outputSchema[0];
                Assert.Equal(treesColumnName, treeValuesColumn.Name);
                VectorDataViewType treeValuesType = treeValuesColumn.Type as VectorDataViewType;
                Assert.NotNull(treeValuesType);
                Assert.Equal(NumberDataViewType.Single, treeValuesType.ItemType);
                Assert.Equal(10, treeValuesType.Size);
                // Below we check the only metadata field.
                Assert.Single(treeValuesColumn.Annotations.Schema);
                VBuffer<ReadOnlyMemory<char>> slotNames = default;
                treeValuesColumn.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref slotNames);
                Assert.Equal(10, slotNames.Length);
                // Just check the head and the tail of the extracted vector.
                Assert.Equal("Tree000", slotNames.GetItemOrDefault(0).ToString());
                Assert.Equal("Tree009", slotNames.GetItemOrDefault(9).ToString());
            }

            {
                var treeLeafIdsColumn = outputSchema[1];
                // Check column of tree leaf IDs.
                Assert.Equal(leavesColumnName, treeLeafIdsColumn.Name);
                VectorDataViewType treeLeafIdsType = treeLeafIdsColumn.Type as VectorDataViewType;
                Assert.NotNull(treeLeafIdsType);
                Assert.Equal(NumberDataViewType.Single, treeLeafIdsType.ItemType);
                Assert.Equal(50, treeLeafIdsType.Size);
                // Below we check the two leaf-IDs column's metadata fields.
                Assert.Equal(2, treeLeafIdsColumn.Annotations.Schema.Count);
                // Check metadata field IsNormalized's content.
                bool leafIdsNormalizedFlag = false;
                treeLeafIdsColumn.Annotations.GetValue(AnnotationUtils.Kinds.IsNormalized, ref leafIdsNormalizedFlag);
                Assert.True(leafIdsNormalizedFlag);
                // Check metadata field SlotNames's content.
                VBuffer<ReadOnlyMemory<char>> leafIdsSlotNames = default;
                treeLeafIdsColumn.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref leafIdsSlotNames);
                Assert.Equal(50, leafIdsSlotNames.Length);
                // Just check the head and the tail of the extracted vector.
                Assert.Equal("Tree000Leaf000", leafIdsSlotNames.GetItemOrDefault(0).ToString());
                Assert.Equal("Tree009Leaf004", leafIdsSlotNames.GetItemOrDefault(49).ToString());
            }

            {
                var treePathIdsColumn = outputSchema[2];
                // Check column of path IDs.
                Assert.Equal(pathsColumnName, treePathIdsColumn.Name);
                VectorDataViewType treePathIdsType = treePathIdsColumn.Type as VectorDataViewType;
                Assert.NotNull(treePathIdsType);
                Assert.Equal(NumberDataViewType.Single, treePathIdsType.ItemType);
                Assert.Equal(40, treePathIdsType.Size);
                // Below we check the two path-IDs column's metadata fields.
                Assert.Equal(2, treePathIdsColumn.Annotations.Schema.Count);
                // Check metadata field IsNormalized's content.
                bool pathIdsNormalizedFlag = false;
                treePathIdsColumn.Annotations.GetValue(AnnotationUtils.Kinds.IsNormalized, ref pathIdsNormalizedFlag);
                Assert.True(pathIdsNormalizedFlag);
                // Check metadata field SlotNames's content.
                VBuffer<ReadOnlyMemory<char>> pathIdsSlotNames = default;
                treePathIdsColumn.Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref pathIdsSlotNames);
                Assert.Equal(40, pathIdsSlotNames.Length);
                // Just check the head and the tail of the extracted vector.
                Assert.Equal("Tree000Node000", pathIdsSlotNames.GetItemOrDefault(0).ToString());
                Assert.Equal("Tree009Node003", pathIdsSlotNames.GetItemOrDefault(39).ToString());
            }

        }

        [Fact]
        public void TreeEnsembleFeaturizerTransformerFastTreeBinary()
        {
            // Create data set
            int dataPointCount = 20;
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);

            // Define a tree model whose trees will be extracted to construct a tree featurizer.
            var trainer = ML.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options
                {
                    NumberOfThreads = 1,
                    NumberOfTrees = 1,
                    NumberOfLeaves = 4,
                    MinimumExampleCountPerLeaf = 1
                });

            // Train the defined tree model.
            var model = trainer.Fit(dataView);
            var predicted = model.Transform(dataView);

            // From the trained tree model, a mapper of tree featurizer is created.
            const string treesColumnName = "MyTrees";
            const string leavesColumnName = "MyLeaves";
            const string pathsColumnName = "MyPaths";
            var treeFeaturizer = new TreeEnsembleFeaturizationTransformer(ML, dataView.Schema, dataView.Schema["Features"], model.Model.SubModel,
                treesColumnName: treesColumnName, leavesColumnName: leavesColumnName, pathsColumnName: pathsColumnName);

            // Apply TreeEnsembleFeaturizer to the input data.
            var transformed = treeFeaturizer.Transform(dataView);

            // Extract the outputs of TreeEnsembleFeaturizer.
            var features = transformed.GetColumn<float[]>("Features").ToArray();
            var leafValues = transformed.GetColumn<float[]>(treesColumnName).ToArray();
            var leafIds = transformed.GetColumn<float[]>(leavesColumnName).ToArray();
            var paths = transformed.GetColumn<float[]>(pathsColumnName).ToArray();

            // Check if the TreeEnsembleFeaturizer produce expected values.
            List<int> path = null;
            for (int dataPointIndex = 0; dataPointIndex < dataPointCount; ++dataPointIndex)
            {
                int treeIndex = 0;
                var leafId = model.Model.SubModel.GetLeaf(treeIndex, new VBuffer<float>(10, features[dataPointIndex]), ref path);
                var leafValue = model.Model.SubModel.GetLeafValue(0, leafId);
                Assert.Equal(leafValues[dataPointIndex][treeIndex], leafValue);
                Assert.Equal(1.0, leafIds[dataPointIndex][leafId]);
                foreach (var nodeId in path)
                    Assert.Equal(1.0, paths[dataPointIndex][nodeId]);
            }
        }

        [Fact]
        public void TreeEnsembleFeaturizerTransformerFastForestBinary()
        {
            // Create data set
            int dataPointCount = 20;
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);

            // Define a tree model whose trees will be extracted to construct a tree featurizer.
            var trainer = ML.BinaryClassification.Trainers.FastForest(
                new FastForestBinaryTrainer.Options
                {
                    NumberOfThreads = 1,
                    NumberOfTrees = 1,
                    NumberOfLeaves = 4,
                    MinimumExampleCountPerLeaf = 1
                });

            // Train the defined tree model.
            var model = trainer.Fit(dataView);

            // From the trained tree model, a mapper of tree featurizer is created.
            const string treesColumnName = "MyTrees";
            const string leavesColumnName = "MyLeaves";
            const string pathsColumnName = "MyPaths";
            var treeFeaturizer = new TreeEnsembleFeaturizationTransformer(ML, dataView.Schema, dataView.Schema["Features"], model.Model,
                treesColumnName: treesColumnName, leavesColumnName: leavesColumnName, pathsColumnName: pathsColumnName);

            // Apply TreeEnsembleFeaturizer to the input data.
            var transformed = treeFeaturizer.Transform(dataView);

            // Extract the outputs of TreeEnsembleFeaturizer.
            var features = transformed.GetColumn<float[]>("Features").ToArray();
            var leafValues = transformed.GetColumn<float[]>(treesColumnName).ToArray();
            var leafIds = transformed.GetColumn<float[]>(leavesColumnName).ToArray();
            var paths = transformed.GetColumn<float[]>(pathsColumnName).ToArray();

            // Check if the TreeEnsembleFeaturizer produce expected values.
            List<int> path = null;
            for (int dataPointIndex = 0; dataPointIndex < dataPointCount; ++dataPointIndex)
            {
                int treeIndex = 0;
                var leafId = model.Model.GetLeaf(treeIndex, new VBuffer<float>(10, features[dataPointIndex]), ref path);
                var leafValue = model.Model.GetLeafValue(0, leafId);
                Assert.Equal(leafValues[dataPointIndex][treeIndex], leafValue);
                Assert.Equal(1.0, leafIds[dataPointIndex][leafId]);
                foreach (var nodeId in path)
                    Assert.Equal(1.0, paths[dataPointIndex][nodeId]);
            }
        }

        /// <summary>
        /// A test of <see cref="PretrainedTreeFeaturizationEstimator"/>.
        /// </summary>
        [Fact]
        public void TestPretrainedTreeFeaturizationEstimator()
        {
            // Create data set
            int dataPointCount = 20;
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            // Define a tree model whose trees will be extracted to construct a tree featurizer.
            var trainer = ML.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options
                {
                    NumberOfThreads = 1,
                    NumberOfTrees = 1,
                    NumberOfLeaves = 4,
                    MinimumExampleCountPerLeaf = 1
                });

            // Train the defined tree model.
            var model = trainer.Fit(dataView);
            var predicted = model.Transform(dataView);

            // From the trained tree model, a mapper of tree featurizer is created.
            string featureColumnName = "Features";
            string treesColumnName = "MyTrees"; // a tree-based feature column.
            string leavesColumnName = "MyLeaves"; // a tree-based feature column.
            string pathsColumnName = "MyPaths"; // a tree-based feature column.
            var options = new PretrainedTreeFeaturizationEstimator.Options()
            {
                InputColumnName = featureColumnName,
                ModelParameters = model.Model.SubModel,
                TreesColumnName = treesColumnName,
                LeavesColumnName = leavesColumnName,
                PathsColumnName = pathsColumnName
            };
            var treeFeaturizer = ML.Transforms.FeaturizeByPretrainTreeEnsemble(options).Fit(dataView);

            // Apply TreeEnsembleFeaturizer to the input data.
            var transformed = treeFeaturizer.Transform(dataView);

            // Extract the outputs of TreeEnsembleFeaturizer.
            var features = transformed.GetColumn<float[]>(featureColumnName).ToArray();
            var leafValues = transformed.GetColumn<float[]>(treesColumnName).ToArray();
            var leafIds = transformed.GetColumn<float[]>(leavesColumnName).ToArray();
            var paths = transformed.GetColumn<float[]>(pathsColumnName).ToArray();

            // Check if the TreeEnsembleFeaturizer produce expected values.
            List<int> path = null;
            for (int dataPointIndex = 0; dataPointIndex < dataPointCount; ++dataPointIndex)
            {
                int treeIndex = 0;
                var leafId = model.Model.SubModel.GetLeaf(treeIndex, new VBuffer<float>(10, features[dataPointIndex]), ref path);
                var leafValue = model.Model.SubModel.GetLeafValue(0, leafId);
                Assert.Equal(leafValues[dataPointIndex][treeIndex], leafValue);
                Assert.Equal(1.0, leafIds[dataPointIndex][leafId]);
                foreach (var nodeId in path)
                    Assert.Equal(1.0, paths[dataPointIndex][nodeId]);
            }
        }

        /// <summary>
        /// This test contains several steps.
        ///   1. It first trains a <see cref="FastTreeBinaryModelParameters"/> using <see cref="FastTreeBinaryTrainer"/>.
        ///   2. Then, it creates the a <see cref="PretrainedTreeFeaturizationEstimator"/> from the trained <see cref="FastTreeBinaryModelParameters"/>.
        ///   3. The feature produced in step 2 would be fed into <see cref="SdcaLogisticRegression"/> to enhance the training accuracy of that linear model.
        ///   4. We train another <see cref="SdcaLogisticRegression"/> without features from trees and finally compare their scores.
        /// </summary>
        [Fact]
        public void TreeEnsembleFeaturizingPipeline()
        {
            // Create data set
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            // Define a tree model whose trees will be extracted to construct a tree featurizer.
            var trainer = ML.BinaryClassification.Trainers.FastTree(
                new FastTreeBinaryTrainer.Options
                {
                    NumberOfThreads = 1,
                    NumberOfTrees = 10,
                    NumberOfLeaves = 4,
                    MinimumExampleCountPerLeaf = 10
                });

            // Train the defined tree model. This trained model will be used to construct TreeEnsembleFeaturizationEstimator.
            var treeModel = trainer.Fit(dataView);
            var predicted = treeModel.Transform(dataView);

            // Combine the output of TreeEnsembleFeaturizationTransformer and the original features as the final training features.
            // Then train a linear model.
            var options = new PretrainedTreeFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                ModelParameters = treeModel.Model.SubModel
            };
            var pipeline = ML.Transforms.FeaturizeByPretrainTreeEnsemble(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                .Append(ML.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.BinaryClassification.Evaluate(prediction);

            // Then train the same linear model without tree features.
            var naivePipeline = ML.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features");
            var naiveModel = naivePipeline.Fit(dataView);
            var naivePrediction = naiveModel.Transform(dataView);
            var naiveMetrics = ML.BinaryClassification.Evaluate(naivePrediction);

            // The linear model trained with tree features should perform better than that without tree features.
            Assert.True(metrics.Accuracy > naiveMetrics.Accuracy);
            Assert.True(metrics.LogLoss < naiveMetrics.LogLoss);
            Assert.True(metrics.AreaUnderPrecisionRecallCurve > naiveMetrics.AreaUnderPrecisionRecallCurve);
        }

        [Fact]
        public void TestFastTreeBinaryFeaturizationInPipeline()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastTreeBinaryTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            var options = new FastTreeBinaryFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                TrainerOptions = trainerOptions
            };

            var pipeline = ML.Transforms.FeaturizeByFastTreeBinary(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                .Append(ML.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.BinaryClassification.Evaluate(prediction);

            Assert.True(metrics.Accuracy > 0.98);
            Assert.True(metrics.LogLoss < 0.05);
            Assert.True(metrics.AreaUnderPrecisionRecallCurve > 0.98);
        }

        [Fact]
        public void TestFastForestBinaryFeaturizationInPipeline()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastForestBinaryTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            var options = new FastForestBinaryFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                TrainerOptions = trainerOptions
            };

            var pipeline = ML.Transforms.FeaturizeByFastForestBinary(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                .Append(ML.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.BinaryClassification.Evaluate(prediction);

            Assert.True(metrics.Accuracy > 0.97);
            Assert.True(metrics.LogLoss < 0.07);
            Assert.True(metrics.AreaUnderPrecisionRecallCurve > 0.98);
        }

        [Fact]
        public void TestFastTreeRegressionFeaturizationInPipeline()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastTreeRegressionTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            var options = new FastTreeRegressionFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                TrainerOptions = trainerOptions
            };

            var pipeline = ML.Transforms.FeaturizeByFastTreeRegression(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                .Append(ML.Regression.Trainers.Sdca("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.Regression.Evaluate(prediction);

            Assert.True(metrics.MeanAbsoluteError < 0.2);
            Assert.True(metrics.MeanSquaredError < 0.05);
        }

        [Fact]
        public void TestFastForestRegressionFeaturizationInPipeline()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastForestRegressionTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            var options = new FastForestRegressionFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                TrainerOptions = trainerOptions
            };

            var pipeline = ML.Transforms.FeaturizeByFastForestRegression(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                .Append(ML.Regression.Trainers.Sdca("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.Regression.Evaluate(prediction);

            Assert.True(metrics.MeanAbsoluteError < 0.25);
            Assert.True(metrics.MeanSquaredError < 0.1);
        }

        [Fact]
        public void TestFastTreeTweedieFeaturizationInPipeline()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastTreeTweedieTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            var options = new FastTreeTweedieFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                TrainerOptions = trainerOptions
            };

            var pipeline = ML.Transforms.FeaturizeByFastTreeTweedie(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                .Append(ML.Regression.Trainers.Sdca("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.Regression.Evaluate(prediction);

            Assert.True(metrics.MeanAbsoluteError < 0.25);
            Assert.True(metrics.MeanSquaredError < 0.1);
        }

        [Fact]
        public void TestFastTreeRankingFeaturizationInPipeline()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorUlongGroupIdSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastTreeRankingTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            var options = new FastTreeRankingFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                TrainerOptions = trainerOptions
            };

            var pipeline = ML.Transforms.FeaturizeByFastTreeRanking(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                .Append(ML.Regression.Trainers.Sdca("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.Regression.Evaluate(prediction);

            Assert.True(metrics.MeanAbsoluteError < 0.25);
            Assert.True(metrics.MeanSquaredError < 0.1);
        }

        [Fact]
        public void TestSaveAndLoadTreeFeaturizer()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastForestRegressionTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            var options = new FastForestRegressionFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                TrainerOptions = trainerOptions
            };

            var pipeline = ML.Transforms.FeaturizeByFastForestRegression(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                .Append(ML.Regression.Trainers.Sdca("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.Regression.Evaluate(prediction);

            Assert.True(metrics.MeanAbsoluteError < 0.25);
            Assert.True(metrics.MeanSquaredError < 0.1);

            // Save the trained model into file.
            ITransformer loadedModel = null;
            var tempPath = Path.GetTempFileName();
            using (var file = new SimpleFileHandle(Env, tempPath, true, true))
            {
                using (var fs = file.CreateWriteStream())
                    ML.Model.Save(model, null, fs);

                using (var fs = file.OpenReadStream())
                    loadedModel = ML.Model.Load(fs, out var schema);
            }
            var loadedPrediction = loadedModel.Transform(dataView);
            var loadedMetrics = ML.Regression.Evaluate(loadedPrediction);

            Assert.Equal(metrics.MeanAbsoluteError, loadedMetrics.MeanAbsoluteError);
            Assert.Equal(metrics.MeanSquaredError, loadedMetrics.MeanSquaredError);
        }

        [Fact]
        public void TestSaveAndLoadDoubleTreeFeaturizer()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateFloatLabelFloatFeatureVectorSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastForestRegressionTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            // Trains tree featurization on "Features" and applies on "CopiedFeatures".
            var options = new FastForestRegressionFeaturizationEstimator.Options()
            {
                InputColumnName = "CopiedFeatures",
                TrainerOptions = trainerOptions,
                TreesColumnName = "OhMyTrees",
                LeavesColumnName = "OhMyLeaves",
                PathsColumnName = "OhMyPaths"
            };

            var pipeline = ML.Transforms.CopyColumns("CopiedFeatures", "Features")
                .Append(ML.Transforms.FeaturizeByFastForestRegression(options))
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "OhMyTrees", "OhMyLeaves", "OhMyPaths"))
                .Append(ML.Regression.Trainers.Sdca("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.Regression.Evaluate(prediction);

            Assert.True(metrics.MeanAbsoluteError < 0.25);
            Assert.True(metrics.MeanSquaredError < 0.1);

            // Save the trained model into file and then load it back.
            ITransformer loadedModel = null;
            var tempPath = Path.GetTempFileName();
            using (var file = new SimpleFileHandle(Env, tempPath, true, true))
            {
                using (var fs = file.CreateWriteStream())
                    ML.Model.Save(model, null, fs);

                using (var fs = file.OpenReadStream())
                    loadedModel = ML.Model.Load(fs, out var schema);
            }

            // Compute prediction using the loaded model.
            var loadedPrediction = loadedModel.Transform(dataView);
            var loadedMetrics = ML.Regression.Evaluate(loadedPrediction);

            // Check if the loaded model produces the same result as the trained model.
            Assert.Equal(metrics.MeanAbsoluteError, loadedMetrics.MeanAbsoluteError);
            Assert.Equal(metrics.MeanSquaredError, loadedMetrics.MeanSquaredError);

            var secondPipeline = ML.Transforms.CopyColumns("CopiedFeatures", "Features")
                .Append(ML.Transforms.NormalizeBinning("CopiedFeatures"))
                .Append(ML.Transforms.FeaturizeByFastForestRegression(options))
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "OhMyTrees", "OhMyLeaves", "OhMyPaths"))
                .Append(ML.Regression.Trainers.Sdca("Label", "CombinedFeatures"));
            var secondModel = secondPipeline.Fit(dataView);
            var secondPrediction = secondModel.Transform(dataView);
            var secondMetrics = ML.Regression.Evaluate(secondPrediction);

            // The second pipeline trains a tree featurizer on a bin-based normalized feature, so the second pipeline
            // is different from the first pipeline.
            Assert.NotEqual(metrics.MeanAbsoluteError, secondMetrics.MeanAbsoluteError);
            Assert.NotEqual(metrics.MeanSquaredError, secondMetrics.MeanSquaredError);
        }

        [Fact]
        public void TestFastTreeBinaryFeaturizationInPipelineWithOptionalOutputs()
        {
            int dataPointCount = 200;
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorFloatWeightSamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastTreeBinaryTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "Label"
            };

            var options = new FastTreeBinaryFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TrainerOptions = trainerOptions,
                TreesColumnName = null,
                PathsColumnName = null,
                LeavesColumnName = "Leaves"
            };


            bool isWrong = false;
            try
            {
                var wrongPipeline = ML.Transforms.FeaturizeByFastTreeBinary(options)
                    .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Trees", "Leaves", "Paths"))
                    .Append(ML.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "CombinedFeatures"));
                var wrongModel = wrongPipeline.Fit(dataView);
            }
            catch
            {
                isWrong = true; // Only "Leaves" is produced by the tree featurizer, so accessing "Trees" and "Paths" will lead to an error.
            }
            Assert.True(isWrong);

            var pipeline = ML.Transforms.FeaturizeByFastTreeBinary(options)
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Features", "Leaves"))
                .Append(ML.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "CombinedFeatures"));
            var model = pipeline.Fit(dataView);
            var prediction = model.Transform(dataView);
            var metrics = ML.BinaryClassification.Evaluate(prediction);

            Assert.True(metrics.Accuracy > 0.98);
            Assert.True(metrics.LogLoss < 0.05);
            Assert.True(metrics.AreaUnderPrecisionRecallCurve > 0.98);
        }

        /// <summary>
        /// Apply tree-based featurization on multiclass classification by converting key-typed labels to floats and training
        /// a regression tree model for featurization.
        /// </summary>
        [Fact]
        public void TreeEnsembleFeaturizingPipelineMulticlass()
        {
            int dataPointCount = 1000;
            var data = SamplesUtils.DatasetUtils.GenerateRandomMulticlassClassificationExamples(dataPointCount).ToList();
            var dataView = ML.Data.LoadFromEnumerable(data);
            dataView = ML.Data.Cache(dataView);

            var trainerOptions = new FastForestRegressionTrainer.Options
            {
                NumberOfThreads = 1,
                NumberOfTrees = 10,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 10,
                FeatureColumnName = "Features",
                LabelColumnName = "FloatLabel",
                ShuffleLabels = true
            };

            var options = new FastForestRegressionFeaturizationEstimator.Options()
            {
                InputColumnName = "Features",
                TreesColumnName = "Trees",
                LeavesColumnName = "Leaves",
                PathsColumnName = "Paths",
                TrainerOptions = trainerOptions
            };

            Action<RowWithKey, RowWithFloat> actionConvertKeyToFloat = (RowWithKey rowWithKey, RowWithFloat rowWithFloat) =>
            {
                rowWithFloat.FloatLabel = rowWithKey.KeyLabel == 0 ? float.NaN : rowWithKey.KeyLabel - 1;
            };

            var split = ML.Data.TrainTestSplit(dataView, 0.5);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            var pipeline = ML.Transforms.Conversion.MapValueToKey("KeyLabel", "Label")
                .Append(ML.Transforms.CustomMapping(actionConvertKeyToFloat, "KeyLabel"))
                .Append(ML.Transforms.FeaturizeByFastForestRegression(options))
                .Append(ML.Transforms.Concatenate("CombinedFeatures", "Trees", "Leaves", "Paths"))
                .Append(ML.MulticlassClassification.Trainers.SdcaMaximumEntropy("KeyLabel", "CombinedFeatures"));

            var model = pipeline.Fit(trainData);
            var prediction = model.Transform(testData);
            var metrics = ML.MulticlassClassification.Evaluate(prediction, labelColumnName: "KeyLabel");

            Assert.True(metrics.MacroAccuracy > 0.6);
            Assert.True(metrics.MicroAccuracy > 0.6);
        }

        private class RowWithKey
        {
            [KeyType(4)]
            public uint KeyLabel { get; set; }
        }

        private class RowWithFloat
        {
            public float FloatLabel { get; set; }
        }
    }
}
