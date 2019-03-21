// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
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
            var data = SamplesUtils.DatasetUtils.GenerateBinaryLabelFloatFeatureVectorSamples(1000).ToList();
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
            var treeFeaturizer = new TreeEnsembleFeaturizerBindableMapper(Env, new TreeEnsembleFeaturizerBindableMapper.Arguments(), model.Model);

            // To get output schema, we need to create RoleMappedSchema for calling Bind(...).
            var roleMappedSchema = new RoleMappedSchema(dataView.Schema,
                label: nameof(SamplesUtils.DatasetUtils.BinaryLabelFloatFeatureVectorSample.Label),
                feature: nameof(SamplesUtils.DatasetUtils.BinaryLabelFloatFeatureVectorSample.Features));

            // Retrieve output schema. 
            var boundMapper = (treeFeaturizer as ISchemaBindableMapper).Bind(Env, roleMappedSchema);
            var outputSchema = boundMapper.OutputSchema;

            {
                // Check if output schema is correct.
                var treeValuesColumn = outputSchema[0];
                Assert.Equal("Trees", treeValuesColumn.Name);
                VectorType treeValuesType = treeValuesColumn.Type as VectorType;
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
                Assert.Equal("Leaves", treeLeafIdsColumn.Name);
                VectorType treeLeafIdsType = treeLeafIdsColumn.Type as VectorType;
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
                Assert.Equal("Paths", treePathIdsColumn.Name);
                VectorType treePathIdsType = treePathIdsColumn.Type as VectorType;
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
    }
}
