﻿using System;
using Microsoft.ML.Data;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public class ScoreSchemaTest : TestDataViewBase
    {
        public ScoreSchemaTest(ITestOutputHelper helper)
           : base(helper)
        {
        }

        private VBuffer<ReadOnlyMemory<char>> GenerateKeyNames(int keyCount)
        {
            // Initialize an empty array of strings.
            VBuffer<ReadOnlyMemory<char>> buffer = default;

            // Add strings to the empty buffer using a buffer editor.
            var bufferEditor = VBufferEditor.Create(ref buffer, keyCount);
            for (int i = 0; i < keyCount; ++i)
                bufferEditor.Values[i] = string.Format($"Key-{i}").AsMemory();

            // The input buffer is a string array containing {"Key-{0}", ..., "Key-{keyCount-1}"} now.
            return bufferEditor.Commit();
        }

        [Fact]
        public void SequencePredictorSchemaTest()
        {
            int keyCount = 10;
            var scoreColumnType = new KeyType(DataKind.U4, 0, keyCount);
            VBuffer<ReadOnlyMemory<char>> keyNames = GenerateKeyNames(keyCount);

            var sequenceSchema = ScoreSchemaFactory.CreateSequencePredictionSchema(scoreColumnType,
                MetadataUtils.Const.ScoreColumnKind.SequenceClassification, keyNames);

            // Output schema should only contain one column, which is the predicted label.
            Assert.Single(sequenceSchema);
            var scoreColumn = sequenceSchema[0];

            // Check score column name.
            Assert.Equal(MetadataUtils.Const.ScoreValueKind.PredictedLabel, scoreColumn.Name);

            // Check score column type.
            Assert.True(scoreColumn.Type is KeyType);
            Assert.Equal((scoreColumnType as KeyType).Min, (scoreColumn.Type as KeyType).Min);
            Assert.Equal((scoreColumnType as KeyType).Count, (scoreColumn.Type as KeyType).Count);
            Assert.Equal((scoreColumnType as KeyType).RawKind, (scoreColumn.Type as KeyType).RawKind);
            Assert.Equal((scoreColumnType as KeyType).Contiguous, (scoreColumn.Type as KeyType).Contiguous);

            // Check metadata. Because keyNames is not empty, there should be three metadata fields.
            var scoreMetadata = scoreColumn.Metadata;
            Assert.Equal(3, scoreMetadata.Schema.Count);

            // Check metadata columns' names.
            Assert.Equal(MetadataUtils.Kinds.KeyValues, scoreMetadata.Schema[0].Name);
            Assert.Equal(MetadataUtils.Kinds.ScoreColumnKind, scoreMetadata.Schema[1].Name);
            Assert.Equal(MetadataUtils.Kinds.ScoreValueKind, scoreMetadata.Schema[2].Name);

            // Check metadata columns' types.
            Assert.True(scoreMetadata.Schema[0].Type.IsVector);
            Assert.Equal(keyNames.Length, (scoreMetadata.Schema[0].Type as VectorType).VectorSize);
            Assert.Equal(TextType.Instance, (scoreMetadata.Schema[0].Type as VectorType).ItemType);
            Assert.Equal(TextType.Instance, scoreColumn.Metadata.Schema[1].Type);
            Assert.Equal(TextType.Instance, scoreColumn.Metadata.Schema[2].Type);

            // Check metadata columns' values.
            var keyNamesGetter = scoreMetadata.GetGetter<VBuffer<ReadOnlyMemory<char>>>(0);
            var actualKeyNames = new VBuffer<ReadOnlyMemory<char>>();
            keyNamesGetter(ref actualKeyNames);
            Assert.Equal(keyNames.Length, actualKeyNames.Length);
            Assert.Equal(keyNames.DenseValues(), actualKeyNames.DenseValues());

            var scoreColumnKindGetter = scoreMetadata.GetGetter<ReadOnlyMemory<char>>(1);
            ReadOnlyMemory<char> scoreColumnKindValue = null;
            scoreColumnKindGetter(ref scoreColumnKindValue);
            Assert.Equal(MetadataUtils.Const.ScoreColumnKind.SequenceClassification, scoreColumnKindValue.ToString());

            var scoreValueKindGetter = scoreMetadata.GetGetter<ReadOnlyMemory<char>>(2);
            ReadOnlyMemory<char> scoreValueKindValue = null;
            scoreValueKindGetter(ref scoreValueKindValue);
            Assert.Equal(MetadataUtils.Const.ScoreValueKind.PredictedLabel, scoreValueKindValue.ToString());
        }

        [Fact]
        public void SequencePredictorSchemaWithoutKeyNamesMetadataTest()
        {
            int keyCount = 10;
            var scoreColumnType = new KeyType(DataKind.U4, 0, keyCount);
            VBuffer<ReadOnlyMemory<char>> keyNames = GenerateKeyNames(0);

            var sequenceSchema = ScoreSchemaFactory.CreateSequencePredictionSchema(scoreColumnType,
                MetadataUtils.Const.ScoreColumnKind.SequenceClassification, keyNames);

            // Output schema should only contain one column, which is the predicted label.
            Assert.Single(sequenceSchema);
            var scoreColumn = sequenceSchema[0];

            // Check score column name.
            Assert.Equal(MetadataUtils.Const.ScoreValueKind.PredictedLabel, scoreColumn.Name);

            // Check score column type.
            Assert.True(scoreColumn.Type is KeyType);
            Assert.Equal((scoreColumnType as KeyType).Min, (scoreColumn.Type as KeyType).Min);
            Assert.Equal((scoreColumnType as KeyType).Count, (scoreColumn.Type as KeyType).Count);
            Assert.Equal((scoreColumnType as KeyType).RawKind, (scoreColumn.Type as KeyType).RawKind);
            Assert.Equal((scoreColumnType as KeyType).Contiguous, (scoreColumn.Type as KeyType).Contiguous);

            // Check metadata. Because keyNames is not empty, there should be three metadata fields.
            var scoreMetadata = scoreColumn.Metadata;
            Assert.Equal(2, scoreMetadata.Schema.Count);

            // Check metadata columns' names.
            Assert.Equal(MetadataUtils.Kinds.ScoreColumnKind, scoreMetadata.Schema[0].Name);
            Assert.Equal(MetadataUtils.Kinds.ScoreValueKind, scoreMetadata.Schema[1].Name);

            // Check metadata columns' types.
            Assert.Equal(TextType.Instance, scoreColumn.Metadata.Schema[0].Type);
            Assert.Equal(TextType.Instance, scoreColumn.Metadata.Schema[1].Type);

            // Check metadata columns' values.
            var scoreColumnKindGetter = scoreMetadata.GetGetter<ReadOnlyMemory<char>>(0);
            ReadOnlyMemory<char> scoreColumnKindValue = null;
            scoreColumnKindGetter(ref scoreColumnKindValue);
            Assert.Equal(MetadataUtils.Const.ScoreColumnKind.SequenceClassification, scoreColumnKindValue.ToString());

            var scoreValueKindGetter = scoreMetadata.GetGetter<ReadOnlyMemory<char>>(1);
            ReadOnlyMemory<char> scoreValueKindValue = null;
            scoreValueKindGetter(ref scoreValueKindValue);
            Assert.Equal(MetadataUtils.Const.ScoreValueKind.PredictedLabel, scoreValueKindValue.ToString());
        }
    }
}
