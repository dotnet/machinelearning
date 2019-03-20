using System;
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
            var expectedScoreColumnType = new KeyType(typeof(uint), keyCount);
            VBuffer<ReadOnlyMemory<char>> keyNames = GenerateKeyNames(keyCount);

            var sequenceSchema = ScoreSchemaFactory.CreateSequencePredictionSchema(expectedScoreColumnType,
                AnnotationUtils.Const.ScoreColumnKind.SequenceClassification, keyNames);

            // Output schema should only contain one column, which is the predicted label.
            Assert.Single(sequenceSchema);
            var scoreColumn = sequenceSchema[0];

            // Check score column name.
            Assert.Equal(AnnotationUtils.Const.ScoreValueKind.PredictedLabel, scoreColumn.Name);

            // Check score column type.
            var actualScoreColumnType = scoreColumn.Type as KeyType;
            Assert.NotNull(actualScoreColumnType);
            Assert.Equal(expectedScoreColumnType.Count, actualScoreColumnType.Count);
            Assert.Equal(expectedScoreColumnType.RawType, actualScoreColumnType.RawType);

            // Check metadata. Because keyNames is not empty, there should be three metadata fields.
            var scoreMetadata = scoreColumn.Annotations;
            Assert.Equal(3, scoreMetadata.Schema.Count);

            // Check metadata columns' names.
            Assert.Equal(AnnotationUtils.Kinds.KeyValues, scoreMetadata.Schema[0].Name);
            Assert.Equal(AnnotationUtils.Kinds.ScoreColumnKind, scoreMetadata.Schema[1].Name);
            Assert.Equal(AnnotationUtils.Kinds.ScoreValueKind, scoreMetadata.Schema[2].Name);

            // Check metadata columns' types.
            Assert.True(scoreMetadata.Schema[0].Type is VectorType);
            Assert.Equal(keyNames.Length, (scoreMetadata.Schema[0].Type as VectorType).Size);
            Assert.Equal(TextDataViewType.Instance, (scoreMetadata.Schema[0].Type as VectorType).ItemType);
            Assert.Equal(TextDataViewType.Instance, scoreColumn.Annotations.Schema[1].Type);
            Assert.Equal(TextDataViewType.Instance, scoreColumn.Annotations.Schema[2].Type);

            // Check metadata columns' values.
            var keyNamesGetter = scoreMetadata.GetGetter<VBuffer<ReadOnlyMemory<char>>>(scoreMetadata.Schema[0]);
            var actualKeyNames = new VBuffer<ReadOnlyMemory<char>>();
            keyNamesGetter(ref actualKeyNames);
            Assert.Equal(keyNames.Length, actualKeyNames.Length);
            Assert.Equal(keyNames.DenseValues(), actualKeyNames.DenseValues());

            var scoreColumnKindGetter = scoreMetadata.GetGetter<ReadOnlyMemory<char>>(scoreMetadata.Schema[1]);
            ReadOnlyMemory<char> scoreColumnKindValue = null;
            scoreColumnKindGetter(ref scoreColumnKindValue);
            Assert.Equal(AnnotationUtils.Const.ScoreColumnKind.SequenceClassification, scoreColumnKindValue.ToString());

            var scoreValueKindGetter = scoreMetadata.GetGetter<ReadOnlyMemory<char>>(scoreMetadata.Schema[2]);
            ReadOnlyMemory<char> scoreValueKindValue = null;
            scoreValueKindGetter(ref scoreValueKindValue);
            Assert.Equal(AnnotationUtils.Const.ScoreValueKind.PredictedLabel, scoreValueKindValue.ToString());
        }

        [Fact]
        public void SequencePredictorSchemaWithoutKeyNamesMetadataTest()
        {
            int keyCount = 10;
            var expectedScoreColumnType = new KeyType(typeof(uint), keyCount);
            VBuffer<ReadOnlyMemory<char>> keyNames = GenerateKeyNames(0);

            var sequenceSchema = ScoreSchemaFactory.CreateSequencePredictionSchema(expectedScoreColumnType,
                AnnotationUtils.Const.ScoreColumnKind.SequenceClassification, keyNames);

            // Output schema should only contain one column, which is the predicted label.
            Assert.Single(sequenceSchema);
            var scoreColumn = sequenceSchema[0];

            // Check score column name.
            Assert.Equal(AnnotationUtils.Const.ScoreValueKind.PredictedLabel, scoreColumn.Name);

            // Check score column type.
            var actualScoreColumnType = scoreColumn.Type as KeyType;
            Assert.NotNull(actualScoreColumnType);
            Assert.Equal(expectedScoreColumnType.Count, actualScoreColumnType.Count);
            Assert.Equal(expectedScoreColumnType.RawType, actualScoreColumnType.RawType);

            // Check metadata. Because keyNames is not empty, there should be three metadata fields.
            var scoreMetadata = scoreColumn.Annotations;
            Assert.Equal(2, scoreMetadata.Schema.Count);

            // Check metadata columns' names.
            Assert.Equal(AnnotationUtils.Kinds.ScoreColumnKind, scoreMetadata.Schema[0].Name);
            Assert.Equal(AnnotationUtils.Kinds.ScoreValueKind, scoreMetadata.Schema[1].Name);

            // Check metadata columns' types.
            Assert.Equal(TextDataViewType.Instance, scoreColumn.Annotations.Schema[0].Type);
            Assert.Equal(TextDataViewType.Instance, scoreColumn.Annotations.Schema[1].Type);

            // Check metadata columns' values.
            var scoreColumnKindGetter = scoreMetadata.GetGetter<ReadOnlyMemory<char>>(scoreMetadata.Schema[0]);
            ReadOnlyMemory<char> scoreColumnKindValue = null;
            scoreColumnKindGetter(ref scoreColumnKindValue);
            Assert.Equal(AnnotationUtils.Const.ScoreColumnKind.SequenceClassification, scoreColumnKindValue.ToString());

            var scoreValueKindGetter = scoreMetadata.GetGetter<ReadOnlyMemory<char>>(scoreMetadata.Schema[1]);
            ReadOnlyMemory<char> scoreValueKindValue = null;
            scoreValueKindGetter(ref scoreValueKindValue);
            Assert.Equal(AnnotationUtils.Const.ScoreValueKind.PredictedLabel, scoreValueKindValue.ToString());
        }
    }
}
