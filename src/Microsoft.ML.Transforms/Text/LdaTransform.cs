// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.TextAnalytics;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(LatentDirichletAllocationTransformer.Summary, typeof(IDataTransform), typeof(LatentDirichletAllocationTransformer), typeof(LatentDirichletAllocationTransformer.Options), typeof(SignatureDataTransform),
    "Latent Dirichlet Allocation Transform", LatentDirichletAllocationTransformer.LoaderSignature, "Lda")]

[assembly: LoadableClass(LatentDirichletAllocationTransformer.Summary, typeof(IDataTransform), typeof(LatentDirichletAllocationTransformer), null, typeof(SignatureLoadDataTransform),
    "Latent Dirichlet Allocation Transform", LatentDirichletAllocationTransformer.LoaderSignature)]

[assembly: LoadableClass(LatentDirichletAllocationTransformer.Summary, typeof(LatentDirichletAllocationTransformer), null, typeof(SignatureLoadModel),
    "Latent Dirichlet Allocation Transform", LatentDirichletAllocationTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(LatentDirichletAllocationTransformer), null, typeof(SignatureLoadRowMapper),
    "Latent Dirichlet Allocation Transform", LatentDirichletAllocationTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    // LightLDA transform: Big Topic Models on Modest Compute Clusters.
    // <a href="https://arxiv.org/abs/1412.1576">LightLDA</a> is an implementation of Latent Dirichlet Allocation (LDA).
    // Previous implementations of LDA such as SparseLDA or AliasLDA allow to achieve massive data and model scales,
    // for example models with tens of billions of parameters to be inferred from billions of documents.
    // However this requires using a cluster of thousands of machines with all ensuing costs to setup and maintain.
    // LightLDA solves this problem in a more cost-effective manner by providing an implementation
    // that is efﬁcient enough for modest clusters with at most tens of machines...
    // For more details please see original LightLDA paper:
    // https://arxiv.org/abs/1412.1576
    // http://www.www2015.it/documents/proceedings/proceedings/p1351.pdf
    // and open source implementation:
    // https://github.com/Microsoft/LightLDA
    //
    // See <a href="https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.TestFramework/DataPipe/TestDataPipe.cs"/>
    // for an example on how to use LatentDirichletAllocationTransformer.
    /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
    public sealed class LatentDirichletAllocationTransformer : OneToOneTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:srcs)", Name = "Column", ShortName = "col", SortOrder = 49)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of topics", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "20,40,100,200")]
            [TlcModule.SweepableDiscreteParam("NumTopic", new object[] { 20, 40, 100, 200 })]
            public int NumTopic = LatentDirichletAllocationEstimator.Defaults.NumberOfTopics;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on document-topic vectors")]
            [TGUI(SuggestedSweeps = "1,10,100,200")]
            [TlcModule.SweepableDiscreteParam("AlphaSum", new object[] { 1, 10, 100, 200 })]
            public float AlphaSum = LatentDirichletAllocationEstimator.Defaults.AlphaSum;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on vocab-topic vectors")]
            [TGUI(SuggestedSweeps = "0.01,0.015,0.07,0.02")]
            [TlcModule.SweepableDiscreteParam("Beta", new object[] { 0.01f, 0.015f, 0.07f, 0.02f })]
            public float Beta = LatentDirichletAllocationEstimator.Defaults.Beta;

            [Argument(ArgumentType.Multiple, HelpText = "Number of Metropolis Hasting step")]
            [TGUI(SuggestedSweeps = "2,4,8,16")]
            [TlcModule.SweepableDiscreteParam("Mhstep", new object[] { 2, 4, 8, 16 })]
            public int Mhstep = LatentDirichletAllocationEstimator.Defaults.SamplingStepCount;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations", ShortName = "iter")]
            [TGUI(SuggestedSweeps = "100,200,300,400")]
            [TlcModule.SweepableDiscreteParam("NumIterations", new object[] { 100, 200, 300, 400 })]
            public int NumIterations = LatentDirichletAllocationEstimator.Defaults.MaximumNumberOfIterations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Compute log likelihood over local dataset on this iteration interval", ShortName = "llInterval")]
            public int LikelihoodInterval = LatentDirichletAllocationEstimator.Defaults.LikelihoodInterval;

            // REVIEW: Should change the default when multi-threading support is optimized.
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of training threads. Default value depends on number of logical processors.", ShortName = "t", SortOrder = 50)]
            public int NumThreads = LatentDirichletAllocationEstimator.Defaults.NumberOfThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The threshold of maximum count of tokens per doc", ShortName = "maxNumToken", SortOrder = 50)]
            public int NumMaxDocToken = LatentDirichletAllocationEstimator.Defaults.MaximumTokenCountPerDocument;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of words to summarize the topic", ShortName = "ns")]
            public int NumSummaryTermPerTopic = LatentDirichletAllocationEstimator.Defaults.NumberOfSummaryTermsPerTopic;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of burn-in iterations", ShortName = "burninIter")]
            [TGUI(SuggestedSweeps = "10,20,30,40")]
            [TlcModule.SweepableDiscreteParam("NumBurninIterations", new object[] { 10, 20, 30, 40 })]
            public int NumBurninIterations = LatentDirichletAllocationEstimator.Defaults.NumberOfBurninIterations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Reset the random number generator for each document", ShortName = "reset")]
            public bool ResetRandomGenerator = LatentDirichletAllocationEstimator.Defaults.ResetRandomGenerator;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to output the topic-word summary in text format", ShortName = "summary")]
            public bool OutputTopicWordSummary;
        }

        internal sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of topics")]
            public int? NumTopic;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on document-topic vectors")]
            public float? AlphaSum;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on vocab-topic vectors")]
            public float? Beta;

            [Argument(ArgumentType.Multiple, HelpText = "Number of Metropolis Hasting step")]
            public int? Mhstep;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations", ShortName = "iter")]
            public int? NumIterations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Compute log likelihood over local dataset on this iteration interval", ShortName = "llInterval")]
            public int? LikelihoodInterval;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of training threads", ShortName = "t")]
            public int? NumThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The threshold of maximum count of tokens per doc", ShortName = "maxNumToken")]
            public int? NumMaxDocToken;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of words to summarize the topic", ShortName = "ns")]
            public int? NumSummaryTermPerTopic;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of burn-in iterations", ShortName = "burninIter")]
            public int? NumBurninIterations = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Reset the random number generator for each document", ShortName = "reset")]
            public bool? ResetRandomGenerator;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NumTopic != null || AlphaSum != null || Beta != null || Mhstep != null || NumIterations != null || LikelihoodInterval != null ||
                    NumThreads != null || NumMaxDocToken != null || NumSummaryTermPerTopic != null || ResetRandomGenerator != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        /// <summary>
        /// Provide details about the topics discovered by <a href="https://arxiv.org/abs/1412.1576">LightLDA.</a>
        /// </summary>
        public sealed class ModelParameters
        {
            public struct ItemScore
            {
                public readonly int Item;
                public readonly float Score;
                public ItemScore(int item, float score)
                {
                    Item = item;
                    Score = score;
                }
            }
            public struct WordItemScore
            {
                public readonly int Item;
                public readonly string Word;
                public readonly float Score;
                public WordItemScore(int item, string word, float score)
                {
                    Item = item;
                    Word = word;
                    Score = score;
                }
            }

            // For each topic, provide information about the (item, score) pairs.
            public readonly IReadOnlyList<IReadOnlyList<ItemScore>> ItemScoresPerTopic;

            // For each topic, provide information about the (item, word, score) tuple.
            public readonly IReadOnlyList<IReadOnlyList<WordItemScore>> WordScoresPerTopic;

            internal ModelParameters(IReadOnlyList<IReadOnlyList<ItemScore>> itemScoresPerTopic)
            {
                ItemScoresPerTopic = itemScoresPerTopic;
            }

            internal ModelParameters(IReadOnlyList<IReadOnlyList<WordItemScore>> wordScoresPerTopic)
            {
                WordScoresPerTopic = wordScoresPerTopic;
            }
        }

        [BestFriend]
        internal ModelParameters GetLdaDetails(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < _ldas.Length);

            var ldaState = _ldas[iinfo];
            var mapping = _columnMappings[iinfo];

            return ldaState.GetLdaSummary(mapping);
        }

        private sealed class LdaState : IDisposable
        {
            internal readonly LatentDirichletAllocationEstimator.ColumnOptions InfoEx;
            private readonly int _numVocab;
            private readonly object _preparationSyncRoot;
            private readonly object _testSyncRoot;
            private bool _predictionPreparationDone;
            private LdaSingleBox _ldaTrainer;

            private LdaState()
            {
                _preparationSyncRoot = new object();
                _testSyncRoot = new object();
            }

            internal LdaState(IExceptionContext ectx, LatentDirichletAllocationEstimator.ColumnOptions ex, int numVocab)
                : this()
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(ex, "ex");

                ectx.Assert(numVocab >= 0);
                InfoEx = ex;
                _numVocab = numVocab;

                _ldaTrainer = new LdaSingleBox(
                    InfoEx.NumberOfTopics,
                    numVocab, /* Need to set number of vocabulary here */
                    InfoEx.AlphaSum,
                    InfoEx.Beta,
                    InfoEx.NumberOfIterations,
                    InfoEx.LikelihoodInterval,
                    InfoEx.NumberOfThreads,
                    InfoEx.SamplingStepCount,
                    InfoEx.NumberOfSummaryTermsPerTopic,
                    false,
                    InfoEx.MaximumTokenCountPerDocument);
            }

            internal LdaState(IExceptionContext ectx, ModelLoadContext ctx)
                : this()
            {
                ectx.AssertValue(ctx);

                // *** Binary format ***
                // <ColInfoEx>
                // int: vocabnum
                // long: memblocksize
                // long: aliasMemBlockSize
                // (serializing term by term, for one term)
                // int: term_id, int: topic_num, KeyValuePair<int, int>[]: termTopicVector

                InfoEx = new LatentDirichletAllocationEstimator.ColumnOptions(ectx, ctx);

                _numVocab = ctx.Reader.ReadInt32();
                ectx.CheckDecode(_numVocab > 0);

                long memBlockSize = ctx.Reader.ReadInt64();
                ectx.CheckDecode(memBlockSize > 0);

                long aliasMemBlockSize = ctx.Reader.ReadInt64();
                ectx.CheckDecode(aliasMemBlockSize > 0);

                _ldaTrainer = new LdaSingleBox(
                    InfoEx.NumberOfTopics,
                    _numVocab, /* Need to set number of vocabulary here */
                    InfoEx.AlphaSum,
                    InfoEx.Beta,
                    InfoEx.NumberOfIterations,
                    InfoEx.LikelihoodInterval,
                    InfoEx.NumberOfThreads,
                    InfoEx.SamplingStepCount,
                    InfoEx.NumberOfSummaryTermsPerTopic,
                    false,
                    InfoEx.MaximumTokenCountPerDocument);

                _ldaTrainer.AllocateModelMemory(_numVocab, InfoEx.NumberOfTopics, memBlockSize, aliasMemBlockSize);

                for (int i = 0; i < _numVocab; i++)
                {
                    int termID = ctx.Reader.ReadInt32();
                    ectx.CheckDecode(termID >= 0);
                    int termTopicNum = ctx.Reader.ReadInt32();
                    ectx.CheckDecode(termTopicNum >= 0);

                    int[] topicId = new int[termTopicNum];
                    int[] topicProb = new int[termTopicNum];

                    for (int j = 0; j < termTopicNum; j++)
                    {
                        topicId[j] = ctx.Reader.ReadInt32();
                        topicProb[j] = ctx.Reader.ReadInt32();
                    }

                    //set the topic into _ldaTrainer inner topic table
                    _ldaTrainer.SetModel(termID, topicId, topicProb, termTopicNum);
                }

                //do the preparation
                if (!_predictionPreparationDone)
                {
                    lock (_preparationSyncRoot)
                    {
                        _ldaTrainer.InitializeBeforeTest();
                        _predictionPreparationDone = true;
                    }
                }
            }

            internal ModelParameters GetLdaSummary(VBuffer<ReadOnlyMemory<char>> mapping)
            {
                if (mapping.Length == 0)
                {
                    var itemScoresPerTopicBuilder = ImmutableArray.CreateBuilder<List<ModelParameters.ItemScore>>();
                    for (int i = 0; i < _ldaTrainer.NumTopic; i++)
                    {
                        var scores = _ldaTrainer.GetTopicSummary(i);
                        var itemScores = new List<ModelParameters.ItemScore>();
                        foreach (KeyValuePair<int, float> p in scores)
                        {
                            itemScores.Add(new ModelParameters.ItemScore(p.Key, p.Value));
                        }

                        itemScoresPerTopicBuilder.Add(itemScores);
                    }
                    return new ModelParameters(itemScoresPerTopicBuilder.ToImmutable());
                }
                else
                {
                    ReadOnlyMemory<char> slotName = default;
                    var wordScoresPerTopicBuilder = ImmutableArray.CreateBuilder<List<ModelParameters.WordItemScore>>();
                    for (int i = 0; i < _ldaTrainer.NumTopic; i++)
                    {
                        var scores = _ldaTrainer.GetTopicSummary(i);
                        var wordScores = new List<ModelParameters.WordItemScore>();
                        foreach (KeyValuePair<int, float> p in scores)
                        {
                            mapping.GetItemOrDefault(p.Key, ref slotName);
                            wordScores.Add(new ModelParameters.WordItemScore(p.Key, slotName.ToString(), p.Value));
                        }
                        wordScoresPerTopicBuilder.Add(wordScores);
                    }
                    return new ModelParameters(wordScoresPerTopicBuilder.ToImmutable());
                }
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);
                long memBlockSize = 0;
                long aliasMemBlockSize = 0;
                _ldaTrainer.GetModelStat(out memBlockSize, out aliasMemBlockSize);

                // *** Binary format ***
                // <ColInfoEx>
                // int: vocabnum
                // long: memblocksize
                // long: aliasMemBlockSize
                // (serializing term by term, for one term)
                // int: term_id, int: topic_num, KeyValuePair<int, int>[]: termTopicVector

                InfoEx.Save(ctx);
                ctx.Writer.Write(_ldaTrainer.NumVocab);
                ctx.Writer.Write(memBlockSize);
                ctx.Writer.Write(aliasMemBlockSize);

                //save model from this interface
                for (int i = 0; i < _ldaTrainer.NumVocab; i++)
                {
                    KeyValuePair<int, int>[] termTopicVector = _ldaTrainer.GetModel(i);

                    //write the topic to disk through ctx
                    ctx.Writer.Write(i); //term_id
                    ctx.Writer.Write(termTopicVector.Length);

                    foreach (KeyValuePair<int, int> p in termTopicVector)
                    {
                        ctx.Writer.Write(p.Key);
                        ctx.Writer.Write(p.Value);
                    }
                }
            }

            public void AllocateDataMemory(int docNum, long corpusSize)
            {
                _ldaTrainer.AllocateDataMemory(docNum, corpusSize);
            }

            public int FeedTrain(IExceptionContext ectx, in VBuffer<Double> input)
            {
                Contracts.AssertValue(ectx);

                // REVIEW: Input the counts to your trainer here. This
                // is called multiple times.

                int docSize = 0;
                int termNum = 0;

                var inputValues = input.GetValues();
                for (int i = 0; i < inputValues.Length; i++)
                {
                    int termFreq = GetFrequency(inputValues[i]);
                    if (termFreq < 0)
                    {
                        // Ignore this row.
                        return 0;
                    }
                    if (docSize >= InfoEx.MaximumTokenCountPerDocument - termFreq)
                        break;

                    // If legal then add the term.
                    docSize += termFreq;
                    termNum++;
                }

                // Ignore empty doc.
                if (docSize == 0)
                    return 0;

                int actualSize = 0;
                if (input.IsDense)
                    actualSize = _ldaTrainer.LoadDocDense(inputValues, termNum, input.Length);
                else
                    actualSize = _ldaTrainer.LoadDoc(input.GetIndices(), inputValues, termNum, input.Length);

                ectx.Assert(actualSize == 2 * docSize + 1, string.Format("The doc size are distinct. Actual: {0}, Expected: {1}", actualSize, 2 * docSize + 1));
                return actualSize;
            }

            public void CompleteTrain()
            {
                //allocate all kinds of in memory sample tables
                _ldaTrainer.InitializeBeforeTrain();

                //call native lda trainer to perform the multi-thread training
                _ldaTrainer.Train(""); /* Need to pass in an empty string */
            }

            public void Output(in VBuffer<Double> src, ref VBuffer<float> dst, int numBurninIter, bool reset)
            {
                // Prediction for a single document.
                // LdaSingleBox.InitializeBeforeTest() is NOT thread-safe.
                if (!_predictionPreparationDone)
                {
                    lock (_preparationSyncRoot)
                    {
                        if (!_predictionPreparationDone)
                        {
                            //do some preparation for building tables in native c++
                            _ldaTrainer.InitializeBeforeTest();
                            _predictionPreparationDone = true;
                        }
                    }
                }

                int len = InfoEx.NumberOfTopics;
                var srcValues = src.GetValues();
                if (srcValues.Length == 0)
                {
                    VBufferUtils.Resize(ref dst, len, 0);
                    return;
                }

                VBufferEditor<float> editor;
                // Make sure all the frequencies are valid and truncate if the sum gets too large.
                int docSize = 0;
                int termNum = 0;
                for (int i = 0; i < srcValues.Length; i++)
                {
                    int termFreq = GetFrequency(srcValues[i]);
                    if (termFreq < 0)
                    {
                        // REVIEW: Should this log a warning message? And what should it produce?
                        // It currently produces a vbuffer of all NA values.
                        // REVIEW: Need a utility method to do this...
                        editor = VBufferEditor.Create(ref dst, len);

                        for (int k = 0; k < len; k++)
                            editor.Values[k] = float.NaN;
                        dst = editor.Commit();
                        return;
                    }

                    if (docSize >= InfoEx.MaximumTokenCountPerDocument - termFreq)
                        break;

                    docSize += termFreq;
                    termNum++;
                }

                // REVIEW: Too much memory allocation here on each prediction.
                List<KeyValuePair<int, float>> retTopics;
                if (src.IsDense)
                    retTopics = _ldaTrainer.TestDocDense(srcValues, termNum, numBurninIter, reset);
                else
                    retTopics = _ldaTrainer.TestDoc(src.GetIndices(), srcValues, termNum, numBurninIter, reset);

                int count = retTopics.Count;
                Contracts.Assert(count <= len);

                editor = VBufferEditor.Create(ref dst, len, count);
                double normalizer = 0;
                for (int i = 0; i < count; i++)
                {
                    int index = retTopics[i].Key;
                    float value = retTopics[i].Value;
                    Contracts.Assert(value >= 0);
                    Contracts.Assert(0 <= index && index < len);
                    if (count < len)
                    {
                        Contracts.Assert(i == 0 || editor.Indices[i - 1] < index);
                        editor.Indices[i] = index;
                    }
                    else
                        Contracts.Assert(index == i);

                    editor.Values[i] = value;
                    normalizer += value;
                }

                if (normalizer > 0)
                {
                    for (int i = 0; i < count; i++)
                        editor.Values[i] = (float)(editor.Values[i] / normalizer);
                }

                dst = editor.Commit();
            }

            public void Dispose()
            {
                _ldaTrainer.Dispose();
            }
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly LatentDirichletAllocationTransformer _parent;
            private readonly int[] _srcCols;

            public Mapper(LatentDirichletAllocationTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _srcCols = new int[_parent.ColumnPairs.Length];

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _srcCols[i]))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);

                    var srcCol = inputSchema[_srcCols[i]];
                    var srcType = srcCol.Type as VectorType;
                    if (srcType == null || !srcType.IsKnownSize || !(srcType.ItemType is NumberDataViewType))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName, "known-size vector of float", srcCol.Type.ToString());
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var info = _parent._columns[i];
                    result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, new VectorType(NumberDataViewType.Single, info.NumberOfTopics), null);
                }
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                return GetTopic(input, iinfo);
            }

            private ValueGetter<VBuffer<float>> GetTopic(DataViewRow input, int iinfo)
            {
                var getSrc = RowCursorUtils.GetVecGetterAs<Double>(NumberDataViewType.Double, input, _srcCols[iinfo]);
                var src = default(VBuffer<Double>);
                var lda = _parent._ldas[iinfo];
                int numBurninIter = lda.InfoEx.NumberOfBurninIterations;
                bool reset = lda.InfoEx.ResetRandomGenerator;
                return
                    (ref VBuffer<float> dst) =>
                    {
                        // REVIEW: This will work, but there are opportunities for caching
                        // based on input.Counter that are probably worthwhile given how long inference takes.
                        getSrc(ref src);
                        lda.Output(in src, ref dst, numBurninIter, reset);
                    };
            }
        }

        internal const string LoaderSignature = "LdaTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LIGHTLDA",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LatentDirichletAllocationTransformer).Assembly.FullName);
        }

        private readonly LatentDirichletAllocationEstimator.ColumnOptions[] _columns;
        private readonly LdaState[] _ldas;
        private readonly List<VBuffer<ReadOnlyMemory<char>>> _columnMappings;

        private const string RegistrationName = "LightLda";
        private const string WordTopicModelFilename = "word_topic_summary.txt";
        internal const string Summary = "The LDA transform implements LightLDA, a state-of-the-art implementation of Latent Dirichlet Allocation.";
        internal const string UserName = "Latent Dirichlet Allocation Transform";
        internal const string ShortName = "LightLda";

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(LatentDirichletAllocationEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        /// <summary>
        /// Initializes a new <see cref="LatentDirichletAllocationTransformer"/> object.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="ldas">An array of LdaState objects, where ldas[i] is learnt from the i-th element of <paramref name="columns"/>.</param>
        /// <param name="columnMappings">A list of mappings, where columnMapping[i] is a map of slot names for the i-th element of <paramref name="columns"/>.</param>
        /// <param name="columns">Describes the parameters of the LDA process for each column pair.</param>
        private LatentDirichletAllocationTransformer(IHostEnvironment env,
            LdaState[] ldas,
            List<VBuffer<ReadOnlyMemory<char>>> columnMappings,
            params LatentDirichletAllocationEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LatentDirichletAllocationTransformer)), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(ColumnPairs);
            _ldas = ldas;
            _columnMappings = columnMappings;
            _columns = columns;
        }

        private LatentDirichletAllocationTransformer(IHost host, ModelLoadContext ctx) : base(host, ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // ldaState[num infos]: The LDA parameters

            // Note: columnsLength would be just one in most cases.
            var columnsLength = ColumnPairs.Length;
            _columns = new LatentDirichletAllocationEstimator.ColumnOptions[columnsLength];
            _ldas = new LdaState[columnsLength];
            for (int i = 0; i < _ldas.Length; i++)
            {
                _ldas[i] = new LdaState(Host, ctx);
                _columns[i] = _ldas[i].InfoEx;
            }
        }

        internal static LatentDirichletAllocationTransformer TrainLdaTransformer(IHostEnvironment env, IDataView inputData, params LatentDirichletAllocationEstimator.ColumnOptions[] columns)
        {
            var ldas = new LdaState[columns.Length];

            List<VBuffer<ReadOnlyMemory<char>>> columnMappings;
            using (var ch = env.Start("Train"))
            {
                columnMappings = Train(env, ch, inputData, ldas, columns);
            }

            return new LatentDirichletAllocationTransformer(env, ldas, columnMappings, columns);
        }

        private void Dispose(bool disposing)
        {
            if (_ldas != null)
            {
                foreach (var state in _ldas)
                    state?.Dispose();
            }
            if (disposing)
                GC.SuppressFinalize(this);
        }

        public void Dispose()
        {
            Dispose(true);
        }

        ~LatentDirichletAllocationTransformer()
        {
            Dispose(false);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        // Factory method for SignatureDataTransform.
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.Columns, nameof(options.Columns));

            var cols = options.Columns.Select(colPair => new LatentDirichletAllocationEstimator.ColumnOptions(colPair, options)).ToArray();
            return TrainLdaTransformer(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel
        private static LatentDirichletAllocationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply(
                "Loading Model",
                ch =>
                {
                    // *** Binary Format ***
                    // int: sizeof(float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    h.CheckDecode(cbFloat == sizeof(float));
                    return new LatentDirichletAllocationTransformer(h, ctx);
                });
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(float)
            // <base>
            // ldaState[num infos]: The LDA parameters

            ctx.Writer.Write(sizeof(float));
            SaveColumns(ctx);
            for (int i = 0; i < _ldas.Length; i++)
            {
                _ldas[i].Save(ctx);
            }
        }

        private static int GetFrequency(double value)
        {
            int result = (int)value;
            if (!(result == value && result >= 0))
                return -1;
            return result;
        }

        private static List<VBuffer<ReadOnlyMemory<char>>> Train(IHostEnvironment env, IChannel ch, IDataView inputData, LdaState[] states, params LatentDirichletAllocationEstimator.ColumnOptions[] columns)
        {
            env.AssertValue(ch);
            ch.AssertValue(inputData);
            ch.AssertValue(states);
            ch.Assert(states.Length == columns.Length);

            var activeColumns = new List<DataViewSchema.Column>();
            int[] numVocabs = new int[columns.Length];
            int[] srcCols = new int[columns.Length];

            var columnMappings = new List<VBuffer<ReadOnlyMemory<char>>>();

            var inputSchema = inputData.Schema;
            for (int i = 0; i < columns.Length; i++)
            {
                if (!inputData.Schema.TryGetColumnIndex(columns[i].InputColumnName, out int srcCol))
                    throw env.ExceptSchemaMismatch(nameof(inputData), "input", columns[i].InputColumnName);

                var srcColType = inputSchema[srcCol].Type as VectorType;
                if (srcColType == null || !srcColType.IsKnownSize || !(srcColType.ItemType is NumberDataViewType))
                    throw env.ExceptSchemaMismatch(nameof(inputSchema), "input", columns[i].InputColumnName, "known-size vector of float", srcColType.ToString());

                srcCols[i] = srcCol;
                activeColumns.Add(inputData.Schema[srcCol]);
                numVocabs[i] = 0;

                VBuffer<ReadOnlyMemory<char>> dst = default;
                if (inputSchema[srcCol].HasSlotNames(srcColType.Size))
                    inputSchema[srcCol].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref dst);
                else
                    dst = default(VBuffer<ReadOnlyMemory<char>>);
                columnMappings.Add(dst);
            }

            //the current lda needs the memory allocation before feedin data, so needs two sweeping of the data,
            //one for the pre-calc memory, one for feedin data really
            //another solution can be prepare these two value externally and put them in the beginning of the input file.
            long[] corpusSize = new long[columns.Length];
            int[] numDocArray = new int[columns.Length];

            using (var cursor = inputData.GetRowCursor(activeColumns))
            {
                var getters = new ValueGetter<VBuffer<Double>>[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    corpusSize[i] = 0;
                    numDocArray[i] = 0;
                    getters[i] = RowCursorUtils.GetVecGetterAs<Double>(NumberDataViewType.Double, cursor, srcCols[i]);
                }
                VBuffer<Double> src = default;
                long rowCount = 0;
                while (cursor.MoveNext())
                {
                    ++rowCount;
                    for (int i = 0; i < columns.Length; i++)
                    {
                        int docSize = 0;
                        getters[i](ref src);

                        // compute term, doc instance#.
                        var srcValues = src.GetValues();
                        for (int termID = 0; termID < srcValues.Length; termID++)
                        {
                            int termFreq = GetFrequency(srcValues[termID]);
                            if (termFreq < 0)
                            {
                                // Ignore this row.
                                docSize = 0;
                                break;
                            }

                            if (docSize >= columns[i].MaximumTokenCountPerDocument - termFreq)
                                break; //control the document length

                            //if legal then add the term
                            docSize += termFreq;
                        }

                        // Ignore empty doc
                        if (docSize == 0)
                            continue;

                        numDocArray[i]++;
                        corpusSize[i] += docSize * 2 + 1;   // in the beggining of each doc, there is a cursor variable

                        // increase numVocab if needed.
                        if (numVocabs[i] < src.Length)
                            numVocabs[i] = src.Length;
                    }
                }

                // No data to train on, just return
                if (rowCount == 0)
                    return columnMappings;

                for (int i = 0; i < columns.Length; ++i)
                {
                    if (numDocArray[i] != rowCount)
                    {
                        ch.Assert(numDocArray[i] < rowCount);
                        ch.Warning($"Column '{columns[i].InputColumnName}' has skipped {rowCount - numDocArray[i]} of {rowCount} rows either empty or with negative, non-finite, or fractional values.");
                    }
                }
            }

            // Initialize all LDA states
            for (int i = 0; i < columns.Length; i++)
            {
                var state = new LdaState(env, columns[i], numVocabs[i]);

                if (numDocArray[i] == 0 || corpusSize[i] == 0)
                    throw ch.Except("The specified documents are all empty in column '{0}'.", columns[i].InputColumnName);

                state.AllocateDataMemory(numDocArray[i], corpusSize[i]);
                states[i] = state;
            }

            using (var cursor = inputData.GetRowCursor(activeColumns))
            {
                int[] docSizeCheck = new int[columns.Length];
                // This could be optimized so that if multiple trainers consume the same column, it is
                // fed into the train method once.
                var getters = new ValueGetter<VBuffer<Double>>[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    docSizeCheck[i] = 0;
                    getters[i] = RowCursorUtils.GetVecGetterAs<Double>(NumberDataViewType.Double, cursor, srcCols[i]);
                }

                VBuffer<double> src = default;

                while (cursor.MoveNext())
                {
                    for (int i = 0; i < columns.Length; i++)
                    {
                        getters[i](ref src);
                        docSizeCheck[i] += states[i].FeedTrain(env, in src);
                    }
                }

                for (int i = 0; i < columns.Length; i++)
                {
                    env.Assert(corpusSize[i] == docSizeCheck[i]);
                    states[i].CompleteTrain();
                }
            }

            return columnMappings;
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
            => new Mapper(this, schema);
    }

    /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
    public sealed class LatentDirichletAllocationEstimator : IEstimator<LatentDirichletAllocationTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const int NumberOfTopics = 100;
            public const float AlphaSum = 100;
            public const float Beta = 0.01f;
            public const int SamplingStepCount = 4;
            public const int MaximumNumberOfIterations = 200;
            public const int LikelihoodInterval = 5;
            public const int NumberOfThreads = 0;
            public const int MaximumTokenCountPerDocument = 512;
            public const int NumberOfSummaryTermsPerTopic = 10;
            public const int NumberOfBurninIterations = 10;
            public const bool ResetRandomGenerator = false;
        }

        private readonly IHost _host;
        private readonly ImmutableArray<ColumnOptions> _columns;

        /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="numberOfTopics">The number of topics.</param>
        /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
        /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
        /// <param name="samplingStepCount">Number of Metropolis Hasting step.</param>
        /// <param name="maximumNumberOfIterations">Number of iterations.</param>
        /// <param name="numberOfThreads">The number of training threads. Default value depends on number of logical processors.</param>
        /// <param name="maximumTokenCountPerDocument">The threshold of maximum count of tokens per doc.</param>
        /// <param name="numberOfSummaryTermsPerTopic">The number of words to summarize the topic.</param>
        /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
        /// <param name="numberOfBurninIterations">The number of burn-in iterations.</param>
        /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
        internal LatentDirichletAllocationEstimator(IHostEnvironment env,
            string outputColumnName, string inputColumnName = null,
            int numberOfTopics = Defaults.NumberOfTopics,
            float alphaSum = Defaults.AlphaSum,
            float beta = Defaults.Beta,
            int samplingStepCount = Defaults.SamplingStepCount,
            int maximumNumberOfIterations = Defaults.MaximumNumberOfIterations,
            int numberOfThreads = Defaults.NumberOfThreads,
            int maximumTokenCountPerDocument = Defaults.MaximumTokenCountPerDocument,
            int numberOfSummaryTermsPerTopic = Defaults.NumberOfSummaryTermsPerTopic,
            int likelihoodInterval = Defaults.LikelihoodInterval,
            int numberOfBurninIterations = Defaults.NumberOfBurninIterations,
            bool resetRandomGenerator = Defaults.ResetRandomGenerator)
            : this(env, new[] { new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName,
                numberOfTopics, alphaSum, beta, samplingStepCount, maximumNumberOfIterations, likelihoodInterval, numberOfThreads, maximumTokenCountPerDocument,
                numberOfSummaryTermsPerTopic, numberOfBurninIterations, resetRandomGenerator) })
        { }

        /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="columns">Describes the parameters of the LDA process for each column pair.</param>
        internal LatentDirichletAllocationEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(LatentDirichletAllocationEstimator));
            _columns = columns.ToImmutableArray();
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>
            /// Name of the column resulting from the transformation of <cref see="InputColumnName"/>.
            /// </summary>
            public readonly string Name;
            /// <summary>
            /// Name of column to transform.
            /// </summary>
            public readonly string InputColumnName;
            /// <summary>
            /// The number of topics.
            /// </summary>
            public readonly int NumberOfTopics;
            /// <summary>
            /// Dirichlet prior on document-topic vectors.
            /// </summary>
            public readonly float AlphaSum;
            /// <summary>
            /// Dirichlet prior on vocab-topic vectors.
            /// </summary>
            public readonly float Beta;
            /// <summary>
            /// Number of Metropolis Hasting step.
            /// </summary>
            public readonly int SamplingStepCount;
            /// <summary>
            /// Number of iterations.
            /// </summary>
            public readonly int NumberOfIterations;
            /// <summary>
            /// Compute log likelihood over local dataset on this iteration interval.
            /// </summary>
            public readonly int LikelihoodInterval;
            /// <summary>
            /// The number of training threads.
            /// </summary>
            public readonly int NumberOfThreads;
            /// <summary>
            /// The threshold of maximum count of tokens per doc.
            /// </summary>
            public readonly int MaximumTokenCountPerDocument;
            /// <summary>
            /// The number of words to summarize the topic.
            /// </summary>
            public readonly int NumberOfSummaryTermsPerTopic;
            /// <summary>
            /// The number of burn-in iterations.
            /// </summary>
            public readonly int NumberOfBurninIterations;
            /// <summary>
            /// Reset the random number generator for each document.
            /// </summary>
            public readonly bool ResetRandomGenerator;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">The column containing the output scores over a set of topics, represented as a vector of floats. </param>
            /// <param name="inputColumnName">The column representing the document as a vector of floats.A null value for the column means <paramref name="inputColumnName"/> is replaced. </param>
            /// <param name="numberOfTopics">The number of topics.</param>
            /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
            /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
            /// <param name="samplingStepCount">Number of Metropolis Hasting step.</param>
            /// <param name="maximumNumberOfIterations">Number of iterations.</param>
            /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
            /// <param name="numberOfThreads">The number of training threads. Default value depends on number of logical processors.</param>
            /// <param name="maximumTokenCountPerDocument">The threshold of maximum count of tokens per doc.</param>
            /// <param name="numberOfSummaryTermsPerTopic">The number of words to summarize the topic.</param>
            /// <param name="numberOfBurninIterations">The number of burn-in iterations.</param>
            /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
            public ColumnOptions(string name,
                string inputColumnName = null,
                int numberOfTopics = LatentDirichletAllocationEstimator.Defaults.NumberOfTopics,
                float alphaSum = LatentDirichletAllocationEstimator.Defaults.AlphaSum,
                float beta = LatentDirichletAllocationEstimator.Defaults.Beta,
                int samplingStepCount = LatentDirichletAllocationEstimator.Defaults.SamplingStepCount,
                int maximumNumberOfIterations = LatentDirichletAllocationEstimator.Defaults.MaximumNumberOfIterations,
                int likelihoodInterval = LatentDirichletAllocationEstimator.Defaults.LikelihoodInterval,
                int numberOfThreads = LatentDirichletAllocationEstimator.Defaults.NumberOfThreads,
                int maximumTokenCountPerDocument = LatentDirichletAllocationEstimator.Defaults.MaximumTokenCountPerDocument,
                int numberOfSummaryTermsPerTopic = LatentDirichletAllocationEstimator.Defaults.NumberOfSummaryTermsPerTopic,
                int numberOfBurninIterations = LatentDirichletAllocationEstimator.Defaults.NumberOfBurninIterations,
                bool resetRandomGenerator = LatentDirichletAllocationEstimator.Defaults.ResetRandomGenerator)
            {
                Contracts.CheckValue(name, nameof(name));
                Contracts.CheckValueOrNull(inputColumnName);
                Contracts.CheckParam(numberOfTopics > 0, nameof(numberOfTopics), "Must be positive.");
                Contracts.CheckParam(samplingStepCount > 0, nameof(samplingStepCount), "Must be positive.");
                Contracts.CheckParam(maximumNumberOfIterations > 0, nameof(maximumNumberOfIterations), "Must be positive.");
                Contracts.CheckParam(likelihoodInterval > 0, nameof(likelihoodInterval), "Must be positive.");
                Contracts.CheckParam(numberOfThreads >= 0, nameof(numberOfThreads), "Must be positive or zero.");
                Contracts.CheckParam(maximumTokenCountPerDocument > 0, nameof(maximumTokenCountPerDocument), "Must be positive.");
                Contracts.CheckParam(numberOfSummaryTermsPerTopic > 0, nameof(numberOfSummaryTermsPerTopic), "Must be positive");
                Contracts.CheckParam(numberOfBurninIterations >= 0, nameof(numberOfBurninIterations), "Must be non-negative.");

                Name = name;
                InputColumnName = inputColumnName ?? name;
                NumberOfTopics = numberOfTopics;
                AlphaSum = alphaSum;
                Beta = beta;
                SamplingStepCount = samplingStepCount;
                NumberOfIterations = maximumNumberOfIterations;
                LikelihoodInterval = likelihoodInterval;
                NumberOfThreads = numberOfThreads;
                MaximumTokenCountPerDocument = maximumTokenCountPerDocument;
                NumberOfSummaryTermsPerTopic = numberOfSummaryTermsPerTopic;
                NumberOfBurninIterations = numberOfBurninIterations;
                ResetRandomGenerator = resetRandomGenerator;
            }

            internal ColumnOptions(LatentDirichletAllocationTransformer.Column item, LatentDirichletAllocationTransformer.Options options) :
                this(item.Name,
                    item.Source ?? item.Name,
                    item.NumTopic ?? options.NumTopic,
                    item.AlphaSum ?? options.AlphaSum,
                    item.Beta ?? options.Beta,
                    item.Mhstep ?? options.Mhstep,
                    item.NumIterations ?? options.NumIterations,
                    item.LikelihoodInterval ?? options.LikelihoodInterval,
                    item.NumThreads ?? options.NumThreads,
                    item.NumMaxDocToken ?? options.NumMaxDocToken,
                    item.NumSummaryTermPerTopic ?? options.NumSummaryTermPerTopic,
                    item.NumBurninIterations ?? options.NumBurninIterations,
                    item.ResetRandomGenerator ?? options.ResetRandomGenerator)
            {
            }

            internal ColumnOptions(IExceptionContext ectx, ModelLoadContext ctx)
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(ctx);

                // *** Binary format ***
                // int NumTopic;
                // float AlphaSum;
                // float Beta;
                // int MHStep;
                // int NumIter;
                // int LikelihoodInterval;
                // int NumThread;
                // int NumMaxDocToken;
                // int NumSummaryTermPerTopic;
                // int NumBurninIter;
                // byte ResetRandomGenerator;

                NumberOfTopics = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumberOfTopics > 0);

                AlphaSum = ctx.Reader.ReadSingle();

                Beta = ctx.Reader.ReadSingle();

                SamplingStepCount = ctx.Reader.ReadInt32();
                ectx.CheckDecode(SamplingStepCount > 0);

                NumberOfIterations = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumberOfIterations > 0);

                LikelihoodInterval = ctx.Reader.ReadInt32();
                ectx.CheckDecode(LikelihoodInterval > 0);

                NumberOfThreads = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumberOfThreads >= 0);

                MaximumTokenCountPerDocument = ctx.Reader.ReadInt32();
                ectx.CheckDecode(MaximumTokenCountPerDocument > 0);

                NumberOfSummaryTermsPerTopic = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumberOfSummaryTermsPerTopic > 0);

                NumberOfBurninIterations = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumberOfBurninIterations >= 0);

                ResetRandomGenerator = ctx.Reader.ReadBoolByte();
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int NumTopic;
                // float AlphaSum;
                // float Beta;
                // int MHStep;
                // int NumIter;
                // int LikelihoodInterval;
                // int NumThread;
                // int NumMaxDocToken;
                // int NumSummaryTermPerTopic;
                // int NumBurninIter;
                // byte ResetRandomGenerator;

                ctx.Writer.Write(NumberOfTopics);
                ctx.Writer.Write(AlphaSum);
                ctx.Writer.Write(Beta);
                ctx.Writer.Write(SamplingStepCount);
                ctx.Writer.Write(NumberOfIterations);
                ctx.Writer.Write(LikelihoodInterval);
                ctx.Writer.Write(NumberOfThreads);
                ctx.Writer.Write(MaximumTokenCountPerDocument);
                ctx.Writer.Write(NumberOfSummaryTermsPerTopic);
                ctx.Writer.Write(NumberOfBurninIterations);
                ctx.Writer.WriteBoolByte(ResetRandomGenerator);
            }
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (col.ItemType.RawType != typeof(float) || col.Kind == SchemaShape.Column.VectorKind.Scalar)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, "vector of float", col.GetTypeString());

                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);
            }

            return new SchemaShape(result.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="LatentDirichletAllocationTransformer"/>.
        /// </summary>
        public LatentDirichletAllocationTransformer Fit(IDataView input)
        {
            return LatentDirichletAllocationTransformer.TrainLdaTransformer(_host, input, _columns.ToArray());
        }
    }
}
