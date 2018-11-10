// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TextAnalytics;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Float = System.Single;

[assembly: LoadableClass(LdaTransformer.Summary, typeof(IDataTransform), typeof(LdaTransformer), typeof(LdaTransformer.Arguments), typeof(SignatureDataTransform),
    "Latent Dirichlet Allocation Transform", LdaTransformer.LoaderSignature, "Lda")]

[assembly: LoadableClass(LdaTransformer.Summary, typeof(IDataTransform), typeof(LdaTransformer), null, typeof(SignatureLoadDataTransform),
    "Latent Dirichlet Allocation Transform", LdaTransformer.LoaderSignature)]

[assembly: LoadableClass(LdaTransformer.Summary, typeof(LdaTransformer), null, typeof(SignatureLoadModel),
    "Latent Dirichlet Allocation Transform", LdaTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(LdaTransformer), null, typeof(SignatureLoadRowMapper),
    "Latent Dirichlet Allocation Transform", LdaTransformer.LoaderSignature)]

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
    // for an example on how to use LdaTransformer.
    /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
    public sealed class LdaTransformer : OneToOneTransformerBase
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:srcs)", ShortName = "col", SortOrder = 49)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of topics in the LDA", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "20,40,100,200")]
            [TlcModule.SweepableDiscreteParam("NumTopic", new object[] { 20, 40, 100, 200 })]
            public int NumTopic = LdaEstimator.Defaults.NumTopic;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on document-topic vectors")]
            [TGUI(SuggestedSweeps = "1,10,100,200")]
            [TlcModule.SweepableDiscreteParam("AlphaSum", new object[] { 1, 10, 100, 200 })]
            public Single AlphaSum = LdaEstimator.Defaults.AlphaSum;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on vocab-topic vectors")]
            [TGUI(SuggestedSweeps = "0.01,0.015,0.07,0.02")]
            [TlcModule.SweepableDiscreteParam("Beta", new object[] { 0.01f, 0.015f, 0.07f, 0.02f })]
            public Single Beta = LdaEstimator.Defaults.Beta;

            [Argument(ArgumentType.Multiple, HelpText = "Number of Metropolis Hasting step")]
            [TGUI(SuggestedSweeps = "2,4,8,16")]
            [TlcModule.SweepableDiscreteParam("Mhstep", new object[] { 2, 4, 8, 16 })]
            public int Mhstep = LdaEstimator.Defaults.Mhstep;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations", ShortName = "iter")]
            [TGUI(SuggestedSweeps = "100,200,300,400")]
            [TlcModule.SweepableDiscreteParam("NumIterations", new object[] { 100, 200, 300, 400 })]
            public int NumIterations = LdaEstimator.Defaults.NumIterations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Compute log likelihood over local dataset on this iteration interval", ShortName = "llInterval")]
            public int LikelihoodInterval = LdaEstimator.Defaults.LikelihoodInterval;

            // REVIEW: Should change the default when multi-threading support is optimized.
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of training threads. Default value depends on number of logical processors.", ShortName = "t", SortOrder = 50)]
            public int? NumThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The threshold of maximum count of tokens per doc", ShortName = "maxNumToken", SortOrder = 50)]
            public int NumMaxDocToken = LdaEstimator.Defaults.NumMaxDocToken;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of words to summarize the topic", ShortName = "ns")]
            public int NumSummaryTermPerTopic = LdaEstimator.Defaults.NumSummaryTermPerTopic;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of burn-in iterations", ShortName = "burninIter")]
            [TGUI(SuggestedSweeps = "10,20,30,40")]
            [TlcModule.SweepableDiscreteParam("NumBurninIterations", new object[] { 10, 20, 30, 40 })]
            public int NumBurninIterations = LdaEstimator.Defaults.NumBurninIterations;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Reset the random number generator for each document", ShortName = "reset")]
            public bool ResetRandomGenerator = LdaEstimator.Defaults.ResetRandomGenerator;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to output the topic-word summary in text format", ShortName = "summary")]
            public bool OutputTopicWordSummary;
        }

        public sealed class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of topics in the LDA")]
            public int? NumTopic;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on document-topic vectors")]
            public Single? AlphaSum;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on vocab-topic vectors")]
            public Single? Beta;

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

            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (NumTopic != null || AlphaSum != null || Beta != null || Mhstep != null || NumIterations != null || LikelihoodInterval != null ||
                    NumThreads != null || NumMaxDocToken != null || NumSummaryTermPerTopic != null || ResetRandomGenerator != null)
                    return false;
                return TryUnparseCore(sb);
            }
        }

        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly int NumTopic;
            public readonly Single AlphaSum;
            public readonly Single Beta;
            public readonly int MHStep;
            public readonly int NumIter;
            public readonly int LikelihoodInterval;
            public readonly int NumThread;
            public readonly int NumMaxDocToken;
            public readonly int NumSummaryTermPerTopic;
            public readonly int NumBurninIter;
            public readonly bool ResetRandomGenerator;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of output column.</param>
            /// <param name="numTopic">The number of topics in the LDA.</param>
            /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
            /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
            /// <param name="mhStep">Number of Metropolis Hasting step.</param>
            /// <param name="numIter">Number of iterations.</param>
            /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
            /// <param name="numThread">The number of training threads. Default value depends on number of logical processors.</param>
            /// <param name="numMaxDocToken">The threshold of maximum count of tokens per doc.</param>
            /// <param name="numSummaryTermPerTopic">The number of words to summarize the topic.</param>
            /// <param name="numBurninIter">The number of burn-in iterations.</param>
            /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
            public ColumnInfo(string input, string output, int numTopic, Single alphaSum, Single beta, int mhStep, int numIter, int likelihoodInterval,
                int numThread, int numMaxDocToken, int numSummaryTermPerTopic, int numBurninIter, bool resetRandomGenerator)
            {
                Input = input;
                Output = output;
                NumTopic = numTopic;
                AlphaSum = alphaSum;
                Beta = beta;
                MHStep = mhStep;
                NumIter = numIter;
                LikelihoodInterval = likelihoodInterval;
                NumThread = numThread;
                NumMaxDocToken = numMaxDocToken;
                NumSummaryTermPerTopic = numSummaryTermPerTopic;
                NumBurninIter = numBurninIter;
                ResetRandomGenerator = resetRandomGenerator;
            }

            internal ColumnInfo(IExceptionContext ectx, ModelLoadContext ctx)
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(ctx);

                // *** Binary format ***
                // int NumTopic;
                // Single AlphaSum;
                // Single Beta;
                // int MHStep;
                // int NumIter;
                // int LikelihoodInterval;
                // int NumThread;
                // int NumMaxDocToken;
                // int NumSummaryTermPerTopic;
                // int NumBurninIter;
                // byte ResetRandomGenerator;

                NumTopic = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumTopic > 0);

                AlphaSum = ctx.Reader.ReadSingle();

                Beta = ctx.Reader.ReadSingle();

                MHStep = ctx.Reader.ReadInt32();
                ectx.CheckDecode(MHStep > 0);

                NumIter = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumIter > 0);

                LikelihoodInterval = ctx.Reader.ReadInt32();
                ectx.CheckDecode(LikelihoodInterval > 0);

                NumThread = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumThread >= 0);

                NumMaxDocToken = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumMaxDocToken > 0);

                NumSummaryTermPerTopic = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumSummaryTermPerTopic > 0);

                NumBurninIter = ctx.Reader.ReadInt32();
                ectx.CheckDecode(NumBurninIter >= 0);

                ResetRandomGenerator = ctx.Reader.ReadBoolByte();
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int NumTopic;
                // Single AlphaSum;
                // Single Beta;
                // int MHStep;
                // int NumIter;
                // int LikelihoodInterval;
                // int NumThread;
                // int NumMaxDocToken;
                // int NumSummaryTermPerTopic;
                // int NumBurninIter;
                // byte ResetRandomGenerator;

                ctx.Writer.Write(NumTopic);
                ctx.Writer.Write(AlphaSum);
                ctx.Writer.Write(Beta);
                ctx.Writer.Write(MHStep);
                ctx.Writer.Write(NumIter);
                ctx.Writer.Write(LikelihoodInterval);
                ctx.Writer.Write(NumThread);
                ctx.Writer.Write(NumMaxDocToken);
                ctx.Writer.Write(NumSummaryTermPerTopic);
                ctx.Writer.Write(NumBurninIter);
                ctx.Writer.WriteBoolByte(ResetRandomGenerator);
            }
        }

        public sealed class LdaState : IDisposable
        {
            internal readonly ColumnInfo InfoEx;
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

            internal LdaState(IExceptionContext ectx, ColumnInfo ex, int numVocab)
                : this()
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(ex, "ex");

                ectx.Assert(numVocab >= 0);
                InfoEx = ex;
                _numVocab = numVocab;

                _ldaTrainer = new LdaSingleBox(
                    InfoEx.NumTopic,
                    numVocab, /* Need to set number of vocabulary here */
                    InfoEx.AlphaSum,
                    InfoEx.Beta,
                    InfoEx.NumIter,
                    InfoEx.LikelihoodInterval,
                    InfoEx.NumThread,
                    InfoEx.MHStep,
                    InfoEx.NumSummaryTermPerTopic,
                    false,
                    InfoEx.NumMaxDocToken);
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

                InfoEx = new ColumnInfo(ectx, ctx);

                _numVocab = ctx.Reader.ReadInt32();
                ectx.CheckDecode(_numVocab > 0);

                long memBlockSize = ctx.Reader.ReadInt64();
                ectx.CheckDecode(memBlockSize > 0);

                long aliasMemBlockSize = ctx.Reader.ReadInt64();
                ectx.CheckDecode(aliasMemBlockSize > 0);

                _ldaTrainer = new LdaSingleBox(
                    InfoEx.NumTopic,
                    _numVocab, /* Need to set number of vocabulary here */
                    InfoEx.AlphaSum,
                    InfoEx.Beta,
                    InfoEx.NumIter,
                    InfoEx.LikelihoodInterval,
                    InfoEx.NumThread,
                    InfoEx.MHStep,
                    InfoEx.NumSummaryTermPerTopic,
                    false,
                    InfoEx.NumMaxDocToken);

                _ldaTrainer.AllocateModelMemory(_numVocab, InfoEx.NumTopic, memBlockSize, aliasMemBlockSize);

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
                    _ldaTrainer.InitializeBeforeTest();
                    _predictionPreparationDone = true;
                }
            }

            public void Save(ModelSaveContext ctx)
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

                for (int i = 0; i < input.Count; i++)
                {
                    int termFreq = GetFrequency(input.Values[i]);
                    if (termFreq < 0)
                    {
                        // Ignore this row.
                        return 0;
                    }
                    if (docSize >= InfoEx.NumMaxDocToken - termFreq)
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
                    actualSize = _ldaTrainer.LoadDocDense(input.Values, termNum, input.Length);
                else
                    actualSize = _ldaTrainer.LoadDoc(input.Indices, input.Values, termNum, input.Length);

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

            public void Output(in VBuffer<Double> src, ref VBuffer<Float> dst, int numBurninIter, bool reset)
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

                int len = InfoEx.NumTopic;
                var values = dst.Values;
                var indices = dst.Indices;
                if (src.Count == 0)
                {
                    dst = new VBuffer<Float>(len, 0, values, indices);
                    return;
                }

                // Make sure all the frequencies are valid and truncate if the sum gets too large.
                int docSize = 0;
                int termNum = 0;
                for (int i = 0; i < src.Count; i++)
                {
                    int termFreq = GetFrequency(src.Values[i]);
                    if (termFreq < 0)
                    {
                        // REVIEW: Should this log a warning message? And what should it produce?
                        // It currently produces a vbuffer of all NA values.
                        // REVIEW: Need a utility method to do this...
                        if (Utils.Size(values) < len)
                            values = new Float[len];
                        for (int k = 0; k < len; k++)
                            values[k] = Float.NaN;
                        dst = new VBuffer<Float>(len, values, indices);
                        return;
                    }

                    if (docSize >= InfoEx.NumMaxDocToken - termFreq)
                        break;

                    docSize += termFreq;
                    termNum++;
                }

                // REVIEW: Too much memory allocation here on each prediction.
                List<KeyValuePair<int, float>> retTopics;
                if (src.IsDense)
                    retTopics = _ldaTrainer.TestDocDense(src.Values, termNum, numBurninIter, reset);
                else
                    retTopics = _ldaTrainer.TestDoc(src.Indices.Take(src.Count).ToArray(), src.Values.Take(src.Count).ToArray(), termNum, numBurninIter, reset);

                int count = retTopics.Count;
                Contracts.Assert(count <= len);
                if (Utils.Size(values) < count)
                    values = new Float[count];
                if (count < len && Utils.Size(indices) < count)
                    indices = new int[count];

                double normalizer = 0;
                for (int i = 0; i < count; i++)
                {
                    int index = retTopics[i].Key;
                    Float value = retTopics[i].Value;
                    Contracts.Assert(value >= 0);
                    Contracts.Assert(0 <= index && index < len);
                    if (count < len)
                    {
                        Contracts.Assert(i == 0 || indices[i - 1] < index);
                        indices[i] = index;
                    }
                    else
                        Contracts.Assert(index == i);

                    values[i] = value;
                    normalizer += value;
                }

                if (normalizer > 0)
                {
                    for (int i = 0; i < count; i++)
                        values[i] = (Float)(values[i] / normalizer);
                }
                dst = new VBuffer<Float>(len, count, values, indices);
            }

            public void Dispose()
            {
                _ldaTrainer.Dispose();
            }
        }

        private sealed class Mapper : MapperBase
        {
            private readonly LdaTransformer _parent;
            private readonly ColumnType[] _srcTypes;
            private readonly int[] _srcCols;

            public Mapper(LdaTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _srcTypes = new ColumnType[_parent.ColumnPairs.Length];
                _srcCols = new int[_parent.ColumnPairs.Length];

                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out _srcCols[i]);
                    var srcCol = inputSchema[_srcCols[i]];
                    _srcTypes[i] = srcCol.Type;
                }
            }

            public override Schema.Column[] GetOutputColumns()
            {
                var result = new Schema.Column[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                    result[i] = new Schema.Column(_parent.ColumnPairs[i].output, _parent._types[i], null);
                return result;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;

                var test = TestType(_srcTypes[iinfo]);
                return GetTopic(input, iinfo);
            }

            private ValueGetter<VBuffer<Float>> GetTopic(IRow input, int iinfo)
            {
                var getSrc = RowCursorUtils.GetVecGetterAs<Double>(NumberType.R8, input, _srcCols[iinfo]);
                var src = default(VBuffer<Double>);
                var lda = _parent._ldas[iinfo];
                int numBurninIter = lda.InfoEx.NumBurninIter;
                bool reset = lda.InfoEx.ResetRandomGenerator;
                return
                    (ref VBuffer<Float> dst) =>
                    {
                        // REVIEW: This will work, but there are opportunities for caching
                        // based on input.Counter that are probably worthwhile given how long inference takes.
                        getSrc(ref src);
                        lda.Output(in src, ref dst, numBurninIter, reset);
                    };
            }
        }

        public const string LoaderSignature = "LdaTransformer";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LIGHTLDA",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LdaTransformer).Assembly.FullName);
        }

        private readonly ColumnInfo[] _exes;
        private readonly LdaState[] _ldas;
        private readonly ColumnType[] _types;

        private const string RegistrationName = "LightLda";
        private const string WordTopicModelFilename = "word_topic_summary.txt";
        internal const string Summary = "The LDA transform implements LightLDA, a state-of-the-art implementation of Latent Dirichlet Allocation.";
        internal const string UserName = "Latent Dirichlet Allocation Transform";
        internal const string ShortName = "LightLda";

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        internal LdaTransformer(IHostEnvironment env, IDataView input, ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LdaTransformer)), GetColumnPairs(columns))
        {
            _exes = columns;
            _types = new ColumnType[columns.Length];
            _ldas = new LdaState[columns.Length];

            for (int i = 0; i < columns.Length; i++)
            {
                _types[i] = new VectorType(NumberType.Float, _exes[i].NumTopic);
            }
            using (var ch = Host.Start("Train"))
            {
                Train(ch, input, _ldas);
            }
        }

        private LdaTransformer(IHost host, ModelLoadContext ctx) : base(host, ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // ldaState[num infos]: The LDA parameters

            // Note: columnsLength would be just one in most cases.
            var columnsLength = ColumnPairs.Length;
            _exes = new ColumnInfo[columnsLength];
            _ldas = new LdaState[columnsLength];
            _types = new ColumnType[columnsLength];
            for (int i = 0; i < _ldas.Length; i++)
            {
                _ldas[i] = new LdaState(Host, ctx);
                _exes[i] = _ldas[i].InfoEx;
                _types[i] = new VectorType(NumberType.Float, _ldas[i].InfoEx.NumTopic);
            }
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

        ~LdaTransformer()
        {
            Dispose(false);
        }

        public LdaState GetLdaState(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < _ldas.Length);
            return _ldas[iinfo];
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(Schema.Create(inputSchema));

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            using (var ch = env.Start("ValidateArgs"))
            {

                for (int i = 0; i < cols.Length; i++)
                {
                    var item = args.Column[i];
                    cols[i] = new ColumnInfo(item.Source,
                        item.Name ?? item.Source,
                        item.NumTopic ?? args.NumTopic,
                        item.AlphaSum ?? args.AlphaSum,
                        item.Beta ?? args.Beta,
                        item.Mhstep ?? args.Mhstep,
                        item.NumIterations ?? args.NumIterations,
                        item.LikelihoodInterval ?? args.LikelihoodInterval,
                        item.NumThreads ?? args.NumThreads ?? 0,
                        item.NumMaxDocToken ?? args.NumMaxDocToken,
                        item.NumSummaryTermPerTopic ?? args.NumSummaryTermPerTopic,
                        item.NumBurninIterations ?? args.NumBurninIterations,
                        item.ResetRandomGenerator ?? args.ResetRandomGenerator);
                };
            }
            return new LdaTransformer(env, input, cols).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel
        private static LdaTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    h.CheckDecode(cbFloat == sizeof(Float));
                    return new LdaTransformer(h, ctx);
                });
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            // ldaState[num infos]: The LDA parameters

            ctx.Writer.Write(sizeof(Float));
            SaveColumns(ctx);
            for (int i = 0; i < _ldas.Length; i++)
            {
                _ldas[i].Save(ctx);
            }
        }

        private static string TestType(ColumnType t)
        {
            // LDA consumes term frequency vectors, so I am assuming VBuffer<Float> is an appropriate input type.
            // It must also be of known size for the sake of the LDA trainer initialization.
            if (t.IsKnownSizeVector && t.ItemType is NumberType)
                return null;
            return "Expected vector of number type of known size.";
        }

        private static int GetFrequency(double value)
        {
            int result = (int)value;
            if (!(result == value && result >= 0))
                return -1;
            return result;
        }

        private void Train(IChannel ch, IDataView trainingData, LdaState[] states)
        {
            Host.AssertValue(ch);
            ch.AssertValue(trainingData);
            ch.AssertValue(states);
            ch.Assert(states.Length == _exes.Length);

            bool[] activeColumns = new bool[trainingData.Schema.ColumnCount];
            int[] numVocabs = new int[_exes.Length];
            int[] srcCols = new int[_exes.Length];

            for (int i = 0; i < _exes.Length; i++)
            {
                if (!trainingData.Schema.TryGetColumnIndex(ColumnPairs[i].input, out int srcCol))
                    throw Host.ExceptSchemaMismatch(nameof(trainingData), "input", ColumnPairs[i].input);

                srcCols[i] = srcCol;
                activeColumns[srcCol] = true;
                numVocabs[i] = 0;
            }

            //the current lda needs the memory allocation before feedin data, so needs two sweeping of the data,
            //one for the pre-calc memory, one for feedin data really
            //another solution can be prepare these two value externally and put them in the beginning of the input file.
            long[] corpusSize = new long[_exes.Length];
            int[] numDocArray = new int[_exes.Length];

            using (var cursor = trainingData.GetRowCursor(col => activeColumns[col]))
            {
                var getters = new ValueGetter<VBuffer<Double>>[_exes.Length];
                for (int i = 0; i < _exes.Length; i++)
                {
                    corpusSize[i] = 0;
                    numDocArray[i] = 0;
                    getters[i] = RowCursorUtils.GetVecGetterAs<Double>(NumberType.R8, cursor, srcCols[i]);
                }
                VBuffer<Double> src = default(VBuffer<Double>);
                long rowCount = 0;

                while (cursor.MoveNext())
                {
                    ++rowCount;
                    for (int i = 0; i < _exes.Length; i++)
                    {
                        int docSize = 0;
                        getters[i](ref src);

                        // compute term, doc instance#.
                        for (int termID = 0; termID < src.Count; termID++)
                        {
                            int termFreq = GetFrequency(src.Values[termID]);
                            if (termFreq < 0)
                            {
                                // Ignore this row.
                                docSize = 0;
                                break;
                            }

                            if (docSize >= _exes[i].NumMaxDocToken - termFreq)
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

                for (int i = 0; i < _exes.Length; ++i)
                {
                    if (numDocArray[i] != rowCount)
                    {
                        ch.Assert(numDocArray[i] < rowCount);
                        ch.Warning($"Column '{ColumnPairs[i].input}' has skipped {rowCount - numDocArray[i]} of {rowCount} rows either empty or with negative, non-finite, or fractional values.");
                    }
                }
            }

            // Initialize all LDA states
            for (int i = 0; i < _exes.Length; i++)
            {
                var state = new LdaState(Host, _exes[i], numVocabs[i]);
                if (numDocArray[i] == 0 || corpusSize[i] == 0)
                    throw ch.Except("The specified documents are all empty in column '{0}'.", ColumnPairs[i].input);

                state.AllocateDataMemory(numDocArray[i], corpusSize[i]);
                states[i] = state;
            }

            using (var cursor = trainingData.GetRowCursor(col => activeColumns[col]))
            {
                int[] docSizeCheck = new int[_exes.Length];
                // This could be optimized so that if multiple trainers consume the same column, it is
                // fed into the train method once.
                var getters = new ValueGetter<VBuffer<Double>>[_exes.Length];
                for (int i = 0; i < _exes.Length; i++)
                {
                    docSizeCheck[i] = 0;
                    getters[i] = RowCursorUtils.GetVecGetterAs<Double>(NumberType.R8, cursor, srcCols[i]);
                }

                VBuffer<Double> src = default(VBuffer<Double>);

                while (cursor.MoveNext())
                {
                    for (int i = 0; i < _exes.Length; i++)
                    {
                        getters[i](ref src);
                        docSizeCheck[i] += states[i].FeedTrain(Host, in src);
                    }
                }
                for (int i = 0; i < _exes.Length; i++)
                {
                    Host.Assert(corpusSize[i] == docSizeCheck[i]);
                    states[i].CompleteTrain();
                }
            }
        }

        protected override IRowMapper MakeRowMapper(Schema schema)
        {
            return new Mapper(this, schema);
        }
    }

    /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
    public sealed class LdaEstimator : IEstimator<LdaTransformer>
    {
        internal static class Defaults
        {
            public const int NumTopic = 100;
            public const Single AlphaSum = 100;
            public const Single Beta = 0.01f;
            public const int Mhstep = 4;
            public const int NumIterations = 200;
            public const int LikelihoodInterval = 5;
            public const int NumThreads = 0;
            public const int NumMaxDocToken = 512;
            public const int NumSummaryTermPerTopic = 10;
            public const int NumBurninIterations = 10;
            public const bool ResetRandomGenerator = false;
        }

        private readonly IHost _host;
        private readonly LdaTransformer.ColumnInfo[] _columns;

        /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="inputColumn">The column containing text to tokenize.</param>
        /// <param name="outputColumn">The column containing output tokens. Null means <paramref name="inputColumn"/> is replaced.</param>
        /// <param name="numTopic">The number of topics in the LDA.</param>
        /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
        /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
        /// <param name="mhstep">Number of Metropolis Hasting step.</param>
        /// <param name="numIterations">Number of iterations.</param>
        /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
        /// <param name="numThreads">The number of training threads. Default value depends on number of logical processors.</param>
        /// <param name="numMaxDocToken">The threshold of maximum count of tokens per doc.</param>
        /// <param name="numSummaryTermPerTopic">The number of words to summarize the topic.</param>
        /// <param name="numBurninIterations">The number of burn-in iterations.</param>
        /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
        public LdaEstimator(IHostEnvironment env,
            string inputColumn,
            string outputColumn = null,
            int numTopic = Defaults.NumTopic,
            Single alphaSum = Defaults.AlphaSum,
            Single beta = Defaults.Beta,
            int mhstep = Defaults.Mhstep,
            int numIterations = Defaults.NumIterations,
            int likelihoodInterval = Defaults.LikelihoodInterval,
            int numThreads = Defaults.NumThreads,
            int numMaxDocToken = Defaults.NumMaxDocToken,
            int numSummaryTermPerTopic = Defaults.NumSummaryTermPerTopic,
            int numBurninIterations = Defaults.NumBurninIterations,
            bool resetRandomGenerator = Defaults.ResetRandomGenerator)
            : this(env, new[] { new LdaTransformer.ColumnInfo(inputColumn, outputColumn ?? inputColumn,
                numTopic, alphaSum, beta, mhstep, numIterations, likelihoodInterval, numThreads, numMaxDocToken,
                numSummaryTermPerTopic, numBurninIterations, resetRandomGenerator) })
        { }

        /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="columns">Pairs of columns to compute LDA.</param>
        public LdaEstimator(IHostEnvironment env, params LdaTransformer.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(LdaEstimator));
            _columns = columns;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in _columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (col.ItemType.RawKind != DataKind.R4 || col.Kind != SchemaShape.Column.VectorKind.Vector)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);

                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);
            }

            return new SchemaShape(result.Values);
        }

        public LdaTransformer Fit(IDataView input)
        {
            return new LdaTransformer(_host, input, _columns);
        }
    }
}
