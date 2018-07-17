// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.TextAnalytics;

[assembly: LoadableClass(typeof(LdaTransform), typeof(LdaTransform.Arguments), typeof(SignatureDataTransform),
    LdaTransform.UserName, LdaTransform.LoaderSignature, LdaTransform.ShortName, DocName = "transform/LdaTransform.md")]

[assembly: LoadableClass(typeof(LdaTransform), null, typeof(SignatureLoadDataTransform),
    LdaTransform.UserName, LdaTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.TextAnalytics
{
    /// <summary>
    /// LightLDA transform: Big Topic Models on Modest Compute Clusters.
    /// <see href="http://arxiv.org/abs/1412.1576">LightLDA</see> is an implementation of Latent Dirichlet Allocation (LDA).
    /// Previous implementations of LDA such as SparseLDA or AliasLDA allow to achieve massive data and model scales,
    /// for example models with tens of billions of parameters to be inferred from billions of documents.
    /// However this requires using a cluster of thousands of machines with all ensuing costs to setup and maintain.
    /// LightLDA solves this problem in a more cost-effective manner by providing an implementation 
    /// that is efﬁcient enough for modest clusters with at most tens of machines... 
    /// For more details please see original LightLDA paper: 
    /// http://arxiv.org/abs/1412.1576
    /// http://www.www2015.it/documents/proceedings/proceedings/p1351.pdf
    /// and open source implementation: 
    /// https://github.com/Microsoft/LightLDA
    /// 
    /// See <a href="https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.TestFramework/DataPipe/TestDataPipe.cs"/>
    /// for an example on how to use LdaTransform.
    /// </summary>
    public sealed class LdaTransform : OneToOneTransformBase
    {
        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:srcs)", ShortName = "col", SortOrder = 49)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of topics in the LDA", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "20,40,100,200")]
            [TlcModule.SweepableDiscreteParam("NumTopic", new object[] { 20, 40, 100, 200 })]
            public int NumTopic = 100;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on document-topic vectors")]
            [TGUI(SuggestedSweeps = "1,10,100,200")]
            [TlcModule.SweepableDiscreteParam("AlphaSum", new object[] { 1, 10, 100, 200 })]
            public Single AlphaSum = 100;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Dirichlet prior on vocab-topic vectors")]
            [TGUI(SuggestedSweeps = "0.01,0.015,0.07,0.02")]
            [TlcModule.SweepableDiscreteParam("Beta", new object[] { 0.01f, 0.015f, 0.07f, 0.02f })]
            public Single Beta = 0.01f;

            [Argument(ArgumentType.Multiple, HelpText = "Number of Metropolis Hasting step")]
            [TGUI(SuggestedSweeps = "2,4,8,16")]
            [TlcModule.SweepableDiscreteParam("Mhstep", new object[] { 2, 4, 8, 16 })]
            public int Mhstep = 4;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of iterations", ShortName = "iter")]
            [TGUI(SuggestedSweeps = "100,200,300,400")]
            [TlcModule.SweepableDiscreteParam("NumIterations", new object[] { 100, 200, 300, 400 })]
            public int NumIterations = 200;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Compute log likelihood over local dataset on this iteration interval", ShortName = "llInterval")]
            public int LikelihoodInterval = 5;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The threshold of maximum count of tokens per doc", ShortName = "maxNumToken", SortOrder = 50)]
            public int NumMaxDocToken = 512;

            // REVIEW: Should change the default when multi-threading support is optimized.
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of training threads. Default value depends on number of logical processors.", ShortName = "t", SortOrder = 50)]
            public int? NumThreads;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of words to summarize the topic", ShortName = "ns")]
            public int NumSummaryTermPerTopic = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of burn-in iterations", ShortName = "burninIter")]
            [TGUI(SuggestedSweeps = "10,20,30,40")]
            [TlcModule.SweepableDiscreteParam("NumBurninIterations", new object[] { 10, 20, 30, 40 })]
            public int NumBurninIterations = 10;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Reset the random number generator for each document", ShortName = "reset")]
            public bool ResetRandomGenerator;

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

        private sealed class ColInfoEx
        {
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

            public ColInfoEx(IExceptionContext ectx, Column item, Arguments args)
            {
                Contracts.AssertValue(ectx);

                NumTopic = item.NumTopic ?? args.NumTopic;
                Contracts.CheckUserArg(NumTopic > 0, nameof(item.NumTopic), "Must be positive.");

                AlphaSum = item.AlphaSum ?? args.AlphaSum;

                Beta = item.Beta ?? args.Beta;

                MHStep = item.Mhstep ?? args.Mhstep;
                ectx.CheckUserArg(MHStep > 0, nameof(item.Mhstep), "Must be positive.");

                NumIter = item.NumIterations ?? args.NumIterations;
                ectx.CheckUserArg(NumIter > 0, nameof(item.NumIterations), "Must be positive.");

                LikelihoodInterval = item.LikelihoodInterval ?? args.LikelihoodInterval;
                ectx.CheckUserArg(LikelihoodInterval > 0, nameof(item.LikelihoodInterval), "Must be positive.");

                NumThread = item.NumThreads ?? args.NumThreads ?? 0;
                ectx.CheckUserArg(NumThread >= 0, nameof(item.NumThreads), "Must be positive or zero.");

                NumMaxDocToken = item.NumMaxDocToken ?? args.NumMaxDocToken;
                ectx.CheckUserArg(NumMaxDocToken > 0, nameof(item.NumMaxDocToken), "Must be positive.");

                NumSummaryTermPerTopic = item.NumSummaryTermPerTopic ?? args.NumSummaryTermPerTopic;
                ectx.CheckUserArg(NumSummaryTermPerTopic > 0, nameof(item.NumSummaryTermPerTopic), "Must be positive");

                NumBurninIter = item.NumBurninIterations ?? args.NumBurninIterations;
                ectx.CheckUserArg(NumBurninIter >= 0, nameof(item.NumBurninIterations), "Must be non-negative.");

                ResetRandomGenerator = item.ResetRandomGenerator ?? args.ResetRandomGenerator;
            }

            public ColInfoEx(IExceptionContext ectx, ModelLoadContext ctx)
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

            public void Save(ModelSaveContext ctx)
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

        public const string LoaderSignature = "LdaTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LIGHTLDA",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly ColInfoEx[] _exes;
        private readonly LdaState[] _ldas;
        private readonly ColumnType[] _types;
        private readonly bool _saveText;

        private const string RegistrationName = "LightLda";
        private const string WordTopicModelFilename = "word_topic_summary.txt";
        internal const string Summary = "The LDA transform implements LightLDA, a state-of-the-art implementation of Latent Dirichlet Allocation.";
        internal const string UserName = "Latent Dirichlet Allocation Transform";
        internal const string ShortName = "LightLda";

        public LdaTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, args.Column, input, TestType)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(args.NumTopic > 0, nameof(args.NumTopic), "Must be positive.");
            Host.CheckValue(input, nameof(input));
            Host.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column));
            _exes = new ColInfoEx[Infos.Length];
            _types = new ColumnType[Infos.Length];
            _ldas = new LdaState[Infos.Length];
            _saveText = args.OutputTopicWordSummary;
            for (int i = 0; i < Infos.Length; i++)
            {
                var ex = new ColInfoEx(Host, args.Column[i], args);
                _exes[i] = ex;
                _types[i] = new VectorType(NumberType.Float, ex.NumTopic);
            }
            using (var ch = Host.Start("Train"))
            {
                Train(ch, input, _ldas);
                ch.Done();
            }
            Metadata.Seal();
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

        ~LdaTransform()
        {
            Dispose(false);
        }

        private LdaTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestType)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // ldaState[num infos]: The LDA parameters

            // Note: infos.length would be just one in most cases.
            _exes = new ColInfoEx[Infos.Length];
            _ldas = new LdaState[Infos.Length];
            _types = new ColumnType[Infos.Length];
            for (int i = 0; i < _ldas.Length; i++)
            {
                _ldas[i] = new LdaState(Host, ctx);
                _exes[i] = _ldas[i].InfoEx;
                _types[i] = new VectorType(NumberType.Float, _ldas[i].InfoEx.NumTopic);
            }
            using (var ent = ctx.Repository.OpenEntryOrNull("model", WordTopicModelFilename))
            {
                _saveText = ent != null;
            }
            Metadata.Seal();
        }

        public static LdaTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(input, nameof(input));

            return h.Apply(
                "Loading Model",
                ch =>
                {
                    // *** Binary Format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    h.CheckDecode(cbFloat == sizeof(Float));
                    return new LdaTransform(h, ctx, input);
                });
        }

        public string GetTopicSummary()
        {
            StringWriter writer = new StringWriter();
            VBuffer<DvText> slotNames = default(VBuffer<DvText>);
            for (int i = 0; i < _ldas.Length; i++)
            {
                GetSlotNames(i, ref slotNames);
                _ldas[i].GetTopicSummaryWriter(slotNames)(writer);
                writer.WriteLine();
            }
            return writer.ToString();
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
            SaveBase(ctx);
            Host.Assert(_ldas.Length == Infos.Length);
            VBuffer<DvText> slotNames = default(VBuffer<DvText>);
            for (int i = 0; i < _ldas.Length; i++)
            {
                GetSlotNames(i, ref slotNames);
                _ldas[i].Save(ctx, _saveText, slotNames);
            }
        }

        private void GetSlotNames(int iinfo, ref VBuffer<DvText> dst)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            if (Source.Schema.HasSlotNames(Infos[iinfo].Source, Infos[iinfo].TypeSrc.ValueCount))
                Source.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, Infos[iinfo].Source, ref dst);
            else
                dst = default(VBuffer<DvText>);
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
            ch.Assert(states.Length == Infos.Length);

            bool[] activeColumns = new bool[trainingData.Schema.ColumnCount];
            int[] numVocabs = new int[Infos.Length];

            for (int i = 0; i < Infos.Length; i++)
            {
                activeColumns[Infos[i].Source] = true;
                numVocabs[i] = 0;
            }

            //the current lda needs the memory allocation before feedin data, so needs two sweeping of the data, 
            //one for the pre-calc memory, one for feedin data really
            //another solution can be prepare these two value externally and put them in the beginning of the input file.
            long[] corpusSize = new long[Infos.Length];
            int[] numDocArray = new int[Infos.Length];

            using (var cursor = trainingData.GetRowCursor(col => activeColumns[col]))
            {
                var getters = new ValueGetter<VBuffer<Double>>[Utils.Size(Infos)];
                for (int i = 0; i < Infos.Length; i++)
                {
                    corpusSize[i] = 0;
                    numDocArray[i] = 0;
                    getters[i] = RowCursorUtils.GetVecGetterAs<Double>(NumberType.R8, cursor, Infos[i].Source);
                }
                VBuffer<Double> src = default(VBuffer<Double>);
                long rowCount = 0;

                while (cursor.MoveNext())
                {
                    ++rowCount;
                    for (int i = 0; i < Infos.Length; i++)
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

                for (int i = 0; i < Infos.Length; ++i)
                {
                    if (numDocArray[i] != rowCount)
                    {
                        ch.Assert(numDocArray[i] < rowCount);
                        ch.Warning($"Column '{Infos[i].Name}' has skipped {rowCount - numDocArray[i]} of {rowCount} rows either empty or with negative, non-finite, or fractional values.");
                    }
                }
            }

            // Initialize all LDA states
            for (int i = 0; i < Infos.Length; i++)
            {
                var state = new LdaState(Host, _exes[i], numVocabs[i]);
                if (numDocArray[i] == 0 || corpusSize[i] == 0)
                    throw ch.Except("The specified documents are all empty in column '{0}'.", Infos[i].Name);

                state.AllocateDataMemory(numDocArray[i], corpusSize[i]);
                states[i] = state;
            }

            using (var cursor = trainingData.GetRowCursor(col => activeColumns[col]))
            {
                int[] docSizeCheck = new int[Infos.Length];
                // This could be optimized so that if multiple trainers consume the same column, it is
                // fed into the train method once.
                var getters = new ValueGetter<VBuffer<Double>>[Utils.Size(Infos)];
                for (int i = 0; i < Infos.Length; i++)
                {
                    docSizeCheck[i] = 0;
                    getters[i] = RowCursorUtils.GetVecGetterAs<Double>(NumberType.R8, cursor, Infos[i].Source);
                }

                VBuffer<Double> src = default(VBuffer<Double>);

                while (cursor.MoveNext())
                {
                    for (int i = 0; i < Infos.Length; i++)
                    {
                        getters[i](ref src);
                        docSizeCheck[i] += states[i].FeedTrain(Host, ref src);
                    }
                }
                for (int i = 0; i < Infos.Length; i++)
                {
                    Host.Assert(corpusSize[i] == docSizeCheck[i]);
                    states[i].CompleteTrain();
                }
            }
        }

        private sealed class LdaState : IDisposable
        {
            public readonly ColInfoEx InfoEx;
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

            public LdaState(IExceptionContext ectx, ColInfoEx ex, int numVocab)
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

            public LdaState(IExceptionContext ectx, ModelLoadContext ctx)
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

                InfoEx = new ColInfoEx(ectx, ctx);

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

            public Action<TextWriter> GetTopicSummaryWriter(VBuffer<DvText> mapping)
            {
                Action<TextWriter> writeAction;

                if (mapping.Length == 0)
                {
                    writeAction =
                        writer =>
                        {
                            for (int i = 0; i < _ldaTrainer.NumTopic; i++)
                            {
                                KeyValuePair<int, float>[] topicSummaryVector = _ldaTrainer.GetTopicSummary(i);
                                writer.Write("{0}\t{1}\t", i, topicSummaryVector.Length);
                                foreach (KeyValuePair<int, float> p in topicSummaryVector)
                                    writer.Write("{0}:{1}\t", p.Key, p.Value);
                                writer.WriteLine();
                            }
                        };
                }
                else
                {
                    writeAction =
                        writer =>
                        {
                            DvText slotName = default(DvText);
                            for (int i = 0; i < _ldaTrainer.NumTopic; i++)
                            {
                                KeyValuePair<int, float>[] topicSummaryVector = _ldaTrainer.GetTopicSummary(i);
                                writer.Write("{0}\t{1}\t", i, topicSummaryVector.Length);
                                foreach (KeyValuePair<int, float> p in topicSummaryVector)
                                {
                                    mapping.GetItemOrDefault(p.Key, ref slotName);
                                    writer.Write("{0}[{1}]:{2}\t", p.Key, slotName, p.Value);
                                }
                                writer.WriteLine();
                            }
                        };
                }

                return writeAction;
            }

            public void Save(ModelSaveContext ctx, bool saveText, VBuffer<DvText> mapping)
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

                var writeAction = GetTopicSummaryWriter(mapping);

                // save word-topic summary in text
                if (saveText)
                    ctx.SaveTextStream(WordTopicModelFilename, writeAction);
            }

            public void AllocateDataMemory(int docNum, long corpusSize)
            {
                _ldaTrainer.AllocateDataMemory(docNum, corpusSize);
            }

            public int FeedTrain(IExceptionContext ectx, ref VBuffer<Double> input)
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

            public void Output(ref VBuffer<Double> src, ref VBuffer<Float> dst, int numBurninIter, bool reset)
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

        private ColumnType[] InitColumnTypes(int numTopics)
        {
            Host.Assert(Utils.Size(Infos) > 0);
            var types = new ColumnType[Infos.Length];
            for (int c = 0; c < Infos.Length; c++)
                types[c] = new VectorType(NumberType.Float, numTopics);
            return types;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Utils.Size(_types));
            return _types[iinfo];
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            return GetTopic(input, iinfo);
        }

        private ValueGetter<VBuffer<Float>> GetTopic(IRow input, int iinfo)
        {
            var getSrc = RowCursorUtils.GetVecGetterAs<Double>(NumberType.R8, input, Infos[iinfo].Source);
            var src = default(VBuffer<Double>);
            var lda = _ldas[iinfo];
            int numBurninIter = lda.InfoEx.NumBurninIter;
            bool reset = lda.InfoEx.ResetRandomGenerator;
            return
                (ref VBuffer<Float> dst) =>
                {
                    // REVIEW: This will work, but there are opportunities for caching
                    // based on input.Counter that are probably worthwhile given how long inference takes.
                    getSrc(ref src);
                    lda.Output(ref src, ref dst, numBurninIter, reset);
                };
        }
    }
}
