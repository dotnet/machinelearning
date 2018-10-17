// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;

[assembly: LoadableClass(WordEmbeddingsTransform.Summary, typeof(IDataTransform), typeof(WordEmbeddingsTransform), typeof(WordEmbeddingsTransform.Arguments),
    typeof(SignatureDataTransform), WordEmbeddingsTransform.UserName, "WordEmbeddingsTransform", WordEmbeddingsTransform.ShortName, DocName = "transform/WordEmbeddingsTransform.md")]

[assembly: LoadableClass(WordEmbeddingsTransform.Summary, typeof(IDataTransform), typeof(WordEmbeddingsTransform), null, typeof(SignatureLoadDataTransform),
    WordEmbeddingsTransform.UserName, WordEmbeddingsTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(WordEmbeddingsTransform), null, typeof(SignatureLoadModel),
    WordEmbeddingsTransform.UserName, WordEmbeddingsTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(WordEmbeddingsTransform), null, typeof(SignatureLoadRowMapper),
    WordEmbeddingsTransform.UserName, WordEmbeddingsTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
    public sealed class WordEmbeddingsTransform : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
        {
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
                return TryUnparseCore(sb);
            }
        }

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 0)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Pre-trained model used to create the vocabulary", ShortName = "model", SortOrder = 1)]
            public PretrainedModelKind? ModelKind = PretrainedModelKind.Sswe;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Filename for custom word embedding model",
                ShortName = "dataFile", SortOrder = 2)]
            public string CustomLookupTable;
        }

        internal const string Summary = "Word Embeddings transform is a text featurizer which converts vectors of text tokens into sentence " +
            "vectors using a pre-trained model";
        internal const string UserName = "Word Embeddings Transform";
        internal const string ShortName = "WordEmbeddings";
        public const string LoaderSignature = "WordEmbeddingsTransform";

        public static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "W2VTRANS",
                verWrittenCur: 0x00010001, //Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(WordEmbeddingsTransform).Assembly.FullName);
        }

        private readonly PretrainedModelKind? _modelKind;
        private readonly string _modelFileNameWithPath;
        private static object _embeddingsLock = new object();
        private readonly bool _customLookup;
        private readonly int _linesToSkip;
        private readonly Model _currentVocab;
        private static Dictionary<string, WeakReference<Model>> _vocab = new Dictionary<string, WeakReference<Model>>();
        public IReadOnlyCollection<(string input, string output)> Columns => ColumnPairs.AsReadOnly();

        private sealed class Model
        {
            public readonly BigArray<float> WordVectors;
            private readonly NormStr.Pool _pool;
            public readonly int Dimension;

            public Model(int dimension)
            {
                Dimension = dimension;
                WordVectors = new BigArray<float>();
                _pool = new NormStr.Pool();
            }

            public void AddWordVector(IChannel ch, string word, float[] wordVector)
            {
                ch.Assert(wordVector.Length == Dimension);
                if (_pool.Get(word) == null)
                {
                    _pool.Add(word);
                    WordVectors.AddRange(wordVector, Dimension);
                }
            }

            public bool GetWordVector(ref ReadOnlyMemory<char> word, float[] wordVector)
            {
                NormStr str = _pool.Get(word);
                if (str != null)
                {
                    WordVectors.CopyTo(str.Id * Dimension, wordVector, Dimension);
                    return true;
                }
                return false;
            }

            public long GetNumWords()
            {
                return _pool.LongCount();
            }

            public List<string> GetWordLabels()
            {

                var labels = new List<string>();
                foreach (var label in _pool)
                {
                    labels.Add(new string(label.Value.ToArray()));
                }
                return labels;
            }

        }

        /// <summary>
        /// Information for each column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;

            public ColumnInfo(string input, string output)
            {
                Contracts.CheckNonEmpty(input, nameof(input));
                Contracts.CheckNonEmpty(output, nameof(output));

                Input = input;
                Output = output;
            }
        }

        private const string RegistrationName = "WordEmbeddings";

        private const int Timeout = 10 * 60 * 1000;

        /// <summary>
        /// Instantiates <see cref="WordEmbeddingsTransform"/> using the pretrained word embedding model specified by <paramref name="modelKind"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the output column.</param>
        /// <param name="modelKind">The pretrained word embedding model.</param>
        public WordEmbeddingsTransform(IHostEnvironment env, string inputColumn, string outputColumn,
           PretrainedModelKind modelKind = PretrainedModelKind.Sswe)
           : this(env, modelKind, new ColumnInfo(inputColumn, outputColumn))
        {
        }

        /// <summary>
        /// Instantiates <see cref="WordEmbeddingsTransform"/> using the custom word embedding model by loading it from the file specified by the <paramref name="customModelFile"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="inputColumn">Name of the input column.</param>
        /// <param name="outputColumn">Name of the output column.</param>
        /// <param name="customModelFile">Filename for custom word embedding model.</param>
        public WordEmbeddingsTransform(IHostEnvironment env, string inputColumn, string outputColumn, string customModelFile)
           : this(env, customModelFile, new ColumnInfo(inputColumn, outputColumn))
        {
        }

        /// <summary>
        /// Instantiates <see cref="WordEmbeddingsTransform"/> using the pretrained word embedding model specified by <paramref name="modelKind"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="modelKind">The pretrained word embedding model.</param>
        /// <param name="columns">Input/Output columns.</param>
        public WordEmbeddingsTransform(IHostEnvironment env, PretrainedModelKind modelKind, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            env.CheckUserArg(Enum.IsDefined(typeof(PretrainedModelKind), modelKind), nameof(modelKind));

            _modelKind = modelKind;
            _modelFileNameWithPath = EnsureModelFile(env, out _linesToSkip, (PretrainedModelKind)_modelKind);
            _currentVocab = GetVocabularyDictionary();
        }

        /// <summary>
        /// Instantiates <see cref="WordEmbeddingsTransform"/> using the custom word embedding model by loading it from the file specified by the <paramref name="customModelFile"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="customModelFile">Filename for custom word embedding model.</param>
        /// <param name="columns">Input/Output columns.</param>
        public WordEmbeddingsTransform(IHostEnvironment env, string customModelFile, params ColumnInfo[] columns)
           : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            env.CheckValue(customModelFile, nameof(customModelFile));
            Host.CheckNonWhiteSpace(customModelFile, nameof(customModelFile));

            _modelKind = null;
            _customLookup = true;
            _modelFileNameWithPath = customModelFile;
            _currentVocab = GetVocabularyDictionary();
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            if (args.ModelKind == null)
                args.ModelKind = PretrainedModelKind.Sswe;
            env.CheckUserArg(!args.ModelKind.HasValue || Enum.IsDefined(typeof(PretrainedModelKind), args.ModelKind), nameof(args.ModelKind));

            env.CheckValue(args.Column, nameof(args.Column));

            var cols = new ColumnInfo[args.Column.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = args.Column[i];
                cols[i] = new ColumnInfo(
                    item.Source ?? item.Name,
                    item.Name);
            }

            bool customLookup = !string.IsNullOrWhiteSpace(args.CustomLookupTable);
            if (customLookup)
                return new WordEmbeddingsTransform(env, args.CustomLookupTable, cols).MakeDataTransform(input);
            else
                return new WordEmbeddingsTransform(env, args.ModelKind.Value, cols).MakeDataTransform(input);
        }

        private WordEmbeddingsTransform(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            Host.AssertValue(ctx);
            _customLookup = ctx.Reader.ReadBoolByte();

            if (_customLookup)
            {
                _modelFileNameWithPath = ctx.LoadNonEmptyString();
                _modelKind = null;
            }
            else
            {
                _modelKind = (PretrainedModelKind)ctx.Reader.ReadUInt32();
                _modelFileNameWithPath = EnsureModelFile(Host, out _linesToSkip, (PretrainedModelKind)_modelKind);
            }

            Host.CheckNonWhiteSpace(_modelFileNameWithPath, nameof(_modelFileNameWithPath));
            _currentVocab = GetVocabularyDictionary();
        }

        public static WordEmbeddingsTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new WordEmbeddingsTransform(h, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            base.SaveColumns(ctx);
            ctx.Writer.WriteBoolByte(_customLookup);
            if (_customLookup)
                ctx.SaveString(_modelFileNameWithPath);
            else
                ctx.Writer.Write((uint)_modelKind);
        }

        protected override IRowMapper MakeRowMapper(ISchema schema)
           => new Mapper(this, Schema.Create(schema));

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var colType = inputSchema.GetColumnType(srcCol);
            if (!(colType.IsVector && colType.ItemType.IsText))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].input, "Text", inputSchema.GetColumnType(srcCol).ToString());
        }

        private sealed class Mapper : MapperBase, ISaveAsOnnx
        {
            private readonly WordEmbeddingsTransform _parent;
            private readonly VectorType _outputType;

            public Mapper(WordEmbeddingsTransform parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                Host.CheckValue(inputSchema, nameof(inputSchema));
                Host.CheckValue(parent, nameof(parent));

                _parent = parent;
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    _parent.CheckInputColumn(inputSchema, i, ColMapNewToOld[i]);
                }
                _outputType = new VectorType(NumberType.R4, 3 * _parent._currentVocab.Dimension);
            }

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public override Schema.Column[] GetOutputColumns()
                => _parent.ColumnPairs.Select(x => new Schema.Column(x.output, _outputType, null)).ToArray();

            public void SaveAsOnnx(OnnxContext ctx)
            {
                foreach (var (input, output) in _parent.Columns)
                {
                    var srcVariableName = ctx.GetVariableName(input);
                    var schema = _parent.GetOutputSchema(InputSchema);
                    schema.TryGetColumnIndex(output, out int colIndex);
                    var dstVariableName = ctx.AddIntermediateVariable(schema.GetColumnType(colIndex), output);
                    SaveAsOnnxCore(ctx, srcVariableName, dstVariableName);
                }
            }

            private void SaveAsOnnxCore(OnnxContext ctx, string srcVariableName, string dstVariableName)
            {
                // Converts 1 column that is taken as input to the transform into one column of output
                //
                // Symbols:
                // j: # of features per word in the pretrained model
                // n: # of input words
                // X: word input, n tensor
                // k: # of words in pretrained model (known when transform is created)
                // S: word labels, k tensor (known when transform is created)
                // D: word embeddings, k-by-j tensor(known when transform is created)
                //
                // X [n]-- > LabelEncoder(classes_strings = S [k], default_int64 = 1)        D [k, j]->Initializer
                //                              |                                                   |
                //                             Y[n]                                                 |
                //                              |                                                   |
                //                              -------->(indices) Gather (axis= 1)(data) <---------
                //                                                      |
                //                                                   W[n, j]
                //                                                /     |      \
                //                        ---------------------- /      |       \----------------------
                //						/                               |                                \
                //              ReduceMin(axes = [0])         ReduceMean(axes = [0])         ReduceMax(axes = [0])
                //                      |                                |                                |
                //                    J[j]                             K[j]                             L[j]
                //                      |                                |                                |
                //                       --------------------Concat (axis = 1) ---------------------------
                //                                                       |
                //                                                   P[j * 3]

                // Allocate D, a constant tensor
                var shapeD = new long[] { _parent._currentVocab.GetNumWords(), _parent._currentVocab.Dimension };
                var tensorD = _parent._currentVocab.WordVectors;
                var nameD = ctx.AddInitializer(tensorD, shapeD, "WordEmbeddingWeights");

                // Retrieve name of X
                var nameX = srcVariableName;

                // Do label encoding
                var nameY = ctx.AddIntermediateVariable(null, "LabelEncodedInput", true);
                var nodeY = ctx.CreateNode("LabelEncoder", nameX, nameY, ctx.GetNodeName("LabelEncoder"));
                nodeY.AddAttribute("classes_strings", _parent._currentVocab.GetWordLabels());
                nodeY.AddAttribute("default_int64", 1);

                // Do gather
                var nameW = ctx.AddIntermediateVariable(null, "WeightsOfInput", true);
                var nodeW = ctx.CreateNode("Gather", new[] { nameD, nameY }, new[] { nameW }, ctx.GetNodeName("Gather"), "");

                long[] axes = new long[] { 0 };
                // Do reduce min
                var nameJ = ctx.AddIntermediateVariable(null, "MinWeights", true);
                var nodeJ = ctx.CreateNode("ReduceMin", nameW, nameJ, ctx.GetNodeName("ReduceMin"), "");
                nodeJ.AddAttribute("axes", axes);

                // Do reduce mean
                var nameK = ctx.AddIntermediateVariable(null, "MeanWeights", true);
                var nodeK = ctx.CreateNode("ReduceMean", nameW, nameK, ctx.GetNodeName("ReduceMean"), "");
                nodeK.AddAttribute("axes", axes);

                // Do reduce max
                var nameL = ctx.AddIntermediateVariable(null, "MaxWeights", true);
                var nodeL = ctx.CreateNode("ReduceMax", nameW, nameL, ctx.GetNodeName("ReduceMax"), "");
                nodeL.AddAttribute("axes", axes);

                // Do concat
                var nameP = dstVariableName;
                var nodeP = ctx.CreateNode("Concat", new[] { nameJ, nameK, nameL }, new[] { nameP }, ctx.GetNodeName("Concat"), "");
                nodeP.AddAttribute("axis", 1);
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;
                return GetGetterVec(input, iinfo);
            }

            private ValueGetter<VBuffer<float>> GetGetterVec(IRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var colType = input.Schema.GetColumnType(ColMapNewToOld[iinfo]);
                Host.Assert(colType.IsVector);
                Host.Assert(colType.ItemType.IsText);

                var srcGetter = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(ColMapNewToOld[iinfo]);
                var src = default(VBuffer<ReadOnlyMemory<char>>);
                int dimension = _parent._currentVocab.Dimension;
                float[] wordVector = new float[_parent._currentVocab.Dimension];

                return
                    (ref VBuffer<float> dst) =>
                    {
                        int deno = 0;
                        srcGetter(ref src);
                        var values = dst.Values;
                        if (Utils.Size(values) != 3 * dimension)
                            values = new float[3 * dimension];
                        int offset = 2 * dimension;
                        for (int i = 0; i < dimension; i++)
                        {
                            values[i] = float.MaxValue;
                            values[i + dimension] = 0;
                            values[i + offset] = float.MinValue;
                        }
                        for (int word = 0; word < src.Count; word++)
                        {
                            if (_parent._currentVocab.GetWordVector(ref src.Values[word], wordVector))
                            {
                                deno++;
                                for (int i = 0; i < dimension; i++)
                                {
                                    float currentTerm = wordVector[i];
                                    if (values[i] > currentTerm)
                                        values[i] = currentTerm;
                                    values[dimension + i] += currentTerm;
                                    if (values[offset + i] < currentTerm)
                                        values[offset + i] = currentTerm;
                                }
                            }
                        }

                        if (deno != 0)
                            for (int index = 0; index < dimension; index++)
                                values[index + dimension] /= deno;

                        dst = new VBuffer<float>(values.Length, values, dst.Indices);
                    };
            }
        }

        public enum PretrainedModelKind
        {
            [TGUI(Label = "GloVe 50D")]
            GloVe50D = 0,

            [TGUI(Label = "GloVe 100D")]
            GloVe100D = 1,

            [TGUI(Label = "GloVe 200D")]
            GloVe200D = 2,

            [TGUI(Label = "GloVe 300D")]
            GloVe300D = 3,

            [TGUI(Label = "GloVe Twitter 25D")]
            GloVeTwitter25D = 4,

            [TGUI(Label = "GloVe Twitter 50D")]
            GloVeTwitter50D = 5,

            [TGUI(Label = "GloVe Twitter 100D")]
            GloVeTwitter100D = 6,

            [TGUI(Label = "GloVe Twitter 200D")]
            GloVeTwitter200D = 7,

            [TGUI(Label = "fastText Wikipedia 300D")]
            FastTextWikipedia300D = 8,

            [TGUI(Label = "Sentiment-Specific Word Embedding")]
            Sswe = 9
        }

        private static Dictionary<PretrainedModelKind, string> _modelsMetaData = new Dictionary<PretrainedModelKind, string>()
        {
             { PretrainedModelKind.GloVe50D, "glove.6B.50d.txt" },
             { PretrainedModelKind.GloVe100D, "glove.6B.100d.txt" },
             { PretrainedModelKind.GloVe200D, "glove.6B.200d.txt" },
             { PretrainedModelKind.GloVe300D, "glove.6B.300d.txt" },
             { PretrainedModelKind.GloVeTwitter25D, "glove.twitter.27B.25d.txt" },
             { PretrainedModelKind.GloVeTwitter50D, "glove.twitter.27B.50d.txt" },
             { PretrainedModelKind.GloVeTwitter100D, "glove.twitter.27B.100d.txt" },
             { PretrainedModelKind.GloVeTwitter200D, "glove.twitter.27B.200d.txt" },
             { PretrainedModelKind.FastTextWikipedia300D, "wiki.en.vec" },
             { PretrainedModelKind.Sswe, "sentiment.emd" }
        };

        private static Dictionary<PretrainedModelKind, int> _linesToSkipInModels = new Dictionary<PretrainedModelKind, int>()
            { { PretrainedModelKind.FastTextWikipedia300D, 1 } };

        private string EnsureModelFile(IHostEnvironment env, out int linesToSkip, PretrainedModelKind kind)
        {
            linesToSkip = 0;
            if (_modelsMetaData.ContainsKey(kind))
            {
                var modelFileName = _modelsMetaData[kind];
                if (_linesToSkipInModels.ContainsKey(kind))
                    linesToSkip = _linesToSkipInModels[kind];
                using (var ch = Host.Start("Ensuring resources"))
                {
                    string dir = kind == PretrainedModelKind.Sswe ? Path.Combine("Text", "Sswe") : "WordVectors";
                    var url = $"{dir}/{modelFileName}";
                    var ensureModel = ResourceManagerUtils.Instance.EnsureResource(Host, ch, url, modelFileName, dir, Timeout);
                    ensureModel.Wait();
                    var errorResult = ResourceManagerUtils.GetErrorMessage(out var errorMessage, ensureModel.Result);
                    if (errorResult != null)
                    {
                        var directory = Path.GetDirectoryName(errorResult.FileName);
                        var name = Path.GetFileName(errorResult.FileName);
                        throw ch.Except($"{errorMessage}\nModel file for Word Embedding transform could not be found! " +
                            $@"Please copy the model file '{name}' from '{url}' to '{directory}'.");
                    }
                    return ensureModel.Result.FileName;
                }
            }
            throw Host.Except($"Can't map model kind = {kind} to specific file, please refer to https://aka.ms/MLNetIssue for assistance");
        }

        private Model GetVocabularyDictionary()
        {
            int dimension = 0;
            if (!File.Exists(_modelFileNameWithPath))
                throw Host.Except("Custom word embedding model file '{0}' could not be found for Word Embeddings transform.", _modelFileNameWithPath);

            if (_vocab.ContainsKey(_modelFileNameWithPath) && _vocab[_modelFileNameWithPath] != null)
            {
                if (_vocab[_modelFileNameWithPath].TryGetTarget(out Model model))
                {
                    dimension = model.Dimension;
                    return model;
                }
            }

            lock (_embeddingsLock)
            {
                if (_vocab.ContainsKey(_modelFileNameWithPath) && _vocab[_modelFileNameWithPath] != null)
                {
                    if (_vocab[_modelFileNameWithPath].TryGetTarget(out Model modelObject))
                    {
                        dimension = modelObject.Dimension;
                        return modelObject;
                    }
                }

                Model model = null;
                using (StreamReader sr = File.OpenText(_modelFileNameWithPath))
                {
                    string line;
                    int lineNumber = 1;
                    char[] delimiters = { ' ', '\t' };
                    using (var ch = Host.Start(LoaderSignature))
                    using (var pch = Host.StartProgressChannel("Building Vocabulary from Model File for Word Embeddings Transform"))
                    {
                        var header = new ProgressHeader(new[] { "lines" });
                        pch.SetHeader(header, e => e.SetProgress(0, lineNumber));
                        string firstLine = sr.ReadLine();
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (lineNumber >= _linesToSkip)
                            {
                                string[] words = line.TrimEnd().Split(delimiters);
                                dimension = words.Length - 1;
                                if (model == null)
                                    model = new Model(dimension);
                                if (model.Dimension != dimension)
                                    ch.Warning($"Dimension mismatch while reading model file: '{_modelFileNameWithPath}', line number {lineNumber + 1}, expected dimension = {model.Dimension}, received dimension = {dimension}");
                                else
                                {
                                    float tmp;
                                    string key = words[0];
                                    float[] value = words.Skip(1).Select(x => float.TryParse(x, out tmp) ? tmp : Single.NaN).ToArray();
                                    if (!value.Contains(Single.NaN))
                                        model.AddWordVector(ch, key, value);
                                    else
                                        ch.Warning($"Parsing error while reading model file: '{_modelFileNameWithPath}', line number {lineNumber + 1}");
                                }
                            }
                            lineNumber++;
                        }

                        // Handle first line of the embedding file separately since some embedding files including fastText have a single-line header
                        string[] wordsInFirstLine = firstLine.TrimEnd().Split(delimiters);
                        dimension = wordsInFirstLine.Length - 1;
                        if (model == null)
                            model = new Model(dimension);
                        if (model.Dimension == dimension)
                        {
                            float temp;
                            string firstKey = wordsInFirstLine[0];
                            float[] firstValue = wordsInFirstLine.Skip(1).Select(x => float.TryParse(x, out temp) ? temp : Single.NaN).ToArray();
                            if (!firstValue.Contains(Single.NaN))
                                model.AddWordVector(ch, firstKey, firstValue);
                        }
                        pch.Checkpoint(lineNumber);
                    }
                }
                _vocab[_modelFileNameWithPath] = new WeakReference<Model>(model, false);
                return model;
            }
        }
    }

    public sealed class WordEmbeddingsEstimator : IEstimator<WordEmbeddingsTransform>
    {
        private readonly IHost _host;
        private readonly WordEmbeddingsTransform.ColumnInfo[] _columns;
        private readonly WordEmbeddingsTransform.PretrainedModelKind? _modelKind;
        private readonly string _customLookupTable;

        public WordEmbeddingsEstimator(IHostEnvironment env, string inputColumn, string outputColumn,
           WordEmbeddingsTransform.PretrainedModelKind modelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe)
            : this(env, modelKind, new WordEmbeddingsTransform.ColumnInfo(inputColumn, outputColumn))
        {
        }

        public WordEmbeddingsEstimator(IHostEnvironment env, string inputColumn, string outputColumn, string customModelFile)
            : this(env, customModelFile, new WordEmbeddingsTransform.ColumnInfo(inputColumn, outputColumn))
        {
        }

        public WordEmbeddingsEstimator(IHostEnvironment env,
            WordEmbeddingsTransform.PretrainedModelKind modelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe, params WordEmbeddingsTransform.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(WordEmbeddingsEstimator));
            _modelKind = modelKind;
            _customLookupTable = null;
            _columns = columns;
        }

        public WordEmbeddingsEstimator(IHostEnvironment env, string customModelFile, params WordEmbeddingsTransform.ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(WordEmbeddingsEstimator));
            _modelKind = null;
            _customLookupTable = customModelFile;
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
                if (!(col.ItemType is TextType) || (col.Kind != SchemaShape.Column.VectorKind.VariableVector && col.Kind != SchemaShape.Column.VectorKind.Vector))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, new VectorType(TextType.Instance).ToString(), col.GetTypeString());

                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.Vector, NumberType.R4, false);
            }

            return new SchemaShape(result.Values);
        }

        public WordEmbeddingsTransform Fit(IDataView input)
        {
            bool customLookup = !string.IsNullOrWhiteSpace(_customLookupTable);
            if (customLookup)
                return new WordEmbeddingsTransform(_host, _customLookupTable, _columns);
            else
                return new WordEmbeddingsTransform(_host, _modelKind.Value, _columns);
        }
    }

    public static class WordEmbeddingsStaticExtensions
    {
        /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
        /// <param name="input">Vector of tokenized text.</param>
        /// <param name="modelKind">The pretrained word embedding model.</param>
        /// <returns></returns>
        public static Vector<float> WordEmbeddings(this VarVector<string> input, WordEmbeddingsTransform.PretrainedModelKind modelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutColumn(input, modelKind);
        }

        /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
        /// <param name="input">Vector of tokenized text.</param>
        /// <param name="customModelFile">The custom word embedding model file.</param>
        public static Vector<float> WordEmbeddings(this VarVector<string> input, string customModelFile)
        {
            Contracts.CheckValue(input, nameof(input));
            return new OutColumn(input, customModelFile);
        }

        private sealed class OutColumn : Vector<float>
        {
            public PipelineColumn Input { get; }

            public OutColumn(VarVector<string> input, WordEmbeddingsTransform.PretrainedModelKind modelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe)
                : base(new Reconciler(modelKind), input)
            {
                Input = input;
            }

            public OutColumn(VarVector<string> input, string customModelFile = null)
                : base(new Reconciler(customModelFile), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly WordEmbeddingsTransform.PretrainedModelKind? _modelKind;
            private readonly string _customLookupTable;

            public Reconciler(WordEmbeddingsTransform.PretrainedModelKind modelKind = WordEmbeddingsTransform.PretrainedModelKind.Sswe)
            {
                _modelKind = modelKind;
                _customLookupTable = null;
            }

            public Reconciler(string customModelFile)
            {
                _modelKind = null;
                _customLookupTable = customModelFile;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);

                var cols = new WordEmbeddingsTransform.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var outCol = (OutColumn)toOutput[i];
                    cols[i] = new WordEmbeddingsTransform.ColumnInfo(inputNames[outCol.Input], outputNames[outCol]);
                }

                bool customLookup = !string.IsNullOrWhiteSpace(_customLookupTable);
                if (customLookup)
                    return new WordEmbeddingsEstimator(env, _customLookupTable, cols);
                else
                    return new WordEmbeddingsEstimator(env, _modelKind.Value, cols);
            }
        }
    }
}
