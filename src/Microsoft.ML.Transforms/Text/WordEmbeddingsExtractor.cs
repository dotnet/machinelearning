// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Text;

[assembly: LoadableClass(WordEmbeddingTransformer.Summary, typeof(IDataTransform), typeof(WordEmbeddingTransformer), typeof(WordEmbeddingTransformer.Options),
    typeof(SignatureDataTransform), WordEmbeddingTransformer.UserName, "WordEmbeddingsTransform", WordEmbeddingTransformer.ShortName, DocName = "transform/WordEmbeddingsTransform.md")]

[assembly: LoadableClass(WordEmbeddingTransformer.Summary, typeof(IDataTransform), typeof(WordEmbeddingTransformer), null, typeof(SignatureLoadDataTransform),
    WordEmbeddingTransformer.UserName, WordEmbeddingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(WordEmbeddingTransformer), null, typeof(SignatureLoadModel),
    WordEmbeddingTransformer.UserName, WordEmbeddingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(WordEmbeddingTransformer), null, typeof(SignatureLoadRowMapper),
    WordEmbeddingTransformer.UserName, WordEmbeddingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Text
{
    /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
    public sealed class WordEmbeddingTransformer : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
        {
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
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 0)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Pre-trained model used to create the vocabulary", ShortName = "model", SortOrder = 1)]
            public WordEmbeddingEstimator.PretrainedModelKind? ModelKind = WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Filename for custom word embedding model",
                ShortName = "dataFile", SortOrder = 2)]
            public string CustomLookupTable;
        }

        internal const string Summary = "Word Embeddings transform is a text featurizer which converts vectors of text tokens into sentence " +
            "vectors using a pre-trained model";
        internal const string UserName = "Word Embeddings Transform";
        internal const string ShortName = "WordEmbeddings";
        internal const string LoaderSignature = "WordEmbeddingsTransform";

        internal static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "W2VTRANS",
                verWrittenCur: 0x00010001, //Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(WordEmbeddingTransformer).Assembly.FullName);
        }

        private readonly WordEmbeddingEstimator.PretrainedModelKind? _modelKind;
        private readonly string _modelFileNameWithPath;
        private static object _embeddingsLock = new object();
        private readonly bool _customLookup;
        private readonly int _linesToSkip;
        private readonly Model _currentVocab;
        private static Dictionary<string, WeakReference<Model>> _vocab = new Dictionary<string, WeakReference<Model>>();

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
                    WordVectors.AddRange(wordVector);
                }
            }

            public bool GetWordVector(in ReadOnlyMemory<char> word, float[] wordVector)
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

        private const string RegistrationName = "WordEmbeddings";

        private const int Timeout = 10 * 60 * 1000;

        /// <summary>
        /// Instantiates <see cref="WordEmbeddingTransformer"/> using the pretrained word embedding model specified by <paramref name="modelKind"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="modelKind">The pretrained word embedding model.</param>
        internal WordEmbeddingTransformer(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
           WordEmbeddingEstimator.PretrainedModelKind modelKind = WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding)
           : this(env, modelKind, new WordEmbeddingEstimator.ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Instantiates <see cref="WordEmbeddingTransformer"/> using the custom word embedding model by loading it from the file specified by the <paramref name="customModelFile"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="customModelFile">Filename for custom word embedding model.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        internal WordEmbeddingTransformer(IHostEnvironment env, string outputColumnName, string customModelFile, string inputColumnName = null)
           : this(env, customModelFile, new WordEmbeddingEstimator.ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Instantiates <see cref="WordEmbeddingTransformer"/> using the pretrained word embedding model specified by <paramref name="modelKind"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="modelKind">The pretrained word embedding model.</param>
        /// <param name="columns">Input/Output columns.</param>
        internal WordEmbeddingTransformer(IHostEnvironment env, WordEmbeddingEstimator.PretrainedModelKind modelKind, params WordEmbeddingEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            env.CheckUserArg(Enum.IsDefined(typeof(WordEmbeddingEstimator.PretrainedModelKind), modelKind), nameof(modelKind));

            _modelKind = modelKind;
            _modelFileNameWithPath = EnsureModelFile(env, out _linesToSkip, (WordEmbeddingEstimator.PretrainedModelKind)_modelKind);
            _currentVocab = GetVocabularyDictionary(env);
        }

        /// <summary>
        /// Instantiates <see cref="WordEmbeddingTransformer"/> using the custom word embedding model by loading it from the file specified by the <paramref name="customModelFile"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="customModelFile">Filename for custom word embedding model.</param>
        /// <param name="columns">Input/Output columns.</param>
        internal WordEmbeddingTransformer(IHostEnvironment env, string customModelFile, params WordEmbeddingEstimator.ColumnOptions[] columns)
           : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            env.CheckValue(customModelFile, nameof(customModelFile));
            Host.CheckNonWhiteSpace(customModelFile, nameof(customModelFile));

            _modelKind = null;
            _customLookup = true;
            _modelFileNameWithPath = customModelFile;
            _currentVocab = GetVocabularyDictionary(env);
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(WordEmbeddingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            if (options.ModelKind == null)
                options.ModelKind = WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding;
            env.CheckUserArg(!options.ModelKind.HasValue || Enum.IsDefined(typeof(WordEmbeddingEstimator.PretrainedModelKind), options.ModelKind), nameof(options.ModelKind));

            env.CheckValue(options.Columns, nameof(options.Columns));

            var cols = new WordEmbeddingEstimator.ColumnOptions[options.Columns.Length];
            for (int i = 0; i < cols.Length; i++)
            {
                var item = options.Columns[i];
                cols[i] = new WordEmbeddingEstimator.ColumnOptions(
                    item.Name,
                    item.Source ?? item.Name);
            }

            bool customLookup = !string.IsNullOrWhiteSpace(options.CustomLookupTable);
            if (customLookup)
                return new WordEmbeddingTransformer(env, options.CustomLookupTable, cols).MakeDataTransform(input);
            else
                return new WordEmbeddingTransformer(env, options.ModelKind.Value, cols).MakeDataTransform(input);
        }

        private WordEmbeddingTransformer(IHost host, ModelLoadContext ctx)
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
                _modelKind = (WordEmbeddingEstimator.PretrainedModelKind)ctx.Reader.ReadUInt32();
                _modelFileNameWithPath = EnsureModelFile(Host, out _linesToSkip, (WordEmbeddingEstimator.PretrainedModelKind)_modelKind);
            }

            Host.CheckNonWhiteSpace(_modelFileNameWithPath, nameof(_modelFileNameWithPath));
            _currentVocab = GetVocabularyDictionary(host);
        }

        internal static WordEmbeddingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new WordEmbeddingTransformer(h, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
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

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var colType = inputSchema[srcCol].Type;
            if (!(colType is VectorType vectorType && vectorType.ItemType is TextDataViewType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, "Text", inputSchema[srcCol].Type.ToString());
        }

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private readonly WordEmbeddingTransformer _parent;
            private readonly VectorType _outputType;

            public Mapper(WordEmbeddingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                Host.CheckValue(inputSchema, nameof(inputSchema));
                Host.CheckValue(parent, nameof(parent));

                _parent = parent;
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    _parent.CheckInputColumn(inputSchema, i, ColMapNewToOld[i]);
                }
                _outputType = new VectorType(NumberDataViewType.Single, 3 * _parent._currentVocab.Dimension);
            }

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select(x => new DataViewSchema.DetachedColumn(x.outputColumnName, _outputType, null)).ToArray();

            public void SaveAsOnnx(OnnxContext ctx)
            {
                foreach (var (outputColumnName, inputColumnName) in _parent.ColumnPairs)
                {
                    var srcVariableName = ctx.GetVariableName(inputColumnName);
                    var schema = _parent.GetOutputSchema(InputSchema);
                    var dstVariableName = ctx.AddIntermediateVariable(schema[outputColumnName].Type, outputColumnName);
                    SaveAsOnnxCore(ctx, srcVariableName, dstVariableName);
                }
            }

            private void SaveAsOnnxCore(OnnxContext ctx, string srcVariableName, string dstVariableName)
            {
                // Converts 1 column that is taken as input to the transform into one column of output
                //
                // Missing words are mapped to k for finding average, k + 1 for finding min, and k + 2 for finding max
                // Those spots in the dictionary contain a vector of 0s, max floats, and min floats, respectively
                //
                // Symbols:
                // j: length of latent vector of every word in the pretrained model
                // n: length of input tensor (number of words)
                // X: word input, a tensor with n elements.
                // k: # of words in pretrained model (known when transform is created)
                // S: word labels, k tensor (known when transform is created)
                // D: word embeddings, (k + 3)-by-j tensor(known when transform is created). The extra three embeddings
                //      at the end are used for out of vocab words.
                // F: location value representing missing words, equal to k
                // P: output, a j * 3 tensor
                //
                //                                                      X [n]
                //                                                       |
                //                                                     nameX
                //                                                       |
                //                           LabelEncoder (classes_strings = S [k], default_int64 = k)
                //                                                       |
                //                            /----------------------- nameY -----------------------\
                //                           /   |                       |                           \
                //     Initialize (F)-------/----|------ nameF ------> Equal                          \
                //                         /     |                       |                             \
                //                        /      |                     nameA                            \
                //                       /       |                     / |  \                            \
                //                      /        '-------------|      /  |   \                            \
                //                     /                 ------|-----/   |    \------------------          \---------
                //                    /                 /      |         |                       \                   \
                //                    |      Cast (to = int64) |  Cast (to = float)              Not                 |
                //                    |             |          |         |                        |                  |
                //                    |         nameVMin       |       nameB                    nameQ                |
                //                    |             |          |         |                        |                  |
                //                  Add ------------'          | Scale (scale = 2.0)         Cast (to = int32)       |
                //                    |                        |         |                        |                  |
                //                    |                        |     nameSMax                  nameZ                 |
                //                    |                        |         |                        |                  |
                //                    |                        | Cast (to = int64)       ReduceSum (axes = [0])      |
                //                namePMin                     |         |                        |                  |
                //                    |                        |      nameVMax                 nameR                 |
                //                    |                        |         |                        |                  |
                //                    |                        '-- Add --'                Cast (to = float)          |
                //                    |   Initialize (D [k + 3, j]   |                            |                  |
                //                    |             |                |                            |                  |
                //                    |           nameD           namePMax                     nameRF                |
                //                    |             |                |                            |                  |
                //                    |             |                |                     Clip (min = 1.0)          |
                //                    |             |                |                            |                  |
                //                    |             |                |                          nameT                |
                //                    |             |----------------|----------------------------|--------\         |
                //                    |             |                |                            |         \        |
                //                    |   /---------'-------------\  |                            |          '----\  |
                //                  Gather                        Gather                          |               Gather
                //                    |                              |                            |                  |
                //                 nameGMin                       nameGMax                        |                nameW
                //                    |                              |                            |                  |
                //            ReduceMin (axes = [0])      ReduceMax (axes = [0])                  |        ReduceSum (axes = [0])
                //                    |                              |                            |                  |
                //                    |                              |                            |                nameK
                //                    |                              |                            |                  |
                //                    |                              |                            '------- Div ------'
                //                  nameJ                          nameL                                    |
                //                    |                              |                                   nameE
                //                    |                              |                                      |
                //                    '------------------- Concat (axis = 1) -------------------------------'
                //                                                   |
                //                                                 nameP
                //                                                   |
                //                                               P [j * 3]

                long[] axes = new long[] { 0 };
                // Allocate D, a constant tensor representing word embedding weights.
                var shapeD = new long[] { _parent._currentVocab.GetNumWords() + 3, _parent._currentVocab.Dimension };
                var wordVectors = _parent._currentVocab.WordVectors;
                var tensorD = new List<float>();
                tensorD.AddRange(wordVectors);
                // Out-of-vocab embedding vector for combining embeddings by mean.
                tensorD.AddRange(Enumerable.Repeat(0.0f, _parent._currentVocab.Dimension));
                // Out-of-vocab embedding vector for combining embeddings by element-wise min.
                tensorD.AddRange(Enumerable.Repeat(float.MaxValue, _parent._currentVocab.Dimension));
                // Out-of-vocab embedding vector for combining embeddings by element-wise max.
                tensorD.AddRange(Enumerable.Repeat(float.MinValue, _parent._currentVocab.Dimension));
                var nameD = ctx.AddInitializer(tensorD, shapeD, "WordEmbeddingWeights");

                // Allocate F, a value representing an out-of-dictionary word.
                var tensorF = _parent._currentVocab.GetNumWords();
                var nameF = ctx.AddInitializer(tensorF, "NotFoundValueComp");

                // Retrieve X, name of input.
                var nameX = srcVariableName;

                // Do label encoding. Out-of-vocab tokens will be mapped to the size of vocabulary. Because the index of vocabulary
                // is zero-based, the size of vocabulary is just greater then the max indexes computed from in-vocab tokens by one.
                var nameY = ctx.AddIntermediateVariable(null, "LabelEncodedInput", true);
                var nodeY = ctx.CreateNode("LabelEncoder", nameX, nameY, ctx.GetNodeName("LabelEncoder"));
                nodeY.AddAttribute("classes_strings", _parent._currentVocab.GetWordLabels());
                nodeY.AddAttribute("default_int64", _parent._currentVocab.GetNumWords());

                // Do steps necessary for min and max embedding vectors.

                // Map to boolean vector representing missing words. The following Equal produces 1 if a token is missing and 0 otherwise.
                var nameA = ctx.AddIntermediateVariable(null, "NotFoundValuesBool", true);
                var nodeA = ctx.CreateNode("Equal", new[] { nameY, nameF }, new[] { nameA }, ctx.GetNodeName("Equal"), "");

                // Cast the not found vector to a vector of floats.
                var nameB = ctx.AddIntermediateVariable(null, "NotFoundValuesFloat", true);
                var nodeB = ctx.CreateNode("Cast", nameA, nameB, ctx.GetNodeName("Cast"), "");
                nodeB.AddAttribute("to", 1);

                // Scale the not found vector to get the location bias for max weights.
                var nameSMax = ctx.AddIntermediateVariable(null, "ScaleMax", true);
                var nodeSMax = ctx.CreateNode("Scale", nameB, nameSMax, ctx.GetNodeName("Scale"), "");
                nodeSMax.AddAttribute("scale", 2.0);

                // Cast scaled word label locations to ints.
                var nameVMin = ctx.AddIntermediateVariable(null, "CastMin", true);
                var nodeVMin = ctx.CreateNode("Cast", nameA, nameVMin, ctx.GetNodeName("Cast"), "");
                nodeVMin.AddAttribute("to", 7);

                var nameVMax = ctx.AddIntermediateVariable(null, "CastMax", true);
                var nodeVMax = ctx.CreateNode("Cast", nameSMax, nameVMax, ctx.GetNodeName("Cast"), "");
                nodeVMax.AddAttribute("to", 7);

                // Add the scaled options back to originals. The outputs of the following Add operators are almost identical
                // the output of the previous LabelEncoder. The only difference is that out-of-vocab tokens are mapped to k+1
                // for applying ReduceMin and k+2 for applying ReduceMax so that out-of-vocab tokens do not affect embedding results at all.
                var namePMin = ctx.AddIntermediateVariable(null, "AddMin", true);
                var nodePMin = ctx.CreateNode("Add", new[] { nameY, nameVMin }, new[] { namePMin }, ctx.GetNodeName("Add"), "");

                var namePMax = ctx.AddIntermediateVariable(null, "AddMax", true);
                var nodePMax = ctx.CreateNode("Add", new[] { nameY, nameVMax }, new[] { namePMax }, ctx.GetNodeName("Add"), "");

                // Map encoded words to their embedding vectors, mapping missing ones to min/max.
                var nameGMin = ctx.AddIntermediateVariable(null, "GatheredMin", true);
                var nodeGMin = ctx.CreateNode("Gather", new[] { nameD, namePMin }, new[] { nameGMin }, ctx.GetNodeName("Gather"), "");

                var nameGMax = ctx.AddIntermediateVariable(null, "GatheredMax", true);
                var nodeGMax = ctx.CreateNode("Gather", new[] { nameD, namePMax }, new[] { nameGMax }, ctx.GetNodeName("Gather"), "");

                // Merge all embedding vectors using element-wise min/max per embedding coordinate.
                var nameJ = ctx.AddIntermediateVariable(null, "MinWeights", true);
                var nodeJ = ctx.CreateNode("ReduceMin", nameGMin, nameJ, ctx.GetNodeName("ReduceMin"), "");
                nodeJ.AddAttribute("axes", axes);

                var nameL = ctx.AddIntermediateVariable(null, "MaxWeights", true);
                var nodeL = ctx.CreateNode("ReduceMax", nameGMax, nameL, ctx.GetNodeName("ReduceMax"), "");
                nodeL.AddAttribute("axes", axes);

                // Do steps necessary for mean embedding vector.

                // Map encoded words to their embedding vectors using Gather.
                var nameW = ctx.AddIntermediateVariable(null, "GatheredMean", true);
                var nodeW = ctx.CreateNode("Gather", new[] { nameD, nameY }, new[] { nameW }, ctx.GetNodeName("Gather"), "");

                // Find the sum of the embedding vectors.
                var nameK = ctx.AddIntermediateVariable(null, "SumWeights", true);
                var nodeK = ctx.CreateNode("ReduceSum", nameW, nameK, ctx.GetNodeName("ReduceSum"), "");
                nodeK.AddAttribute("axes", axes);

                // Flip the boolean vector representing missing words to represent found words.
                var nameQ = ctx.AddIntermediateVariable(null, "FoundValuesBool", true);
                var nodeQ = ctx.CreateNode("Not", nameA, nameQ, ctx.GetNodeName("Not"), "");

                // Cast the found words vector to ints.
                var nameZ = ctx.AddIntermediateVariable(null, "FoundValuesInt", true);
                var nodeZ = ctx.CreateNode("Cast", nameQ, nameZ, ctx.GetNodeName("Cast"), "");
                nodeZ.AddAttribute("to", 6);

                // Sum the number of total found words.
                var nameR = ctx.AddIntermediateVariable(null, "NumWordsFoundInt", true);
                var nodeR = ctx.CreateNode("ReduceSum", nameZ, nameR, ctx.GetNodeName("ReduceSum"), "");
                nodeR.AddAttribute("axes", axes);

                // Cast the found words to float.
                var nameRF = ctx.AddIntermediateVariable(null, "NumWordsFoundFloat", true);
                var nodeRF = ctx.CreateNode("Cast", nameR, nameRF, ctx.GetNodeName("Cast"), "");
                nodeRF.AddAttribute("to", 1);

                // Clip the number of found words to prevent division by 0.
                var nameT = ctx.AddIntermediateVariable(null, "NumWordsClippedFloat", true);
                var nodeT = ctx.CreateNode("Clip", nameRF, nameT, ctx.GetNodeName("Clip"), "");
                nodeT.AddAttribute("min", 1.0f);

                // Divide total sum by number of words found to get the average embedding vector of the input string vector.
                var nameE = ctx.AddIntermediateVariable(null, "MeanWeights", true);
                var nodeE = ctx.CreateNode("Div", new[] { nameK, nameT }, new[] { nameE }, ctx.GetNodeName("Div"), "");

                // Concatenate the final embeddings produced by the three reduction strategies.
                var nameP = dstVariableName;
                var nodeP = ctx.CreateNode("Concat", new[] { nameJ, nameE, nameL }, new[] { nameP }, ctx.GetNodeName("Concat"), "");
                nodeP.AddAttribute("axis", 1);
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;
                return GetGetterVec(input, iinfo);
            }

            private ValueGetter<VBuffer<float>> GetGetterVec(DataViewRow input, int iinfo)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                var column = input.Schema[ColMapNewToOld[iinfo]];
                Host.Assert(column.Type is VectorType);
                Host.Assert(column.Type.GetItemType() is TextDataViewType);

                var srcGetter = input.GetGetter<VBuffer<ReadOnlyMemory<char>>>(column);
                var src = default(VBuffer<ReadOnlyMemory<char>>);
                int dimension = _parent._currentVocab.Dimension;
                float[] wordVector = new float[_parent._currentVocab.Dimension];

                return
                    (ref VBuffer<float> dst) =>
                    {
                        int deno = 0;
                        srcGetter(ref src);
                        var editor = VBufferEditor.Create(ref dst, 3 * dimension);
                        int offset = 2 * dimension;
                        for (int i = 0; i < dimension; i++)
                        {
                            editor.Values[i] = float.MaxValue;
                            editor.Values[i + dimension] = 0;
                            editor.Values[i + offset] = float.MinValue;
                        }
                        var srcValues = src.GetValues();
                        for (int word = 0; word < srcValues.Length; word++)
                        {
                            if (_parent._currentVocab.GetWordVector(in srcValues[word], wordVector))
                            {
                                deno++;
                                for (int i = 0; i < dimension; i++)
                                {
                                    float currentTerm = wordVector[i];
                                    if (editor.Values[i] > currentTerm)
                                        editor.Values[i] = currentTerm;
                                    editor.Values[dimension + i] += currentTerm;
                                    if (editor.Values[offset + i] < currentTerm)
                                        editor.Values[offset + i] = currentTerm;
                                }
                            }
                        }

                        if (deno != 0)
                            for (int index = 0; index < dimension; index++)
                                editor.Values[index + dimension] /= deno;

                        dst = editor.Commit();
                    };
            }
        }

        private static Dictionary<WordEmbeddingEstimator.PretrainedModelKind, string> _modelsMetaData = new Dictionary<WordEmbeddingEstimator.PretrainedModelKind, string>()
        {
             { WordEmbeddingEstimator.PretrainedModelKind.GloVe50D, "glove.6B.50d.txt" },
             { WordEmbeddingEstimator.PretrainedModelKind.GloVe100D, "glove.6B.100d.txt" },
             { WordEmbeddingEstimator.PretrainedModelKind.GloVe200D, "glove.6B.200d.txt" },
             { WordEmbeddingEstimator.PretrainedModelKind.GloVe300D, "glove.6B.300d.txt" },
             { WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter25D, "glove.twitter.27B.25d.txt" },
             { WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter50D, "glove.twitter.27B.50d.txt" },
             { WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter100D, "glove.twitter.27B.100d.txt" },
             { WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter200D, "glove.twitter.27B.200d.txt" },
             { WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D, "wiki.en.vec" },
             { WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding, "sentiment.emd" }
        };

        private static Dictionary<WordEmbeddingEstimator.PretrainedModelKind, int> _linesToSkipInModels = new Dictionary<WordEmbeddingEstimator.PretrainedModelKind, int>()
            { { WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D, 1 } };

        private string EnsureModelFile(IHostEnvironment env, out int linesToSkip, WordEmbeddingEstimator.PretrainedModelKind kind)
        {
            linesToSkip = 0;
            if (_modelsMetaData.ContainsKey(kind))
            {
                var modelFileName = _modelsMetaData[kind];
                if (_linesToSkipInModels.ContainsKey(kind))
                    linesToSkip = _linesToSkipInModels[kind];
                using (var ch = Host.Start("Ensuring resources"))
                {
                    string dir = kind == WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding ? Path.Combine("Text", "Sswe") : "WordVectors";
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

        private Model GetVocabularyDictionary(IHostEnvironment hostEnvironment)
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

                using (var ch = Host.Start(LoaderSignature))
                using (var pch = Host.StartProgressChannel("Building Vocabulary from Model File for Word Embeddings Transform"))
                {
                    var parsedData = new ConcurrentBag<(string key, float[] values, long lineNumber)>();
                    int skippedLinesCount = Math.Max(1, _linesToSkip);
                    var invariantCulture = _modelKind != null;
                    Parallel.ForEach(File.ReadLines(_modelFileNameWithPath).Skip(skippedLinesCount), GetParallelOptions(hostEnvironment),
                        (line, parallelState, lineNumber) =>
                        {
                            (bool isSuccess, string key, float[] values) = LineParser.ParseKeyThenNumbers(line, invariantCulture);

                            if (isSuccess)
                                parsedData.Add((key, values, lineNumber + skippedLinesCount));
                            else // we use shared state here (ch) but it's not our hot path and we don't care about unhappy-path performance
                                ch.Warning($"Parsing error while reading model file: '{_modelFileNameWithPath}', line number {lineNumber + skippedLinesCount}");
                        });

                    Model model = null;
                    foreach (var parsedLine in parsedData.OrderBy(parsedLine => parsedLine.lineNumber))
                    {
                        dimension = parsedLine.values.Length;
                        if (model == null)
                            model = new Model(dimension);
                        if (model.Dimension != dimension)
                            ch.Warning($"Dimension mismatch while reading model file: '{_modelFileNameWithPath}', line number {parsedLine.lineNumber}, expected dimension = {model.Dimension}, received dimension = {dimension}");
                        else
                            model.AddWordVector(ch, parsedLine.key, parsedLine.values);
                    }

                    // Handle first line of the embedding file separately since some embedding files including fastText have a single-line header
                    var firstLine = File.ReadLines(_modelFileNameWithPath).First();
                    string[] wordsInFirstLine = firstLine.TrimEnd().Split(' ', '\t');
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

                    _vocab[_modelFileNameWithPath] = new WeakReference<Model>(model, false);
                    return model;
                }
            }
        }

        private static ParallelOptions GetParallelOptions(IHostEnvironment hostEnvironment)
            => new ParallelOptions(); // we provide default options and let the Parallel decide
    }

    /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
    public sealed class WordEmbeddingEstimator : IEstimator<WordEmbeddingTransformer>
    {
        private readonly IHost _host;
        private readonly ColumnOptions[] _columns;
        private readonly PretrainedModelKind? _modelKind;
        private readonly string _customLookupTable;

        /// <summary>
        /// Extracts word embeddings.
        /// Output three times more values than dimension of the model specified in <paramref name="modelKind"/>
        /// First set of values represent minumum encountered values (for each dimension), second set represent average (for each dimension)
        /// and third one represent maximum encountered values (for each dimension).
        /// </summary>
        /// <param name="env">The local instance of <see cref="IHostEnvironment"/></param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="modelKind">The embeddings <see cref="PretrainedModelKind"/> to use. </param>
        internal WordEmbeddingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName = null,
           PretrainedModelKind modelKind = PretrainedModelKind.SentimentSpecificWordEmbedding)
            : this(env, modelKind, new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Extracts word embeddings.
        /// Output three times more values than dimension of the model specified in <paramref name="customModelFile"/>
        /// First set of values represent minumum encountered values (for each dimension), second set represent average (for each dimension)
        /// and third one represent maximum encountered values (for each dimension).
        /// </summary>
        /// <param name="env">The local instance of <see cref="IHostEnvironment"/></param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="customModelFile">The path of the pre-trained embeedings model to use. </param>
        /// <param name="inputColumnName">Name of the column to transform. </param>
        internal WordEmbeddingEstimator(IHostEnvironment env, string outputColumnName, string customModelFile, string inputColumnName = null)
            : this(env, customModelFile, new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName))
        {
        }

        /// <summary>
        /// Extracts word embeddings.
        /// Output three times more values than dimension of the model specified in <paramref name="modelKind"/>
        /// First set of values represent minumum encountered values (for each dimension), second set represent average (for each dimension)
        /// and third one represent maximum encountered values (for each dimension).
        /// </summary>
        /// <param name="env">The local instance of <see cref="IHostEnvironment"/></param>
        /// <param name="modelKind">The embeddings <see cref="PretrainedModelKind"/> to use. </param>
        /// <param name="columns">The array columns, and per-column configurations to extract embeedings from.</param>
        internal WordEmbeddingEstimator(IHostEnvironment env,
            PretrainedModelKind modelKind = PretrainedModelKind.SentimentSpecificWordEmbedding,
            params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(WordEmbeddingEstimator));
            _modelKind = modelKind;
            _customLookupTable = null;
            _columns = columns;
        }

        internal WordEmbeddingEstimator(IHostEnvironment env, string customModelFile, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(WordEmbeddingEstimator));
            _modelKind = null;
            _customLookupTable = customModelFile;
            _columns = columns;
        }

        /// <summary>
        /// Specifies which word embeddings to use.
        /// </summary>
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
            SentimentSpecificWordEmbedding = 9
        }
        /// <summary>
        /// Information for each column pair.
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
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <cref see="inputColumnName"/>. </param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/> <cref see="name"/> will be used as source.</param>

            public ColumnOptions(string name, string inputColumnName = null)
            {
                Contracts.CheckNonEmpty(name, nameof(name));

                Name = name;
                InputColumnName = inputColumnName ?? name;
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
                if (!(col.ItemType is TextDataViewType) || (col.Kind != SchemaShape.Column.VectorKind.VariableVector && col.Kind != SchemaShape.Column.VectorKind.Vector))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, new VectorType(TextDataViewType.Instance).ToString(), col.GetTypeString());

                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);
            }

            return new SchemaShape(result.Values);
        }

        /// <summary>
        /// Trains and returns a <see cref="WordEmbeddingTransformer"/>.
        /// </summary>
        public WordEmbeddingTransformer Fit(IDataView input)
        {
            bool customLookup = !string.IsNullOrWhiteSpace(_customLookupTable);
            WordEmbeddingTransformer transformer;
            if (customLookup)
                transformer = new WordEmbeddingTransformer(_host, _customLookupTable, _columns);
            else
                transformer = new WordEmbeddingTransformer(_host, _modelKind.Value, _columns);
            // Validate input schema.
            transformer.GetOutputSchema(input.Schema);
            return transformer;
        }
    }
}
