// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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

[assembly: LoadableClass(WordEmbeddingsTransform.Summary, typeof(IDataTransform), typeof(WordEmbeddingsTransform), typeof(WordEmbeddingsTransform.Arguments),
    typeof(SignatureDataTransform), WordEmbeddingsTransform.UserName, "WordEmbeddingsTransform", WordEmbeddingsTransform.ShortName, DocName = "transform/WordEmbeddingsTransform.md")]

[assembly: LoadableClass(typeof(WordEmbeddingsTransform), null, typeof(SignatureLoadDataTransform),
    WordEmbeddingsTransform.UserName, WordEmbeddingsTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <include file='doc.xml' path='doc/members/member[@name="WordEmbeddings"]/*' />
    public sealed class WordEmbeddingsTransform : OneToOneTransformBase
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
                loaderSignature: LoaderSignature);
        }

        private readonly PretrainedModelKind? _modelKind;
        private readonly string _modelFileNameWithPath;
        private readonly Model _currentVocab;
        private static object _embeddingsLock = new object();
        private readonly VectorType _outputType;
        private readonly bool _customLookup;
        private readonly int _linesToSkip;
        private static Dictionary<string, WeakReference<Model>> _vocab = new Dictionary<string, WeakReference<Model>>();

        private sealed class Model
        {
            private readonly BigArray<float> _wordVectors;
            private readonly NormStr.Pool _pool;
            public readonly int Dimension;

            public Model(int dimension)
            {
                Dimension = dimension;
                _wordVectors = new BigArray<float>();
                _pool = new NormStr.Pool();
            }

            public void AddWordVector(IChannel ch, string word, float[] wordVector)
            {
                ch.Assert(wordVector.Length == Dimension);
                if (_pool.Get(word) == null)
                {
                    _pool.Add(word);
                    _wordVectors.AddRange(wordVector, Dimension);
                }
            }

            public bool GetWordVector(ref DvText word, float[] wordVector)
            {
                if (word.IsNA)
                    return false;
                string rawWord = word.GetRawUnderlyingBufferInfo(out int ichMin, out int ichLim);
                NormStr str = _pool.Get(rawWord, ichMin, ichLim);
                if (str != null)
                {
                    _wordVectors.CopyTo(str.Id * Dimension, wordVector, Dimension);
                    return true;
                }
                return false;
            }
        }

        private const string RegistrationName = "WordEmbeddings";

        private const int Timeout = 10 * 60 * 1000;

        /// <summary>
        /// Public constructor corresponding to <see cref="SignatureDataTransform"/>.
        /// </summary>
        public WordEmbeddingsTransform(IHostEnvironment env, Arguments args, IDataView input)
                : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column,
                     input, TestIsTextVector)
        {
            if (args.ModelKind == null)
                args.ModelKind = PretrainedModelKind.Sswe;
            Host.CheckUserArg(!args.ModelKind.HasValue || Enum.IsDefined(typeof(PretrainedModelKind), args.ModelKind), nameof(args.ModelKind));
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _customLookup = !string.IsNullOrWhiteSpace(args.CustomLookupTable);
            if (_customLookup)
            {
                _modelKind = null;
                _modelFileNameWithPath = args.CustomLookupTable;
            }
            else
            {
                _modelKind = args.ModelKind;
                _modelFileNameWithPath = EnsureModelFile(env, out _linesToSkip, (PretrainedModelKind)_modelKind);
            }

            Host.CheckNonWhiteSpace(_modelFileNameWithPath, nameof(_modelFileNameWithPath));
            _currentVocab = GetVocabularyDictionary();
            _outputType = new VectorType(NumberType.R4, 3 * _currentVocab.Dimension);
            Metadata.Seal();
        }

        private WordEmbeddingsTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsTextVector)
        {
            Host.AssertValue(ctx);
            Host.AssertNonEmpty(Infos);
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
            _outputType = new VectorType(NumberType.R4, 3 * _currentVocab.Dimension);
            Metadata.Seal();
        }

        public static WordEmbeddingsTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            return h.Apply("Loading Model",
                ch => new WordEmbeddingsTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            SaveBase(ctx);
            ctx.Writer.WriteBoolByte(_customLookup);
            if (_customLookup)
                ctx.SaveString(_modelFileNameWithPath);
            else
                ctx.Writer.Write((uint)_modelKind);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            return _outputType;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValue(ch);
            ch.AssertValue(input);
            ch.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            var info = Infos[iinfo];
            if (!info.TypeSrc.IsVector)
            {
                throw Host.ExceptParam(nameof(input),
                    "Text input given, expects a text vector");
            }
            return GetGetterVec(ch, input, iinfo);
        }

        private ValueGetter<VBuffer<float>> GetGetterVec(IChannel ch, IRow input, int iinfo)
        {
            Host.AssertValue(ch);
            ch.AssertValue(input);
            ch.Assert(0 <= iinfo && iinfo < Infos.Length);

            var info = Infos[iinfo];
            ch.Assert(info.TypeSrc.IsVector);
            ch.Assert(info.TypeSrc.ItemType.IsText);

            var srcGetter = input.GetGetter<VBuffer<DvText>>(info.Source);
            var src = default(VBuffer<DvText>);
            int dimension = _currentVocab.Dimension;
            float[] wordVector = new float[_currentVocab.Dimension];

            return
                (ref VBuffer<float> dst) =>
                {
                    int deno = 0;
                    srcGetter(ref src);
                    var values = dst.Values;
                    Utils.EnsureSize(ref values, 3 * dimension);
                    int offset = 2 * dimension;
                    for (int i = 0; i < dimension; i++)
                    {
                        values[i] = int.MaxValue;
                        values[i + dimension] = 0;
                        values[i + offset] = int.MinValue;
                    }
                    for (int word = 0; word < src.Count; word++)
                    {
                        if (_currentVocab.GetWordVector(ref src.Values[word], wordVector))
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
                        if (model.Dimension != dimension)
                            ch.Warning($"Dimension mismatch while reading model file: '{_modelFileNameWithPath}', line number 1, expected dimension = {model.Dimension}, received dimension = {dimension}");
                        else
                        {
                            float temp;
                            string firstKey = wordsInFirstLine[0];
                            float[] firstValue = wordsInFirstLine.Skip(1).Select(x => float.TryParse(x, out temp) ? temp : Single.NaN).ToArray();
                            if (!firstValue.Contains(Single.NaN))
                                model.AddWordVector(ch, firstKey, firstValue);
                            else
                                ch.Warning($"Parsing error while reading model file: '{_modelFileNameWithPath}', line number 1");
                        }
                        pch.Checkpoint(lineNumber);
                    }
                }
                _vocab[_modelFileNameWithPath] = new WeakReference<Model>(model, false);
                return model;
            }
        }
    }
}
