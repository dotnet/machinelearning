// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(SvmLightLoader.Summary, typeof(ILegacyDataLoader), typeof(SvmLightLoader), typeof(SvmLightLoader.Options), typeof(SignatureDataLoader),
    SvmLightLoader.UserName, SvmLightLoader.LoaderSignature, "SvmLight", "svm")]

[assembly: LoadableClass(SvmLightLoader.Summary, typeof(ILegacyDataLoader), typeof(SvmLightLoader), null, typeof(SignatureLoadDataLoader),
    SvmLightLoader.UserName, SvmLightLoader.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This attempts to reads data in a format close to the SVM-light format, the goal being
    /// that the majority of SVM-light formatted data should be interpretable by this loader.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// The loader may also be different than SVM-light's parsing behavior, in the following
    /// general ways:
    ///
    /// 1. As an <xref:Microsoft.ML.IDataView>, vectors are required to have a logical length,
    ///    and for practical reasons it's helpful if the output of this loader has a fixed
    ///    length vector type, since few estimators and no basic trainer estimators accept features
    ///    of a variable length vector types. SVM-light had no such concept.
    /// 2. The <xref:Microsoft.ML.IDataView> idiom has different behavior w.r.t. parse errors.
    /// 3. The SVM-light has some restrictions in its format that are unnatural to attempt
    ///    to restrict in the concept of this loader.
    /// 4. Some common "extensions" of this format that have happened over the years are
    ///    accommodated where sensible, often supported by specifying some options.
    ///
    /// The SVM-light format can be summarized here. An SVM-light file can lead with any number
    /// of lines starting with '#'. These are discarded.
    /// {label} {key}:{value} {key}:{value} ... {key}:{value}[#{comment}]
    ///
    /// Lines are not whitespace trimmed, though whitespace within the line, prior to the #
    /// comment character (if any) are ignored. SVM-light itself uses the standard C "isspace"
    /// function, while we respect only space and tab as whitespace. So, the spaces in the
    /// line above could be, say, tabs, and there could even be multiple of them in sequence.
    /// Unlike the text loader's format, for instance, there is no concept of a "blank" field
    /// having any status.
    ///
    /// The feature vector is specified through a series of key/value pairs. SVM-light
    /// requires that the keys be positive, increasing integers, except for three special keys:
    /// cost (we interpret as Weight), qid (we interpret as GroupId) and sid (we ignore these,
    /// but might present them as a column in the future if any of our learners implement anything
    /// resembling slack id). The value for 'cost' is float, 'qid' is a long, and 'sid' is a long
    /// that must be positive. If these keys are specified multiple times, the last one wins.
    ///
    /// SVM-light, if the tail of the value is not interpretable as a number, will ignore the tail.
    /// E.g., "5:3.14hello" will be interpreted the same as "5:3.14". This loader does not support
    /// this syntax.
    ///
    /// We do not retain the restriction on keys needing to be increasing values in our loader,
    /// due to the way we compose our feature vectors, but it will be most efficient if this policy
    /// is still followed. If it is followed a sort will not be required.
    ///
    /// This loader has the special option to read raw text for the keys and convert to feature
    /// indices, retaining the text key values as feature names for the resulting feature vector.
    /// The intent of this is to allow string keys, a common variant of the format, but one
    /// emphatically not allowed by the original format.
    /// ]]></format>
    /// </remarks>
    public sealed class SvmLightLoader : IDataLoader<IMultiStreamSource>
    {
        internal enum FeatureIndices
        {
            ZeroBased,
            OneBased,
            Names
        }

        internal sealed class Options
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the feature vectors.", ShortName = "size")]
            public int InputSize;

            [Argument(ArgumentType.Multiple, HelpText = "Whether the features are indexed by numbers starting at 0, by numbers starting at 1, or by feature names.", ShortName = "indices")]
            public FeatureIndices FeatureIndices = FeatureIndices.OneBased;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of rows used to train the feature name to index mapping transform. If unspecified, all rows will be used.", ShortName = "numxf")]
            public long? NumberOfRows;
        }

        /// <summary>
        /// This class is used as input for the <see cref="CustomMappingTransformer{TSrc, TDst}"/> mapping the input as a line of text.
        /// It is used by two custom mappers: one that maps the text to a float value indicating whether the line of text is a comment
        /// and the other maps the text to an object of type <see cref="IntermediateInput"/>.
        /// </summary>
        private sealed class Input
        {
#pragma warning disable 0649 // Disable warnings about unused members. They are used through reflection.
            public ReadOnlyMemory<char> Text;
#pragma warning restore 0649

            public static void MapComment(Input input, CommentIndicator output)
            {
                // We expand a bit on the SVM-light comment strategy. In SVM-light, a comment line
                // must have the # as the first character, and a totally whitespace or empty line
                // is considered a parse error. However, for the purpose of detecting comments,
                // we detect # after trimming whitespace, and also consider totally blank lines
                // "comments" instead of whitespace.
                ReadOnlyMemory<char> text = ReadOnlyMemoryUtils.TrimWhiteSpace(input.Text);
                if (text.IsEmpty || text.Span[0] == '#')
                    output.IsComment = float.NaN;
                else
                    output.IsComment = 0;
            }
        }

        // This class is used in the CustomMappingTransformer that maps a line of input to a float indicating (by the value NaN)
        // whether the line is a comment line.
        private sealed class CommentIndicator
        {
            public float IsComment;
        }

        /// <summary>
        /// This class contains the mapper that maps an <see cref="Input"/> to an <see cref="IntermediateInput"/>.
        /// The mapper parses the label and weight into floats, the group ID into ulong, the comment into a <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/>,
        /// the feature values into a vector of floats and the feature indices/names into a vector of <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/>.
        /// </summary>
        private sealed class InputMapper
        {
            private readonly char[] _seps;
            private readonly TryParseMapper<float> _tryFloatParse;
            private readonly TryParseMapper<long> _tryLongParse;

            public InputMapper()
            {
                _seps = new char[] { ' ', '\t' };
                _tryFloatParse = Conversions.DefaultInstance.GetTryParseConversion<float>(NumberDataViewType.Single);
                _tryLongParse = Conversions.DefaultInstance.GetTryParseConversion<long>(NumberDataViewType.Int64);
            }

            public void MapInput(Input input, IntermediateInput intermediate)
            {
                ReadOnlyMemory<char> text = ReadOnlyMemoryUtils.TrimWhiteSpace(input.Text);

                // Handle comments, if any. If no comments are present, the value for that column
                // in this row will be empty.
                if (!ReadOnlyMemoryUtils.SplitOne(text, '#', out var left, out var right))
                    right = ReadOnlyMemory<char>.Empty;
                intermediate.Comment = right;

                var ator = ReadOnlyMemoryUtils.Split(left, _seps).GetEnumerator();

                // Empty lines are filtered in the Input.MapComment step.
                var notEmpty = ator.MoveNext();
                Contracts.Assert(notEmpty);

                ReadOnlyMemory<char> token = ator.Current;

                // Parse the label.
                if (_tryFloatParse(in token, out intermediate.Label))
                    intermediate.Weight = 1; // Default weight is of course 1.
                else
                {
                    // Report not parsing out the label?
                    intermediate.Label = float.NaN;
                    intermediate.Weight = float.NaN;
                }

                // Group IDs are missing by default.
                intermediate.GroupId = ulong.MaxValue;

                // SVM-light "special" tokens are the following:
                // qid: Basically our group ID, with similar semantics.
                // sid: Slack ID. This is kind of only relevant to SVMs.
                // cost: Weight.

                var keys = new List<ReadOnlyMemory<char>>();
                var values = new List<float>();
                float val;
                while (ator.MoveNext())
                {
                    token = ator.Current;
                    if (!SplitOneRight(token, ':', out left, out right))
                    {
                        // Report that this appears to be a malformed token? For now just silently ignore.
                        continue;
                    }

                    // Handle the special tokens.
                    if (ReadOnlyMemoryUtils.EqualsStr("cost", left))
                    {
                        if (_tryFloatParse(in right, out val))
                            intermediate.Weight = val;
                    }
                    else if (ReadOnlyMemoryUtils.EqualsStr("qid", left))
                    {
                        // SVM-light has a query ID field, and this can be any long.
                        // That said, I've never seen anyone ever use a negative query
                        // ID, ever. If they do, I'm going to have long.MinValue map
                        // into a missing value, non-negative values map into 0 onwards
                        // as is natural, and in case there are any negative values,
                        // these will be mapped using a straight cast to ulong (so that
                        // -1 would map to ulong.MaxValue).
                        if (_tryLongParse(in right, out long qid))
                        {
                            if (qid >= 0)
                                intermediate.GroupId = (ulong)qid;
                            else
                                intermediate.GroupId = ulong.MaxValue;
                        }
                    }
                    else if (ReadOnlyMemoryUtils.EqualsStr("sid", left))
                    {
                        // We'll pay attention to this insofar that we'll not consider
                        // it a feature, but right now we have no learners that pay
                        // attention to so-called "slack IDs" so we'll ignore these for
                        // right now.
                        continue;
                    }
                    else
                    {
                        // No special handling considered these, so treat it as though it is a feature.
                        if (!_tryFloatParse(in right, out val))
                        {
                            // Report not parsing out the value? For now silently ignore.
                            continue;
                        }
                        keys.Add(left);
                        values.Add(val);
                    }
                }
                intermediate.FeatureKeys = new VBuffer<ReadOnlyMemory<char>>(keys.Count, keys.ToArray());
                intermediate.FeatureValues = new VBuffer<float>(values.Count, values.ToArray());
            }

            private static bool SplitOneRight(ReadOnlyMemory<char> memory, char separator, out ReadOnlyMemory<char> left, out ReadOnlyMemory<char> right)
            {
                if (memory.IsEmpty)
                {
                    left = memory;
                    right = default;
                    return false;
                }

                int index = memory.Span.LastIndexOf(separator);
                if (index == -1)
                {
                    left = memory;
                    right = default;
                    return false;
                }

                left = memory.Slice(0, index);
                right = memory.Slice(index + 1, memory.Length - index - 1);
                return true;
            }
        }

        // This class is used by the CustomMappingTransformer that does the initial parsing of the input.
        // The features are mapped to two fields: a vector of floats for the feature values, and a vector
        // of ReadOnlyMemory<char> for the feature indices/names.
        private sealed class IntermediateInput
        {
            public float Label;
            public float Weight;
            public VBuffer<ReadOnlyMemory<char>> FeatureKeys;
            public VBuffer<float> FeatureValues;
            public ReadOnlyMemory<char> Comment;
            [KeyType(ulong.MaxValue - 1)]
            public ulong GroupId;
        }

        /// <summary>
        /// This class is used for mapping the vector of <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/> field in an <see cref="IntermediateInput"/> object
        /// to a numeric vector of indices. This class is used in case the indices in the input file are numeric. For the case where the input file contains
        /// feature names, instead of a <see cref="CustomMappingTransformer{TSrc, TDst}"/>, we use a <see cref="ValueToKeyMappingTransformer"/>.
        /// </summary>
        private sealed class Indices
        {
            [KeyType(uint.MaxValue - 1)]
            public VBuffer<uint> FeatureKeys;
        }

        private sealed class IndexParser
        {
            private readonly uint _offset;
            private readonly uint _na;

            public IndexParser(bool zeroBased, ulong featureCount)
            {
                _offset = zeroBased ? (uint)0 : 1;
                _na = (uint)featureCount + 1;
            }

            public void ParseIndices(IntermediateInput input, Indices output)
            {
                var editor = VBufferEditor.Create(ref output.FeatureKeys, input.FeatureKeys.Length);
                var inputValues = input.FeatureKeys.GetValues();
                for (int i = 0; i < inputValues.Length; i++)
                {
                    if (Conversions.DefaultInstance.TryParse(in inputValues[i], out uint index))
                    {
                        if (index < _offset)
                        {
                            throw Contracts.Except("Encountered 0 index while parsing a 1-based dataset");
                        }
                        editor.Values[i] = index - _offset + 1;
                    }
                    else
                        throw Contracts.Except($"Encountered non-parsable index '{inputValues[i]}' while parsing dataset");
                }
                output.FeatureKeys = editor.Commit();
            }
        }

        /// <summary>
        /// This class is used by the <see cref="CustomMappingTransformer{TSrc, TDst}"/>
        /// that maps a vector of indices and a vector of values into a single <see cref="VBuffer{T}"/> of values.
        /// </summary>
        private sealed class IntermediateOut
        {
            public VBuffer<uint> FeatureKeys;
            public VBuffer<float> FeatureValues;
        }

        private sealed class Output
        {
            public VBuffer<float> Features;
        }

        /// <summary>
        /// This class contains the mapper that maps an an <see cref="IntermediateOut"/>
        /// to an <see cref="Output"/>.
        /// </summary>
        private sealed class OutputMapper
        {
            private readonly uint _keyMax;
            private readonly BufferBuilder<float> _bldr;
            private readonly BitArray _indexUsed;

            public OutputMapper(int keyCount)
            {
                Contracts.Assert(keyCount > 0);
                // Leave as uint, so that comparisons against uint key values do not
                // incur any sort of implicit value conversions.
                _keyMax = (uint)keyCount;
                _bldr = new BufferBuilder<float>(FloatAdder.Instance);
                _indexUsed = new BitArray((int)_keyMax);
            }

            public void Map(IntermediateOut intermediate, Output output)
            {
                MapCore(ref intermediate.FeatureKeys, ref intermediate.FeatureValues, output);
            }

            private void MapCore(ref VBuffer<uint> keys, ref VBuffer<float> values, Output output)
            {
                Contracts.Check(keys.Length == values.Length, "number of keys does not match number of values.");

                // Both of these inputs should be dense, but still work even if they're not.
                VBufferUtils.Densify(ref keys);
                VBufferUtils.Densify(ref values);
                var keysValues = keys.GetValues();
                var valuesValues = values.GetValues();

                // The output vector could be sparse, so we use BufferBuilder here.
                _bldr.Reset((int)_keyMax, false);
                _indexUsed.SetAll(false);
                for (int i = 0; i < keys.Length; ++i)
                {
                    var key = keysValues[i];
                    if (key == 0 || key > _keyMax)
                        continue;
                    if (_indexUsed[(int)key - 1])
                        throw Contracts.Except("Duplicate keys found in dataset");
                    _bldr.AddFeature((int)key - 1, valuesValues[i]);
                    _indexUsed[(int)key - 1] = true;
                }
                _bldr.GetResult(ref output.Features);
            }
        }

        /// <summary>
        /// This class creates an <see cref="IDataView"/> from an <see cref="IMultiStreamSource"/>, that has a single text column
        /// called "Text".
        /// </summary>
        private sealed class TextDataView : IDataView
        {
            public bool CanShuffle => false;

            public DataViewSchema Schema { get; }

            private readonly IHost _host;
            private readonly IMultiStreamSource _files;

            public TextDataView(IHostEnvironment env, IMultiStreamSource files)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(files, nameof(files));

                _host = env.Register("TextDataView");
                _files = files;

                var bldr = new DataViewSchema.Builder();
                bldr.AddColumn("Text", TextDataViewType.Instance);
                Schema = bldr.ToSchema();
            }

            public long? GetRowCount()
            {
                if (_files.Count == 0)
                    return 0;
                return null;
            }

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                _host.CheckValue(columnsNeeded, nameof(columnsNeeded));
                _host.CheckValueOrNull(rand);
                return new Cursor(this, columnsNeeded.Any());
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                _host.CheckValue(columnsNeeded, nameof(columnsNeeded));
                _host.CheckValueOrNull(rand);
                return new DataViewRowCursor[] { GetRowCursor(columnsNeeded, rand) };
            }

            private sealed class Cursor : RootCursorBase
            {
                private readonly TextDataView _parent;
                private readonly bool _isActive;
                private int _fileIdx;
                private TextReader _currReader;
                private ReadOnlyMemory<char> _text;
                private readonly ValueGetter<ReadOnlyMemory<char>> _getter;

                public override long Batch => 0;

                public override DataViewSchema Schema => _parent.Schema;

                public Cursor(TextDataView parent, bool isActive)
                    : base(parent._host)
                {
                    _parent = parent;
                    _isActive = isActive;
                    if (_parent._files.Count == 0)
                    {
                        // Rather than corrupt MoveNextCore with a bunch of custom logic for
                        // the empty file case and make that less efficient, be slightly inefficient
                        // for our zero-row case.
                        _fileIdx = -1;
                        _currReader = new StringReader("");
                        _currReader.ReadLine();
                        // Beyond this point _currReader will return null from ReadLine.
                    }
                    else
                        _currReader = _parent._files.OpenTextReader(_fileIdx);
                    if (_isActive)
                        _getter = Getter;
                }

                private void Getter(ref ReadOnlyMemory<char> val)
                {
                    Ch.Check(IsGood, "cannot call getter with cursor in its current state");
                    Ch.Assert(_isActive);
                    val = _text;
                }

                public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
                {
                    Ch.CheckParam(column.Index == 0, nameof(column));
                    Ch.CheckParam(_isActive, nameof(column), "requested column not active");

                    ValueGetter<TValue> getter = _getter as ValueGetter<TValue>;
                    if (getter == null)
                        throw Ch.Except($"Invalid TValue: '{typeof(TValue)}', " +
                            $"expected type: '{_getter.GetType().GetGenericArguments().First()}'.");
                    return getter;
                }

                public override ValueGetter<DataViewRowId> GetIdGetter()
                {
                    return
                        (ref DataViewRowId val) =>
                        {
                            Ch.Check(IsGood, "Cannot call ID getter in current state");
                            val = new DataViewRowId((ulong)Position, 0);
                        };
                }

                public override bool IsColumnActive(DataViewSchema.Column column)
                {
                    Ch.CheckParam(column.Index == 0, nameof(column));
                    return _isActive;
                }

                protected override bool MoveNextCore()
                {
                    Ch.AssertValue(_currReader);
                    Ch.Assert(-1 <= _fileIdx && _fileIdx < _parent._files.Count);

                    var count = _parent._files.Count;
                    for (; ; )
                    {
                        var line = _currReader.ReadLine();
                        if (line != null)
                        {
                            if (_isActive)
                                _text = line.AsMemory();
                            return true;
                        }
                        if (++_fileIdx == count)
                            return false;
                        Ch.AssertValue(_parent._files);
                        _currReader = _parent._files.OpenTextReader(_fileIdx);
                    }
                }
            }
        }

        private readonly IHost _host;
        private readonly ITransformer _keyVectorsToIndexVectors;
        private readonly FeatureIndices _indicesKind;
        private readonly ulong _featureCount;
        private readonly DataViewSchema _outputSchema;

        internal const string Summary = "Loads text in the SVM-light format and close variants.";
        internal const string UserName = "SVM-Light Loader";

        internal const string LoaderSignature = "SvmLightLoader";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SVMLTLDR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SvmLightLoader).Assembly.FullName);
        }

        internal SvmLightLoader(IHostEnvironment env, Options options = null, IMultiStreamSource dataSample = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            if (options == null)
                options = new Options();
            _host.CheckUserArg(options.InputSize >= 0, nameof(options.InputSize), "Maximum feature index must be positive, or 0 to infer it from the dataset");

            _indicesKind = options.FeatureIndices;
            if (_indicesKind != FeatureIndices.Names)
            {
                if (options.InputSize > 0)
                    _featureCount = (ulong)options.InputSize;
                else
                {
                    if (dataSample == null || dataSample.Count == 0)
                        throw env.Except("If the number of features is not specified, a dataset must be provided to infer it.");
                    var data = GetData(_host, options.NumberOfRows, dataSample);
                    _featureCount = InferMax(_host, data) + (ulong)(_indicesKind == FeatureIndices.ZeroBased ? 1 : 0);
                }
                _host.Assert(_featureCount <= int.MaxValue);
            }
            else
            {
                // We need to train a ValueToKeyMappingTransformer.
                if (dataSample == null || dataSample.Count == 0)
                    throw env.Except("To use the text feature names option, a dataset must be provided");

                var data = GetData(_host, options.NumberOfRows, dataSample);
                _keyVectorsToIndexVectors = new ValueToKeyMappingEstimator(_host, nameof(IntermediateInput.FeatureKeys)).Fit(data);
                var keyCol = _keyVectorsToIndexVectors.GetOutputSchema(data.Schema).GetColumnOrNull(nameof(Indices.FeatureKeys));
                _host.Assert(keyCol.HasValue);
                var keyType = keyCol.Value.Type.GetItemType() as KeyDataViewType;
                _host.AssertValue(keyType);
                _featureCount = keyType.Count;
            }

            _outputSchema = CreateOutputSchema();
        }

        private SvmLightLoader(IHost host, ModelLoadContext ctx)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(ctx);

            _host = host;

            // *** Binary format ***
            // int: Whether the indices column type is a key type, integer starting with 0 or integet starting with 1.
            // ulong: The number of features.
            // submodel: The transformer converting the indices from text to numeric/key.

            _indicesKind = (FeatureIndices)ctx.Reader.ReadInt32();
            _featureCount = ctx.Reader.ReadUInt64();

            ctx.LoadModelOrNull<ITransformer, SignatureLoadModel>(_host, out _keyVectorsToIndexVectors, "KeysToIndices");
        }

        internal static SvmLightLoader Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoaderSignature);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model", ch => new SvmLightLoader(h, ctx));
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: Whether the indices column type is a key type, integer starting with 0 or integet starting with 1.
            // ulong: The number of features.
            // submodel: The transformer converting the indices from text to numeric/key.

            ctx.Writer.Write((int)_indicesKind);
            ctx.Writer.Write(_featureCount);

            if (_keyVectorsToIndexVectors != null)
                ctx.SaveModel(_keyVectorsToIndexVectors, "KeysToIndices");
        }

        private DataViewSchema CreateOutputSchema()
        {
            var data = GetData(_host, null, new MultiFileSource(null));
            var indexParser = new IndexParser(_indicesKind == FeatureIndices.ZeroBased, _featureCount);
            var schemaDef = SchemaDefinition.Create(typeof(Indices));
            schemaDef[nameof(Indices.FeatureKeys)].ColumnType = new KeyDataViewType(typeof(uint), _featureCount);
            var keyVectorsToIndexVectors = _keyVectorsToIndexVectors ??
                new CustomMappingTransformer<IntermediateInput, Indices>(_host, indexParser.ParseIndices, null);
            var schema = keyVectorsToIndexVectors.GetOutputSchema(data.Schema);
            return CreateOutputTransformer(_host, (int)_featureCount,
                _indicesKind == FeatureIndices.Names, schema).GetOutputSchema(schema);
        }

        private static IDataView GetData(IHostEnvironment env, long? numRows, IMultiStreamSource dataSample)
        {
            IDataView data = new TextDataView(env, dataSample);

            // First stage of the transform, effectively comment out comment lines. Comment lines are those
            // whose very first character is a '#'. We add an NAFilter to filter out entire lines that start with '#'.
            // REVIEW: When ML.NET supports custom mappings for filters, we can replace this stage with a custom filter.

            var transformer = new CustomMappingTransformer<Input, CommentIndicator>(env, Input.MapComment, null);
            data = transformer.Transform(data);
            data = new NAFilter(env, data, columns: nameof(CommentIndicator.IsComment));

            // Second stage of the transform, parse out the features into a text vector of keys/indices and a text vector of values.
            // If we are loading the data for training a KeyToValueTransformer, users can specify the number of rows to train on.
            var inputMapper = new InputMapper();
            data = new CustomMappingTransformer<Input, IntermediateInput>(env, inputMapper.MapInput, null).Transform(data);
            if (numRows.HasValue)
                data = SkipTakeFilter.Create(env, new SkipTakeFilter.TakeOptions() { Count = numRows.Value }, data);
            return data;
        }

        private static uint InferMax(IHostEnvironment env, IDataView view)
        {
            ulong keyMax = 0;
            var parser = Conversions.DefaultInstance.GetTryParseConversion<ulong>(NumberDataViewType.UInt64);
            var col = view.Schema.GetColumnOrNull(nameof(IntermediateInput.FeatureKeys));
            env.Assert(col.HasValue);

            using (var ch = env.Start("Infer key transform"))
            using (var cursor = view.GetRowCursor(col.Value))
            {
                VBuffer<ReadOnlyMemory<char>> result = default;
                var getter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(col.Value);

                long count = 0;
                long missingCount = 0;
                while (cursor.MoveNext())
                {
                    getter(ref result);
                    var values = result.GetValues();
                    for (int i = 0; i < values.Length; ++i)
                    {
                        if (!parser(in values[i], out var val) || val > int.MaxValue)
                        {
                            missingCount++;
                            continue;
                        }
                        count++;
                        if (keyMax < val)
                            keyMax = val;
                    }
                }
                if (missingCount > 0)
                    ch.Warning($"{missingCount} of {count + missingCount} detected keys were missing, unparsable or greater than {int.MaxValue}");
                if (count == 0)
                    throw ch.Except("No int parsable keys found during key transform inference");
                ch.Info("Observed max was {0}", keyMax);
            }
            return (uint)keyMax;
        }

        private static ITransformer CreateOutputTransformer(IHostEnvironment env, int keyCount, bool keyIndices, DataViewSchema inputSchema)
        {
            // Third stage of the transform, do what amounts to a weighted KeyToVector transform.
            // REVIEW: Really the KeyToVector transform should have support for weights on the keys.
            // If we add this, replace this stuff with that.
            var outputMapper = new OutputMapper(keyCount);
            // The size of the output is fixed, so just update the schema definition to reflect that.
            var schemaDef = SchemaDefinition.Create(typeof(Output));
            env.Assert(schemaDef.Count == 1);
            schemaDef[0].ColumnType = new VectorDataViewType(NumberDataViewType.Single, keyCount);

            ITransformer outputTransformer;
            if (keyIndices)
            {
                var col = inputSchema[nameof(Indices.FeatureKeys)];
                var keyValuesCol = col.Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues);
                if (keyValuesCol.HasValue)
                {
                    VBuffer<ReadOnlyMemory<char>> keyValues = default;
                    col.Annotations.GetValue(AnnotationUtils.Kinds.KeyValues, ref keyValues);
                    schemaDef[0].AddAnnotation(AnnotationUtils.Kinds.SlotNames, keyValues, keyValuesCol.Value.Type);
                }
                outputTransformer = new CustomMappingTransformer<IntermediateOut, Output>(env,
                    outputMapper.Map, null, outputSchemaDefinition: schemaDef);
            }
            else
            {
                outputTransformer = new CustomMappingTransformer<IntermediateOut, Output>(env,
                    outputMapper.Map, null, outputSchemaDefinition: schemaDef);
            }

            string[] toKeep = { "Label", "Weight", "GroupId", "Comment", "Features" };
            return outputTransformer.Append(new ColumnSelectingTransformer(env, toKeep, null));
        }

        public DataViewSchema GetOutputSchema() => _outputSchema;

        public IDataView Load(IMultiStreamSource input)
        {
            _host.CheckValue(input, nameof(input));

            var data = GetData(_host, null, input);
            var indexParser = new IndexParser(_indicesKind == FeatureIndices.ZeroBased, _featureCount);
            var keyVectorsToIndexVectors = _keyVectorsToIndexVectors ??
                new CustomMappingTransformer<IntermediateInput, Indices>(_host, indexParser.ParseIndices, null);
            data = keyVectorsToIndexVectors.Transform(data);
            return CreateOutputTransformer(_host, (int)_featureCount, _indicesKind == FeatureIndices.Names, data.Schema).Transform(data);
        }

        // These are legacy constructors needed for ComponentCatalog.
        internal static ILegacyDataLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
        {
            var svmLoader = Create(env, ctx);
            return new LegacyLoader(svmLoader, svmLoader.Load(files));
        }
        internal static ILegacyDataLoader Create(IHostEnvironment env, Options options, IMultiStreamSource files)
        {
            var svmLoader = new SvmLightLoader(env, options, files);
            return new LegacyLoader(svmLoader, svmLoader.Load(files));
        }

        private sealed class LegacyLoader : ILegacyDataLoader
        {
            public bool CanShuffle => _view.CanShuffle;

            public DataViewSchema Schema => _view.Schema;

            private readonly IDataView _view;
            private readonly SvmLightLoader _loader;

            public LegacyLoader(SvmLightLoader loader, IDataView view)
            {
                _loader = loader;
                _view = view;
            }

            public long? GetRowCount() => _view.GetRowCount();

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null) => _view.GetRowCursor(columnsNeeded, rand);

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) => _view.GetRowCursorSet(columnsNeeded, n, rand);

            void ICanSaveModel.Save(ModelSaveContext ctx)
            {
                ((ICanSaveModel)_loader).Save(ctx);
            }
        }
    }
}
