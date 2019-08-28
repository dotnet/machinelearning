//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

//[assembly: LoadableClass(SvmLightLoader.Summary, typeof(SvmLightLoader), typeof(SvmLightLoader.Arguments), typeof(SignatureDataLoader),
//    SvmLightLoader.UserName, SvmLightLoader.LoaderSignature, "SvmLight", "svm", DocName = "loader/SvmLightLoader.md")]

//[assembly: LoadableClass(SvmLightLoader.Summary, typeof(SvmLightLoader), null, typeof(SignatureLoadDataLoader),
//    SvmLightLoader.UserName, SvmLightLoader.LoaderSignature)]

[assembly: LoadableClass(SvmLoader.Summary, typeof(ILegacyDataLoader), typeof(SvmLoader), typeof(SvmLoader.Options), typeof(SignatureDataLoader),
    SvmLoader.UserName, SvmLoader.LoaderSignature, "SvmLight")]

[assembly: LoadableClass(SvmLoader.Summary, typeof(ILegacyDataLoader), typeof(SvmLoader), null, typeof(SignatureLoadDataLoader),
    SvmLoader.UserName, SvmLoader.LoaderSignature)]

namespace Microsoft.ML.Data
{
    public sealed class SvmLoader : IDataLoader<IMultiStreamSource>
    {
        public sealed class Options
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the feature vectors.", ShortName = "size")]
            public int? InputSize;

            [Argument(ArgumentType.Multiple, HelpText = "Whether the features are indexed by numbers, or by feature names.", ShortName = "indices")]
            public bool FeatureIndices = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of rows used to train the feature name to index mapping transform. If unspecified, all rows will be used.", ShortName = "numxf")]
            public long? NumberOfRows;
        }

#pragma warning disable 0649 // Disable warnings about unused members. They are used through reflection.
        [CustomMappingFactoryAttribute("MapComments")]
        public sealed class Input : CustomMappingFactory<Input, CommentIndicator>
        {
            public ReadOnlyMemory<char> Text;

            public override Action<Input, CommentIndicator> GetMapping()
            {
                return MapComment;
            }

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

        public sealed class CommentIndicator
        {
            public float IsComment;
        }

        [CustomMappingFactoryAttribute("MapToKeyValueVectors")]
        public sealed class IntermediateInput : CustomMappingFactory<Input, IntermediateInput>
        {
            private static volatile char[] _seps;
            private static volatile TryParseMapper<float> _tryFloatParse;
            private static volatile TryParseMapper<long> _tryLongParse;

            [NoColumn]
            public static char[] Seps
            {
                get
                {
                    return _seps ??
                        Interlocked.CompareExchange(ref _seps, new[] { ' ', '\t' }, null);
                }
            }

            [NoColumn]
            internal static TryParseMapper<float> TryFloatParse
            {
                get
                {
                    return _tryFloatParse ??
                        Interlocked.CompareExchange(ref _tryFloatParse, Conversions.Instance.GetTryParseConversion<float>(NumberDataViewType.Single), null);
                }
            }

            [NoColumn]
            internal static TryParseMapper<long> TryLongParse
            {
                get
                {
                    return _tryLongParse ??
                        Interlocked.CompareExchange(ref _tryLongParse, Conversions.Instance.GetTryParseConversion<long>(NumberDataViewType.Int64), null);
                }
            }

            public float Label;
            public float Weight;
            public VBuffer<ReadOnlyMemory<char>> FeatureKeys;
            public VBuffer<float> FeatureValues;
            public ReadOnlyMemory<char> Comment;
            [KeyType(ulong.MaxValue - 1)]
            public ulong GroupId;

            public override Action<Input, IntermediateInput> GetMapping()
            {
                return MapInput;
            }

            public static void MapInput(Input input, IntermediateInput intermediate)
            {
                ReadOnlyMemory<char> text = ReadOnlyMemoryUtils.TrimWhiteSpace(input.Text);

                // Handle comments, if any. If no comments are present, the value for that column
                // in this row will be missing text.
                if (!ReadOnlyMemoryUtils.SplitOne(text, '#', out var left, out var right))
                    right = ReadOnlyMemory<char>.Empty;
                intermediate.Comment = right;

                var ator = ReadOnlyMemoryUtils.Split(left, Seps).GetEnumerator();

                if (!ator.MoveNext())
                {
                    intermediate.Label = float.NaN;
                    intermediate.Weight = float.NaN;
                    VBufferUtils.Clear(ref intermediate.FeatureKeys);
                    VBufferUtils.Clear(ref intermediate.FeatureValues);
                    return;
                }

                ReadOnlyMemory<char> token = ator.Current;

                // Parse the label.
                if (TryFloatParse(in token, out intermediate.Label))
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
                        if (TryFloatParse(in right, out val))
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
                        long qid;
                        if (TryLongParse(in right, out qid))
                        {
                            if (qid >= 0)
                                intermediate.GroupId = (ulong)qid + 1;
                            else
                                intermediate.GroupId = (ulong)qid;
                        }
                    }
                    else if (ReadOnlyMemoryUtils.EqualsStr("sid", left))
                    {
                        // We'll pay attention to this insofar that we'll not consider
                        // it a feature, but right now we have no learners that pay
                        // attention to so-called "slack IDs" so we'll ignore these for
                        // right now.
                    }
                    else
                    {
                        // No special handling considered these, so treat it as though it is a feature.
                        if (!TryFloatParse(in right, out val))
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

        [CustomMappingFactoryAttribute("MapTextToIndices")]
        public sealed class Indices : CustomMappingFactory<IntermediateInput, Indices>
        {
            public VBuffer<int> FeatureKeys;

            private static volatile TryParseMapper<int> _parse;
            internal static TryParseMapper<int> Parse
            {
                get
                {
                    return _parse ??
                        Interlocked.CompareExchange(ref _parse, Conversions.Instance.GetTryParseConversion<int>(NumberDataViewType.Int32), null);
                }
            }
            public static void ParseIndices(IntermediateInput input, Indices output)
            {
                var editor = VBufferEditor.Create(ref output.FeatureKeys, input.FeatureKeys.Length);
                var inputValues = input.FeatureKeys.GetValues();
                for (int i = 0; i < inputValues.Length; i++)
                {
                    if (Parse(in inputValues[i], out var index) && index > 0)
                        editor.Values[i] = index - 1;
                    else
                        editor.Values[i] = -1;
                }
                output.FeatureKeys = editor.Commit();
            }

            public override Action<IntermediateInput, Indices> GetMapping()
            {
                return ParseIndices;
            }
        }

        public sealed class IntermediateOutKeys
        {
            public VBuffer<uint> FeatureKeys;
            public VBuffer<float> FeatureValues;
        }

        public sealed class IntermediateOut
        {
            public VBuffer<int> FeatureKeys;
            public VBuffer<float> FeatureValues;
        }

        public sealed class Output
        {
            public VBuffer<float> Features;
        }
#pragma warning restore 0649

        private sealed class OutputMapper
        {
            private readonly uint _keyMax;

            public OutputMapper(int keyCount)
            {
                Contracts.Assert(keyCount > 0);
                // Leave as uint, so that comparisons against uint key values do not
                // incur any sort of implicit value conversions.
                _keyMax = (uint)keyCount;
            }

            public void Map(IntermediateOut intermediate, Output output)
            {
                MapCore(ref intermediate.FeatureKeys, ref intermediate.FeatureValues, output);
            }

            public void Map(IntermediateOutKeys intermediate, Output output)
            {
                MapCore(ref intermediate.FeatureKeys, ref intermediate.FeatureValues, output);
            }

            private void MapCore(ref VBuffer<int> keys, ref VBuffer<float> values, Output output)
            {
                var editor = VBufferEditor.Create(ref output.Features, (int)_keyMax);

                // I fully expect that these inputs will be of equal size. But I don't want to
                // throw in the event that they're not. Instead just have it be an empty vector.
                // REVIEW: Add warning and reporting for bad inputs for these.
                if (keys.Length == values.Length)
                {
                    int key = default;
                    // Both of these inputs should be dense, but still work even if they're not.
                    VBufferUtils.Densify(ref keys);
                    VBufferUtils.Densify(ref values);
                    var keysValues = keys.GetValues();
                    var valuesValues = values.GetValues();
                    for (int i = 0; i < keysValues.Length; ++i)
                    {
                        key = keysValues[i];
                        if (key < 0 || key >= _keyMax)
                            continue;
                        editor.Values[key] = valuesValues[i];
                    }
                }
                output.Features = editor.Commit();
            }

            private void MapCore(ref VBuffer<uint> keys, ref VBuffer<float> values, Output output)
            {
                var editor = VBufferEditor.Create(ref output.Features, (int)_keyMax);
                // I fully expect that these inputs will be of equal size. But I don't want to
                // throw in the event that they're not. Instead just have it be an empty vector.
                // REVIEW tfinley: Add warning and reporting for bad inputs for these.
                if (keys.Length == values.Length)
                {
                    uint key = default;
                    // Both of these inputs should be dense, but still work even if they're not.
                    VBufferUtils.Densify(ref keys);
                    VBufferUtils.Densify(ref values);
                    var keysValues = keys.GetValues();
                    var valuesValues = values.GetValues();
                    for (int i = 0; i < keys.Length; ++i)
                    {
                        key = keys.GetValues()[i];
                        if (key == 0 || key > _keyMax)
                            continue;
                        editor.Values[(int)key - 1] = valuesValues[i];
                    }
                }
                output.Features = editor.Commit();
            }
        }

        private readonly IHost _host;
        //private readonly ITransformer _parseInput;
        //private readonly ITransformer _inputTokeyAndValueVectors;
        private readonly ITransformer _keyVectorsToIndexVectors;
        private readonly bool _featureIndicesAreKeys;
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
                loaderAssemblyName: typeof(SvmLoader).Assembly.FullName);
        }

        internal SvmLoader(IHostEnvironment env, Options options = null, IMultiStreamSource dataSample = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            if (options == null)
                options = new Options();

            // First stage of the transform, effectively comment out comment lines. Comment lines are those
            // whose very first character is a '#'. When Load() is called,
            // add an NAFilter to filter out entire lines that start with '#'.
            //_parseInput = new CustomMappingTransformer<Input, InputWithCommentIndicator>(_host, Input.MapComment, "MapComments");

            // Second stage of the transform, parse out the features into a text vector of keys/indices and a text vector of values.
            //_inputTokeyAndValueVectors = new CustomMappingTransformer<Input, IntermediateInput>(_host, IntermediateInput.MapInput, "MapToKeyValueVectors");

            if (options.FeatureIndices)
            {
                // Add a transformer that parses the text into integers.
                if (options.InputSize.HasValue)
                {
                    _host.CheckUserArg(options.InputSize > 0, nameof(options.InputSize), "Maximum feature index must be positive");
                    _featureCount = (ulong)options.InputSize.Value;
                }
                else
                {
                    var data = GetData(_host, options.NumberOfRows, dataSample);
                    _featureCount = InferMax(_host, data);
                }
                _host.Assert(_featureCount <= int.MaxValue);
                _keyVectorsToIndexVectors = new CustomMappingTransformer<IntermediateInput, Indices>(
                    _host, Indices.ParseIndices, "MapTextToIndices");
            }
            else
            {
                // We need to train a ValueToKeyMappingTransformer.
                var data = GetData(_host, options.NumberOfRows, dataSample);
                _keyVectorsToIndexVectors = new ValueToKeyMappingEstimator(_host, nameof(IntermediateInput.FeatureKeys)).Fit(data);
                var keyCol = _keyVectorsToIndexVectors.GetOutputSchema(data.Schema).GetColumnOrNull(nameof(Indices.FeatureKeys));
                _host.Assert(keyCol.HasValue);
                var keyType = keyCol.Value.Type.GetItemType() as KeyDataViewType;
                _host.AssertValue(keyType);
                _featureCount = keyType.Count;
                _featureIndicesAreKeys = true;
            }

            _outputSchema = CreateOutputSchema();
        }

        private SvmLoader(IHost host, ModelLoadContext ctx)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(ctx);

            _host = host;

            // *** Binary format ***
            // bool byte: Whether the indices column type is a key type.
            // ulong: The number of features.
            // submodel: The transformer converting the indices from text to numeric/key.

            _featureIndicesAreKeys = ctx.Reader.ReadBoolByte();
            _featureCount = ctx.Reader.ReadUInt64();

            ctx.LoadModel<ITransformer, SignatureLoadModel>(_host, out _keyVectorsToIndexVectors, "KeysToIndices");
        }

        internal static SvmLoader Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoaderSignature);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model", ch => new SvmLoader(h, ctx));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bool byte: Whether the indices column type is a key type.
            // ulong: The number of features.
            // submodel: The transformer converting the indices from text to numeric/key.

            ctx.Writer.WriteBoolByte(_featureIndicesAreKeys);
            ctx.Writer.Write(_featureCount);

            ctx.SaveModel(_keyVectorsToIndexVectors, "KeysToIndices");
        }

        // These are legacy constructors needed for ComponentCatalog.
        internal static ILegacyDataLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
            => new LegacyLoader(Create(env, ctx).Load(files));
        internal static ILegacyDataLoader Create(IHostEnvironment env, Options options, IMultiStreamSource files)
            => new LegacyLoader(new SvmLoader(env, options, files).Load(files));

        private DataViewSchema CreateOutputSchema()
        {
            var data = GetData(_host, null, null);
            //var input = new LineLoader(_host, new LineLoader.Arguments(), null);
            //var inputSchema = input.Schema;
            //var schema = GetCommentMapper(_host).GetOutputSchema(inputSchema);
            //var schema = GetKeyAndValueVectorsMapper(_host).GetOutputSchema(data.Schema);
            var schema = _keyVectorsToIndexVectors.GetOutputSchema(data.Schema);
            return CreateOutputTransformer(_host, (int)_featureCount, _featureIndicesAreKeys).GetOutputSchema(schema);
        }

        private static IDataView GetData(IHostEnvironment env, long? numRows, IMultiStreamSource dataSample)
        {
            if (dataSample == null)
                throw env.Except("To use the text feature names option, a dataset must be provided");
            IDataView data = new LineLoader(env, new LineLoader.Arguments(), dataSample);
            var transformer = new CustomMappingTransformer<Input, CommentIndicator>(env, Input.MapComment, null);
            data = transformer.Transform(data);
            data = new NAFilter(env, data, columns: nameof(CommentIndicator.IsComment));
            data = new CustomMappingTransformer<Input, IntermediateInput>(env, IntermediateInput.MapInput, null).Transform(data);
            if (numRows.HasValue)
                data = SkipTakeFilter.Create(env, new SkipTakeFilter.TakeOptions() { Count = numRows.Value }, data);
            return data;
        }

        private static uint InferMax(IHostEnvironment env, IDataView view)
        {
            ulong keyMax = 0;
            var parser = Conversions.Instance.GetTryParseConversion<ulong>(NumberDataViewType.UInt64);
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
                        if (!parser(in values[i], out var val))
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
                    ch.Warning("{0} of {1} detected keys were missing or unparsable", missingCount, count + missingCount);
                if (count == 0)
                    throw ch.Except("No int parsable keys found during key transform inference");
                ch.Info("Observed max was {0}", keyMax);

                if (keyMax > int.MaxValue)
                {
                    // Similarly for missing values/misparses, warn, but don't error.
                    ch.Warning("Indices above {0} will be ignored.", int.MaxValue);
                    keyMax = int.MaxValue;
                }
            }
            return (uint)keyMax;
        }

        private static ITransformer CreateOutputTransformer(IHostEnvironment env, int keyCount, bool keyIndices)
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
                outputTransformer = new CustomMappingTransformer<IntermediateOutKeys, Output>(env,
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
            var data = GetData(_host, null, input);
            data = _keyVectorsToIndexVectors.Transform(data);
            return CreateOutputTransformer(_host, (int)_featureCount, _featureIndicesAreKeys).Transform(data);
        }

        private sealed class LegacyLoader : ILegacyDataLoader
        {
            public bool CanShuffle => _view.CanShuffle;

            public DataViewSchema Schema => _view.Schema;

            private readonly IDataView _view;

            public LegacyLoader(IDataView view)
            {
                _view = view;
            }

            public long? GetRowCount() => _view.GetRowCount();

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null) => _view.GetRowCursor(columnsNeeded, rand);

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) => _view.GetRowCursorSet(columnsNeeded, n, rand);

            void ICanSaveModel.Save(ModelSaveContext ctx)
            {
                ((ICanSaveModel)_view).Save(ctx);
            }
        }
    }

    /// <summary>
    /// This attempts to reads data in a format close to the SVM-light format, the goal being
    /// that the majority of SVM-light formatted data should be interpretable by this loader.
    /// The loader may also be different than SVM-light's parsing behavior, in the following
    /// general ways:
    ///
    /// 1. As an <see cref="IDataView"/>, vectors are required to have a logical length,
    ///    and for practical reasons it's helpful if the output of this loader has a fixed
    ///    length vector type, since few transforms and no basic learners accept features
    ///    of a variable length vector types. SVM-light had no such concept.
    /// 2. The <see cref="IDataView"/> idiom has different behavior w.r.t. parse errors.
    /// 3. The SVM-light has some restrictions in its format that are unnatural to attempt
    ///    to restrict in the concept of this loader.
    /// 4. Some common "extensions" of this format that have happened over the years are
    ///    accomodated where sensible, often supported by specifying some options.
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
    /// Labels are floating point values for regression, and for binary classification being one
    /// of {-1, 0, 1/+1}. Negative class examples are -1, positive class examples are +1, and 0
    /// indicates that the label is unknown and should be classified using semi-supervised
    /// techniques.
    ///
    /// The "0 label" semi-supervised capability was rarely used and none of our learners
    /// currently do anything like this, though it is possible we may introduce semi-supervised
    /// learners in the future. For now this loader just parses this as a single floating point
    /// value, period. (Which means, for our binary classifier learners, that 0 and -1 will both
    /// be treated identically.) If we were to support this, the natural thing would be to have
    /// an option to map 0 to NA, somehow. But practically, variants of the SVM-light format have
    /// promulgated to the point where nearly every time 0 is used, it actually refers to a
    /// negative example, so we may continue to accept this corruption as "correct" by default.
    ///
    /// The actual feature vector is specified through a series of key/value pairs. SVM-light
    /// requires that the keys be positive, increasing integers, except for three special keys:
    /// cost (we interpret as Weight), qid (we interpret as GroupId) and sid (we ignore these,
    /// but might present them as a column in the future if any of our learners implement anything
    /// resembling slack id). The value for 'cost' is float, 'qid' is a long, and 'sid' is a long
    /// that must be positive. If these keys are specified multiple times, the last one wins.
    ///
    /// SVM-light, if the tail of the value is not interpretable as a number, will ignore the tail.
    /// E.g., "5:3.14hello" will be interpreted the same as "5:3.14". I am aware of one real dataset
    /// that took advantage of this, and for now I do not support this type of thing.
    ///
    /// We do not retain the restriction on keys needing to be increasing values in our loader,
    /// due to the way we compose our feature vectors, but it will be most efficient if this policy
    /// is still followed. If it is followed a sort will not be required.
    ///
    /// This loader has the special option through the <c>xf</c> option to specify a transform,
    /// possibly trainable, to convert the raw text of the key values into the key value. The
    /// transform, whatever it is, must in addition to user specified options accept an argument
    /// of the form "column=Name" to identify a column to convert. Ideally there would be some
    /// other way to specify this other than hacking arguments. The intent of this is to allow
    /// things like string keys, a common variant of the format, but one emphatically not allowed
    /// by the original format.
    /// </summary>
    public sealed class SvmLightLoader : ILegacyDataLoader
    {
        internal sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the feature vectors.", ShortName = "size")]
            public int? InputSize;

            // REVIEW: What if we want to apply multiple transforms, somehow, to get the result?
            // This becomes somewhat more awkward.
            [Argument(ArgumentType.Multiple, HelpText = "The key transforms.", ShortName = "xf")]
            public string[] KeyTransform;

            // REVIEW tfinley: Transformer for labels?

            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of rows used to train the key mapping transform. If unspecified, all rows will be used.", ShortName = "numxf")]
            public long? NumTransform;

            // REVIEW tfinley: The problem of how this might interact with things like xf=term/hash seems troublesome to me.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to treat the cost/qid/sid keys as special, as SVM-light parsing does.", ShortName = "spec")]
            public bool SpecialKeys = true;

            // ScopeML uses this variant, seemingly unaware of the cost: special key.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Allow the format variant where label may instead be label:weight or label:weight:tag.", ShortName = "lw")]
            public bool LabelWeight;

            // ScopeML uses this variant, seemingly unaware of the cost: special key.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Include the intermediate columns in the original raw form. Useful for debugging purposes.", ShortName = "raw", Hide = true)]
            public bool RawColumns;
        }

        public bool CanShuffle { get { return _view.CanShuffle; } }

        public DataViewSchema Schema { get; }

        private readonly IHost _host;
        private readonly IDataView _view;
        private readonly IDataTransform _textToKeyInput;
        private readonly IDataTransform _textToIndicesTransform;
        private readonly char[] _seps;
        private readonly TryParseMapper<float> _tryFloatParse;
        private readonly TryParseMapper<long> _tryLongParse;
        private readonly DataViewSchema _schema;
        private readonly bool _rawColumns;

        private readonly bool _specialKeys;
        private readonly bool _labelWeight;

        internal const string Summary = "Loads text in the SVM-light format and close variants.";

        private const string _fkName = "FeatureKeys";

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

        private SvmLightLoader(IHost host)
        {
            Contracts.AssertValue(host);
            _host = host;
            _seps = new char[] { ' ', '\t' };
            _tryFloatParse = Conversions.Instance.GetTryParseConversion<float>(NumberDataViewType.Single);
            _tryLongParse = Conversions.Instance.GetTryParseConversion<long>(NumberDataViewType.Int64);
        }

        [CustomMappingFactoryAttribute("MapTextToIndices")]
        private sealed class TextToIndexMapper : CustomMappingFactory<Lambda.IntermediateInput, Lambda.Indices>
        {
            private static TryParseMapper<int> _parse = Conversions.Instance.GetTryParseConversion<int>(NumberDataViewType.Int32);

            public static void Parse(Lambda.IntermediateInput input, Lambda.Indices output)
            {
                var editor = VBufferEditor.Create(ref output.FeatureKeys, input.FeatureKeys.Length);
                var inputValues = input.FeatureKeys.GetValues();
                for (int i = 0; i < inputValues.Length; i++)
                {
                    if (_parse(in inputValues[i], out var index) && index > 0)
                        editor.Values[i] = index - 1;
                    else
                        editor.Values[i] = -1;
                }
                output.FeatureKeys = editor.Commit();
            }

            public override Action<Lambda.IntermediateInput, Lambda.Indices> GetMapping()
            {
                return Parse;
            }
        }

        internal SvmLightLoader(IHostEnvironment env, Arguments args, IMultiStreamSource data)
            : this(Contracts.CheckRef(env, nameof(env)).Register(LoaderSignature))
        {
            _host.CheckValue(args, nameof(args));
            _host.CheckValue(data, nameof(data));

            LineLoader loader = new LineLoader(_host, new LineLoader.Arguments(), data);
            _host.Assert(loader.Schema.Count == 1);
            _host.Assert(loader.Schema[0].Name == "Text");

            _specialKeys = args.SpecialKeys;
            _labelWeight = args.LabelWeight;
            _rawColumns = args.RawColumns;

            // First stage of the transform, effectively comment out comment lines. Comment lines are those
            // whose very first character is a '#'.
            IDataTransform comment = new CustomMappingTransformer<Lambda.Input, Lambda.InputWithCommentIndicator>(_host, MapComment, null).Transform(loader) as IDataTransform;
            var missingArgs = new NAFilter.Arguments() { Columns = new[] { nameof(Lambda.InputWithCommentIndicator.IsComment) } };
            comment = new NAFilter(_host, missingArgs, comment);

            // Second stage of the transform, parse out the components of the SVM-light-like format!
            _textToKeyInput = new CustomMappingTransformer<Lambda.Input, Lambda.IntermediateInput>(_host, MapInput, null).Transform(comment) as IDataTransform;

            IDataTransform textToKeySubset = _textToKeyInput;
            if (args.NumTransform.HasValue)
                textToKeySubset = SkipTakeFilter.Create(_host, new SkipTakeFilter.TakeOptions() { Count = args.NumTransform.Value }, _textToKeyInput);

            ulong keyMax = 0;
            // Third stage of the transform, instantiate the thing that translates FeatureKeys
            // from VBuffer<ReadOnlyMemory<char>> to key value with known count VBuffer<U4> (and verify it actually does this).
            if (Utils.Size(args.KeyTransform) > 0)
            {
                IDataTransform transform = textToKeySubset;
                foreach (var factory in args.KeyTransform)
                {
                    Contracts.CheckUserArg(factory != null, nameof(args.KeyTransform), "Bad transform specified.");
                    var componentFactory = CmdParser.CreateComponentFactory(typeof(IComponentFactory<IDataView, IDataTransform>), typeof(SignatureDataTransform), factory);
                    var settings = componentFactory.GetSettingsString() + " col=" + _fkName;
                    transform = ComponentCatalog.CreateInstance<IDataTransform>(_host, typeof(SignatureDataTransform), componentFactory.Name, settings, transform);
                }
                if (textToKeySubset != _textToKeyInput)
                {
                    var transView = ApplyTransformUtils.ApplyAllTransformsToData(_host, transform, _textToKeyInput, textToKeySubset);
                    // For various reasons, the above method might in other usages conceivably return something that is
                    // not actually a transform, but in this case we know it is.
                    _textToIndicesTransform = (IDataTransform)transView;
                    Contracts.AssertValue(_textToIndicesTransform);
                }
                else
                {
                    _textToIndicesTransform = transform;
                    keyMax = _textToIndicesTransform.Schema[_fkName].Type.GetItemType().GetKeyCount();
                }
            }
            else
            {
                if (args.InputSize.HasValue)
                {
                    _host.CheckUserArg(args.InputSize > 0, nameof(args.InputSize), "Maximum feature index must be positive");
                    keyMax = (ulong)args.InputSize.Value;
                }
                else
                    keyMax = InferMax(textToKeySubset);
                _host.Assert(keyMax <= int.MaxValue);
                _textToIndicesTransform = new CustomMappingTransformer<Lambda.IntermediateInput, Lambda.Indices>(
                    _host, TextToIndexMapper.Parse, "MapTextToIndices").Transform(_textToKeyInput) as IDataTransform;
            }
            // Check that the output is appropriate.
            int fkCol;
            // Check that the transform actually produced the result.
            if (!_textToIndicesTransform.Schema.TryGetColumnIndex(_fkName, out fkCol))
            {
                _host.Assert(Utils.Size(args.KeyTransform) > 0); // The default behavior should never fail in this way.
                throw _host.Except("Setup of key input conversion failed because it did not produce the required column. " +
                    "The specified transforms may not be usable in this context.");
            }
            // Bad output type is definitely possible, if a user does not specify a transform that produces keys.
            DataViewType fkType = _textToIndicesTransform.Schema[fkCol].Type;

            Init((int)keyMax, out _view, out _schema);
            Schema = _schema;
        }

        private SvmLightLoader(IHost host, ModelLoadContext ctx, IMultiStreamSource files)
            : this(host)
        {
            Contracts.AssertValue(ctx);
            Contracts.AssertValue(host);
            Contracts.AssertValue(files);

            // *** Binary format ***
            // bool byte: Whether the cost:/qid:/sid: special keys should be respected.
            // bool byte: Whether the label:weight or label:weight:tag special label format should be allowed.
            // bool byte: Whether intermediate columns used in calculation will be exposed.
            //
            // Has submodels KeyXf_00, KeyXf_01, ... to encode the key transforms.
            // At least one should exist.

            _specialKeys = ctx.Reader.ReadBoolByte();
            _labelWeight = ctx.Reader.ReadBoolByte();
            _rawColumns = ctx.Reader.ReadBoolByte();

            LineLoader loader = new LineLoader(_host, new LineLoader.Arguments(), files);

            // First stage of the transform, filter out and comment/blank lines.
            var ml = new MLContext();
            IDataTransform comment = ml.Transforms.CustomMapping<Lambda.Input, Lambda.InputWithCommentIndicator>(MapComment, null).Fit(loader).Transform(loader) as IDataTransform;
            var missingArgs = new NAFilter.Arguments() { Columns = new[] { nameof(Lambda.InputWithCommentIndicator.IsComment) } };
            comment = new NAFilter(_host, missingArgs, comment);

            // Second stage of the transform, parse out the components of the SVM-light-like format!
            _textToKeyInput = ml.Transforms.CustomMapping<Lambda.Input, Lambda.IntermediateInput>(MapInput, null).Fit(comment).Transform(comment) as IDataTransform;

            IDataTransform input = _textToKeyInput;
            IDataTransform trans = null;
            int i = 0;

            while (ctx.LoadModelOrNull<IDataTransform, SignatureLoadDataTransform>(host,
                out trans, string.Format("KeyXf_{0:00}", i++), input))
            {
                input = trans;
            }
            _host.CheckDecode(input != _textToKeyInput);
            _textToIndicesTransform = input;

            // Check that the output is appropriate.
            int fkCol;
            // This failing during decoding is possible.
            _host.CheckDecode(_textToIndicesTransform.Schema.TryGetColumnIndex(_fkName, out fkCol));
            // Bad output type is definitely possible.
            DataViewType fkType = _textToIndicesTransform.Schema[fkCol].Type;
            int keyCount = (int)fkType.GetItemType().GetKeyCount();
            _host.CheckDecode(fkType is VectorDataViewType && fkType.GetItemType() is KeyDataViewType && keyCount > 0);

            Init(keyCount, out _view, out _schema);
            Schema = _schema;
        }

        private void Init(int keyCount, out IDataView view, out DataViewSchema schema)
        {
            // Third stage of the transform, do what amounts to a weighted KeyToVector transform.
            // REVIEW tfinley: Really the KeyToVector transform should have support for weights on the keys.
            // If we add this, replace this stuff with that.
            var outputMapper = new Lambda.OutputMap(keyCount);
            // The size of the output is fixed, so just update the schema definition to reflect that.
            var schemaDef = SchemaDefinition.Create(typeof(Lambda.Output));
            _host.Assert(schemaDef.Count == 1);
            schemaDef[0].ColumnType = new VectorDataViewType(NumberDataViewType.Single, keyCount);

            var type = _textToIndicesTransform.Schema[_fkName].Type;
            var rawView = type.GetItemType() is KeyDataViewType ?
                new CustomMappingTransformer<Lambda.IntermediateOutKeys, Lambda.Output>(_host,
                outputMapper.Map, null, outputSchemaDefinition: schemaDef).Transform(_textToIndicesTransform)
                :
                new CustomMappingTransformer<Lambda.IntermediateOut, Lambda.Output>(_host,
                outputMapper.Map, null, outputSchemaDefinition: schemaDef).Transform(_textToIndicesTransform);

            if (_rawColumns)
                view = rawView;
            else
            {
                string[] toKeep = { "Label", "Weight", "GroupId", "Comment", "Features" };
                view = ColumnSelectingTransformer.CreateKeep(_host, rawView, toKeep);
            }

            schema = CreateSchema(rawView.Schema, view.Schema, keyCount);
        }

        private uint InferMax(IDataView view)
        {
            ulong keyMax = 0;
            // First do a little convert over to U8 unbounded keys, so we can infer the maximum
            // based on observations.
            //var type = new KeyDataViewType(typeof(ulong), ulong.MaxValue);
            var parser = Conversions.Instance.GetTryParseConversion<ulong>(NumberDataViewType.UInt64);

            //var ctColTemp = new TypeConvertingTransformer.Column();
            //ctColTemp.Name = ctColTemp.Source = _fkName;
            //ctColTemp.ResultType = InternalDataKind.U8;
            //ctColTemp.KeyCount = new KeyCount(ulong.MaxValue);
            //var ctArgsTemp = new TypeConvertingTransformer.Options();
            //ctArgsTemp.Columns = new TypeConvertingTransformer.Column[] { ctColTemp };
            //var temp = new TypeConvertingTransformer(_host, new TypeConvertingEstimator.ColumnOptions(_fkName, DataKind.UInt64, _fkName, new KeyCount(ulong.MaxValue))).Transform(view);
            //int fkIndex;
            bool nameResult = view.Schema.TryGetColumnIndex(_fkName, out var fkIndex);
            _host.Assert(nameResult);

            using (var ch = _host.Start("Infer key transform"))
            using (var cursor = view.GetRowCursor(view.Schema[fkIndex]))
            {
                VBuffer<ReadOnlyMemory<char>> result = default;
                var getter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(cursor.Schema[fkIndex]);

                long count = 0;
                long missingCount = 0;
                while (cursor.MoveNext())
                {
                    getter(ref result);
                    var values = result.GetValues();
                    for (int i = 0; i < values.Length; ++i)
                    {
                        if (!parser(in values[i], out var val))
                        {
                            missingCount++;
                            continue;
                        }
                        count++;
                        if (keyMax < val)
                            keyMax = val;
                    }
                    //missingCount += result.Length - values.Length;
                }
                if (missingCount > 0)
                    ch.Warning("{0} of {1} detected keys were missing or unparsable", missingCount, count + missingCount);
                if (count == 0)
                    throw ch.Except("No int parsable keys found during key transform inference");
                ch.Info("Observed max was {0}", keyMax);

                if (keyMax > int.MaxValue)
                {
                    // Similarly for missing values/misparses, warn, but don't error.
                    ch.Warning("Indices above {0} will be ignored.", int.MaxValue);
                    keyMax = int.MaxValue;
                }
            }
            return (uint)keyMax;
        }

        internal static SvmLightLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(LoaderSignature);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            h.CheckValue(files, nameof(files));

            return h.Apply("Loading Model",
                ch => new SvmLightLoader(h, ctx, files));
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bool byte: Whether the cost:/qid:/sid: special keys should be respected.
            // bool byte: Whether the label:weight or label:weight:tag special label format should be allowed.
            // bool byte: Whether intermediate columns used in calculation will be exposed.
            //
            // Has submodels KeyXf_00, KeyXf_01, ... to encode the key transforms.
            // At least one should exist.

            ctx.Writer.WriteBoolByte(_specialKeys);
            ctx.Writer.WriteBoolByte(_labelWeight);
            ctx.Writer.WriteBoolByte(_rawColumns);

            int i = 0;
            foreach (var trans in Chain())
                ctx.SaveModel(trans, string.Format("KeyXf_{0:00}", i++));
            _host.Assert(i > 0);
        }

        private IEnumerable<IDataTransform> Chain()
        {
            var stack = new Stack<IDataTransform>();
            IDataTransform trans = _textToIndicesTransform;
            for (; ; )
            {
                stack.Push(trans);
                var source = trans.Source;
                if (source == _textToKeyInput)
                    break;
                trans = source as IDataTransform;
                // The following check will fail if, for instance, we have a transform that
                // takes the _textToKeyTransform during construction, but the transform does
                // not actually identify it as a source.
                _host.Check(trans != null, "Transform chain is invalid.");
            }
            return stack;
        }

        public long? GetRowCount()
        {
            return _view.GetRowCount();
        }

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            return new Cursor(this, _view.GetRowCursor(columnsNeeded, rand));
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            DataViewRowCursor[] set = _view.GetRowCursorSet(columnsNeeded, n, rand);
            for (int i = 0; i < set.Length; ++i)
                set[i] = new Cursor(this, set[i]);
            return set;
        }

        /// <summary>
        /// Some internal classes to allow the lambda transform to do its work.
        /// </summary>
        private static class Lambda
        {
#pragma warning disable 0649 // Disable warnings about unused members. They are used through reflection.
            public sealed class Input
            {
                public ReadOnlyMemory<char> Text;
            }

            public sealed class InputWithCommentIndicator
            {
                public ReadOnlyMemory<char> Text;
                public float IsComment;
            }

            public sealed class IntermediateInput
            {
                public float Label;
                public float Weight;
                public VBuffer<ReadOnlyMemory<char>> FeatureKeys;
                public VBuffer<float> FeatureValues;
                public ReadOnlyMemory<char> Comment;
                [KeyType(ulong.MaxValue - 1)]
                public ulong GroupId;
            }

            public sealed class Indices
            {
                public VBuffer<int> FeatureKeys;
            }

            public sealed class IntermediateOutKeys
            {
                public VBuffer<uint> FeatureKeys;
                public VBuffer<float> FeatureValues;
            }

            public sealed class IntermediateOut
            {
                public VBuffer<int> FeatureKeys;
                public VBuffer<float> FeatureValues;
            }

            public sealed class Output
            {
                [NoColumn]
                public readonly BufferBuilder<float> Builder;

                public VBuffer<float> Features;

                public Output()
                {
                    // REVIEW tfinley: Might other combiners aside from "add" be of interest to anyone?
                    Builder = BufferBuilder<float>.CreateDefault();
                }
            }
#pragma warning restore 0649

            public sealed class OutputMap
            {
                private readonly uint _keyMax;

                public OutputMap(int keyCount)
                {
                    Contracts.Assert(keyCount > 0);
                    // Leave as uint, so that comparisons against uint key values do not
                    // incur any sort of implicit value conversions.
                    _keyMax = (uint)keyCount;
                }

                public void Map(IntermediateOut intermediate, Output output)
                {
                    MapCore(ref intermediate.FeatureKeys, ref intermediate.FeatureValues, output);
                }

                public void Map(IntermediateOutKeys intermediate, Output output)
                {
                    MapCore(ref intermediate.FeatureKeys, ref intermediate.FeatureValues, output);
                }

                private void MapCore(ref VBuffer<int> keys, ref VBuffer<float> values, Output output)
                {
                    output.Builder.Reset((int)_keyMax, dense: false);
                    // I fully expect that these inputs will be of equal size. But I don't want to
                    // throw in the event that they're not. Instead just have it be an empty vector.
                    // REVIEW tfinley: Add warning and reporting for bad inputs for these.
                    if (keys.Length == values.Length)
                    {
                        int key = default;
                        // Both of these inputs should be dense, but still work even if they're not.
                        VBufferUtils.Densify(ref keys);
                        VBufferUtils.Densify(ref values);
                        for (int i = 0; i < keys.Length; ++i)
                        {
                            key = keys.GetValues()[i];
                            if (key < 0 || key >= _keyMax)
                                continue;
                            output.Builder.AddFeature(key, values.GetValues()[i]);
                        }
                    }
                    output.Builder.GetResult(ref output.Features);
                }

                private void MapCore(ref VBuffer<uint> keys, ref VBuffer<float> values, Output output)
                {
                    output.Builder.Reset((int)_keyMax, dense: false);
                    // I fully expect that these inputs will be of equal size. But I don't want to
                    // throw in the event that they're not. Instead just have it be an empty vector.
                    // REVIEW tfinley: Add warning and reporting for bad inputs for these.
                    if (keys.Length == values.Length)
                    {
                        uint key = default;
                        // Both of these inputs should be dense, but still work even if they're not.
                        VBufferUtils.Densify(ref keys);
                        VBufferUtils.Densify(ref values);
                        for (int i = 0; i < keys.Length; ++i)
                        {
                            key = keys.GetValues()[i];
                            if (key == 0 || key > _keyMax)
                                continue;
                            output.Builder.AddFeature((int)key - 1, values.GetValues()[i]);
                        }
                    }
                    output.Builder.GetResult(ref output.Features);
                }
            }
        }

        private void MapComment(Lambda.Input input, Lambda.InputWithCommentIndicator output)
        {
            // We expand a bit on the SVM-light comment strategy. In SVM-light, a comment line
            // must have the # as the first character, and a totally whitespace or empty line
            // is considered a parse error. However, for the purpose of detecting comments,
            // we detect # after trimming whitespace, and also consider totally blank lines
            // "comments" instead of whitespace.
            ReadOnlyMemory<char> text = ReadOnlyMemoryUtils.TrimWhiteSpace(input.Text);
            if (text.IsEmpty || text.Span[0] == '#')
            {
                output.Text = ReadOnlyMemory<char>.Empty;
                output.IsComment = float.NaN;
            }
            else
            {
                output.Text = input.Text;
                output.IsComment = 0;
            }
        }

        private void MapInput(Lambda.Input input, Lambda.IntermediateInput intermediate)
        {
            // REVIEW tfinley: At some point we'll really want some sort of error reporting
            // mechanism like the text loader.

            ReadOnlyMemory<char> left;
            ReadOnlyMemory<char> right;

            ReadOnlyMemory<char> text = ReadOnlyMemoryUtils.TrimWhiteSpace(input.Text);
            // Handle comments, if any. If no comments are present, the value for that column
            // in this row will be missing text.
            if (!ReadOnlyMemoryUtils.SplitOne(text, '#', out left, out right))
                right = ReadOnlyMemory<char>.Empty;
            intermediate.Comment = right;

            // REVIEW tfinley: Something on TLC side that takes a predicate like char.IsWhiteSpace
            // would be preferable I think.
            var ator = ReadOnlyMemoryUtils.Split(left, _seps).GetEnumerator();

            if (!ator.MoveNext())
            {
                intermediate.Label = float.NaN;
                intermediate.Weight = float.NaN;
                VBufferUtils.Clear(ref intermediate.FeatureKeys);
                VBufferUtils.Clear(ref intermediate.FeatureValues);
                return;
            }

            ReadOnlyMemory<char> token = ator.Current;

            // Parse the label.
            if (_tryFloatParse(in token, out intermediate.Label))
                intermediate.Weight = 1; // Default weight is of course 1.
            else if (_labelWeight && ReadOnlyMemoryUtils.SplitOne(token, ':', out left, out right))
            {
                if (!_tryFloatParse(in left, out intermediate.Label))
                {
                    // Report not parsing out the label?
                    // For now do nothing.
                }
                token = right;
                if (ReadOnlyMemoryUtils.SplitOne(token, ':', out left, out right))
                {
                    // In the case of label:weight:tag, tag will take the place of the 'info' comments,
                    // even if some were present, previously. This is a variant produced by ScopeML.
                    intermediate.Comment = right;
                    token = left;
                }
                // At this point, token should be the weight.
                if (!_tryFloatParse(in token, out intermediate.Weight))
                {
                    // Report not parsing out the weight?
                }
            }
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

            //int count = 0;
            var keys = new List<ReadOnlyMemory<char>>();
            var values = new List<float>();
            //var keys = VBufferEditor.Create(ref intermediate.FeatureKeys, ArrayUtils.ArrayMaxSize);
            //var values = VBufferEditor.Create(ref intermediate.FeatureValues, int.MaxValue);
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
                if (_specialKeys)
                {
                    if (ReadOnlyMemoryUtils.EqualsStr("cost", left))
                    {
                        if (_tryFloatParse(in right, out val))
                            intermediate.Weight = val;
                        continue;
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
                        long qid;
                        if (_tryLongParse(in right, out qid))
                        {
                            if (qid >= 0)
                                intermediate.GroupId = (ulong)qid + 1;
                            else
                                intermediate.GroupId = (ulong)qid;
                        }
                        continue;
                    }
                    else if (ReadOnlyMemoryUtils.EqualsStr("sid", left))
                    {
                        // We'll pay attention to this insofar that we'll not consider
                        // it a feature, but right now we have no learners that pay
                        // attention to so-called "slack IDs" so we'll ignore these for
                        // right now.
                        continue;
                    }
                }

                // No special handling considered these, so treat it as though it is a feature.

                if (!_tryFloatParse(in right, out val))
                {
                    // Report not parsing out the value? For now silently ignore.
                    continue;
                }
                keys.Add(left);//.Values[count] = left;
                values.Add(val);//.Values[count++] = val;
            }
            intermediate.FeatureKeys = new VBuffer<ReadOnlyMemory<char>>(keys.Count, keys.ToArray());// keys.CommitTruncated(count);
            intermediate.FeatureValues = new VBuffer<float>(values.Count, values.ToArray());// values.CommitTruncated(count);
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

        private static DataViewSchema CreateSchema(DataViewSchema keysSchema, DataViewSchema schema, int keyCount)
        {
            Contracts.AssertValue(keysSchema);
            Contracts.AssertValue(schema);
            Contracts.Assert(keyCount > 0);

            int keyCol;
            int featureCol;
            bool result = keysSchema.TryGetColumnIndex(_fkName, out keyCol);
            Contracts.Assert(result);
            result = schema.TryGetColumnIndex("Features", out featureCol);
            Contracts.Assert(result);

            if (!keysSchema[keyCol].HasKeyValues())
                return schema;
            return new SchemaImpl(keysSchema, schema, keyCol, featureCol).ToSchema();
        }

        private sealed class Cursor : SynchronizedCursorBase
        {
            private readonly SvmLightLoader _parent;

            public override DataViewSchema Schema { get { return _parent.Schema; } }

            public Cursor(SvmLightLoader parent, DataViewRowCursor toWrap)
                : base(parent._host, toWrap)
            {
                Ch.Assert(toWrap.Schema.Count == parent.Schema.Count);
                _parent = parent;
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                Ch.Check(0 <= col.Index & col.Index < Schema.Count, "col");
                return Input.IsColumnActive(col);
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                Ch.Check(IsColumnActive(col), "col");
                return Input.GetGetter<TValue>(col);
            }
        }

        /// <summary>
        /// Schema implementation to add slot name metadata based on the key value
        /// metadata of the input key column that was being vectorized. This should
        /// only be used if the input key column had key value metadata.
        /// </summary>
        private sealed class SchemaImpl
        {
            private readonly DataViewSchema _keysSchema;
            private readonly DataViewSchema _parent;
            private readonly int _keyCol;
            private readonly int _featureCol;
            private readonly DataViewType _keyType;

            /// <summary>
            /// Constructs the schema.
            /// </summary>
            /// <param name="keysSchema">The schema from which we get the key values metadata.</param>
            /// <param name="schema">The schema we will wrap to present that key value metadata.
            /// Aside from presenting that additional piece of metadata the constructed schema
            /// will appear identical to this input schema.</param>
            /// <param name="keyCol">The column in <paramref name="keysSchema"/> that has the key value metadata.</param>
            /// <param name="featureCol">The column in <paramref name="schema"/> we will expose as having
            /// slot name metadata, the same as the key value metadata.</param>
            public SchemaImpl(DataViewSchema keysSchema, DataViewSchema schema, int keyCol, int featureCol)
            {
                Contracts.AssertValue(keysSchema);
                Contracts.AssertValue(schema);
                Contracts.Assert(0 <= keyCol && keyCol < keysSchema.Count);
                Contracts.Assert(0 <= featureCol && featureCol < schema.Count);
                _keysSchema = keysSchema;
                _parent = schema;
                _keyCol = keyCol;
                _featureCol = featureCol;

                _keyType = _keysSchema[_keyCol].Annotations.Schema.GetColumnOrNull(AnnotationUtils.Kinds.KeyValues)?.Type;
                Contracts.AssertValue(_keyType);
            }

            private bool TryGetColumnIndex(string name, out int col)
            {
                return _parent.TryGetColumnIndex(name, out col);
            }

            private string GetColumnName(int col)
            {
                return _parent[col].Name;
            }

            private DataViewType GetColumnType(int col)
            {
                return _parent[col].Type;
            }

            private IEnumerable<KeyValuePair<string, DataViewType>> GetMetadataTypes(int col)
            {
                var result = _parent[col].Annotations.Schema.Select(c => new KeyValuePair<string, DataViewType>(c.Name, c.Type));
                if (col == _featureCol)
                    return result.Prepend(_keyType.GetPair(AnnotationUtils.Kinds.SlotNames));
                return result;
            }

            private DataViewType GetMetadataTypeOrNull(string kind, int col)
            {
                if (col == _featureCol && kind == AnnotationUtils.Kinds.SlotNames)
                    return _keyType;
                return _parent[col].Annotations.Schema.GetColumnOrNull(kind)?.Type;
            }

            private void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                if (col == _featureCol && kind == AnnotationUtils.Kinds.SlotNames)
                {
                    _keysSchema[_keyCol].Annotations.GetValue(AnnotationUtils.Kinds.KeyValues, ref value);
                    return;
                }
                _parent[col].Annotations.GetValue(kind, ref value);
            }

            public DataViewSchema ToSchema()
            {
                var bldr = new DataViewSchema.Builder();
                foreach (var col in _parent)
                {
                    if (col.Index == _featureCol)
                    {
                        var annotationsBldr = new DataViewSchema.Annotations.Builder();
                        var getter = _keysSchema[_keyCol].Annotations.GetGetter<VBuffer<ReadOnlyMemory<char>>>(_keysSchema[_keyCol].Annotations.Schema[AnnotationUtils.Kinds.KeyValues]);
                        annotationsBldr.AddSlotNames(_parent[_featureCol].Type.GetVectorSize(), getter);
                        bldr.AddColumn(col.Name, col.Type, annotationsBldr.ToAnnotations());
                    }
                    else
                        bldr.AddColumn(col.Name, col.Type);
                }
                return bldr.ToSchema();
            }
        }
    }
}
