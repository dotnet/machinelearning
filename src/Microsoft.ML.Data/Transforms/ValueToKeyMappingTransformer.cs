﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;
using Microsoft.ML.Model.Onnx;
using Microsoft.ML.Model.Pfa;
using Microsoft.ML.Transforms.Conversions;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(ValueToKeyMappingTransformer.Summary, typeof(IDataTransform), typeof(ValueToKeyMappingTransformer),
    typeof(ValueToKeyMappingTransformer.Arguments), typeof(SignatureDataTransform),
    ValueToKeyMappingTransformer.UserName, "Term", "AutoLabel", "TermTransform", "AutoLabelTransform", DocName = "transform/TermTransform.md")]

[assembly: LoadableClass(ValueToKeyMappingTransformer.Summary, typeof(IDataTransform), typeof(ValueToKeyMappingTransformer), null, typeof(SignatureLoadDataTransform),
    ValueToKeyMappingTransformer.UserName, ValueToKeyMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(ValueToKeyMappingTransformer.Summary, typeof(ValueToKeyMappingTransformer), null, typeof(SignatureLoadModel),
    ValueToKeyMappingTransformer.UserName, ValueToKeyMappingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ValueToKeyMappingTransformer), null, typeof(SignatureLoadRowMapper),
    ValueToKeyMappingTransformer.UserName, ValueToKeyMappingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Conversions
{
    // TermTransform builds up term vocabularies (dictionaries).
    // Notes:
    // * Each column builds/uses exactly one "vocabulary" (dictionary).
    // * Output columns are KeyType-valued.
    // * The Key value is the one-based index of the item in the dictionary.
    // * Not found is assigned the value zero.
    /// <include file='doc.xml' path='doc/members/member[@name="TextToKey"]/*' />
    public sealed partial class ValueToKeyMappingTransformer : OneToOneTransformerBase
    {
        public abstract class ColumnBase : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of terms to keep when auto-training", ShortName = "max")]
            public int? MaxNumTerms;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma separated list of terms", Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string Terms;

            [Argument(ArgumentType.AtMostOnce, HelpText = "List of terms", Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string[] Term;

            [Argument(ArgumentType.AtMostOnce, HelpText = "How items should be ordered when vectorized. By default, they will be in the order encountered. " +
                "If by value items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').")]
            public SortOrder? Sort;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether key value metadata should be text, regardless of the actual input type", ShortName = "textkv", Hide = true)]
            public bool? TextKeyValues;

            protected override bool TryUnparseCore(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                // REVIEW: This pattern isn't robust enough. If a new field is added, this code needs
                // to be updated accordingly, or it will break. The only protection we have against this
                // is unit tests....
                if (MaxNumTerms != null || !string.IsNullOrEmpty(Terms) || Sort != null || TextKeyValues != null)
                    return false;
                return base.TryUnparseCore(sb);
            }
        }

        public sealed class Column : ColumnBase
        {
            public static Column Parse(string str)
            {
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

        /// <summary>
        /// Controls how the order of the output keys.
        /// </summary>
        public enum SortOrder : byte
        {
            Occurrence = 0,
            Value = 1,
            // REVIEW: We can think about having a frequency order option. What about
            // other things, like case insensitive (where appropriate), culturally aware, etc.?
        }

        public abstract class ArgumentsBase : TransformInputBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of terms to keep per column when auto-training", ShortName = "max", SortOrder = 5)]
            public int MaxNumTerms = ValueToKeyMappingEstimator.Defaults.MaxNumTerms;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Comma separated list of terms", SortOrder = 105, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string Terms;

            [Argument(ArgumentType.AtMostOnce, HelpText = "List of terms", SortOrder = 106, Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly)]
            public string[] Term;

            [Argument(ArgumentType.AtMostOnce, IsInputFileName = true, HelpText = "Data file containing the terms", ShortName = "data", SortOrder = 110, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string DataFile;

            [Argument(ArgumentType.Multiple, HelpText = "Data loader", NullName = "<Auto>", SortOrder = 111, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IMultiStreamSource, IDataLoader> Loader;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the text column containing the terms", ShortName = "termCol", SortOrder = 112, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly)]
            public string TermsColumn;

            // REVIEW: The behavior of sorting when doing term on an input key value is to sort on the key numbers themselves,
            // that is, to maintain the relative order of the key values. The alternative is that, for these, we would sort on the key
            // value metadata, if present. Both sets of behavior seem potentially valuable.

            // REVIEW: Should we always sort? Opinions are mixed. See work item 7797429.
            [Argument(ArgumentType.AtMostOnce, HelpText = "How items should be ordered when vectorized. By default, they will be in the order encountered. " +
                "If by value items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').", SortOrder = 113)]
            public SortOrder Sort = ValueToKeyMappingEstimator.Defaults.Sort;

            // REVIEW: Should we do this here, or correct the various pieces of code here and in MRS etc. that
            // assume key-values will be string? Once we correct these things perhaps we can see about removing it.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether key value metadata should be text, regardless of the actual input type", ShortName = "textkv", SortOrder = 114, Hide = true)]
            public bool TextKeyValues;
        }

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        internal sealed class ColInfo
        {
            public readonly string Name;
            public readonly string Source;
            public readonly ColumnType TypeSrc;

            public ColInfo(string name, string source, ColumnType type)
            {
                Name = name;
                Source = source;
                TypeSrc = type;
            }
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        public class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;
            public readonly SortOrder Sort;
            public readonly int MaxNumTerms;
            public readonly string[] Term;
            public readonly bool TextKeyValues;

            protected internal string Terms { get; set; }

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="input">Name of input column.</param>
            /// <param name="output">Name of the column resulting from the transformation of <paramref name="input"/>. Null means <paramref name="input"/> is replaced.</param>
            /// <param name="maxNumTerms">Maximum number of terms to keep per column when auto-training.</param>
            /// <param name="sort">How items should be ordered when vectorized. If <see cref="SortOrder.Occurrence"/> choosen they will be in the order encountered.
            /// If <see cref="SortOrder.Value"/>, items are sorted according to their default comparison, for example, text sorting will be case sensitive (for example, 'A' then 'Z' then 'a').</param>
            /// <param name="term">List of terms.</param>
            /// <param name="textKeyValues">Whether key value metadata should be text, regardless of the actual input type.</param>
            public ColumnInfo(string input, string output = null,
                int maxNumTerms = ValueToKeyMappingEstimator.Defaults.MaxNumTerms,
                SortOrder sort = ValueToKeyMappingEstimator.Defaults.Sort,
                string[] term = null,
                bool textKeyValues = false
                )
            {
                Contracts.CheckNonWhiteSpace(input, nameof(input));
                Input = input;
                Output = output ?? input;
                Sort = sort;
                MaxNumTerms = maxNumTerms;
                Term = term;
                TextKeyValues = textKeyValues;
            }
        }
        [BestFriend]
        internal const string Summary = "Converts input values (words, numbers, etc.) to index in a dictionary.";
        [BestFriend]
        internal const string UserName = "Term Transform";
        [BestFriend]
        internal const string LoaderSignature = "TermTransform";
        [BestFriend]
        internal const string FriendlyName = "To Key";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TERMTRNF",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Dropped sizeof(Float)
                verWrittenCur: 0x00010003, // Generalize to multiple types beyond text
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ValueToKeyMappingTransformer).Assembly.FullName);
        }

        private const uint VerNonTextTypesSupported = 0x00010003;
        private const uint VerManagerNonTextTypesSupported = 0x00010002;

        internal const string TermManagerLoaderSignature = "TermManager";
        private static volatile MemoryStreamPool _codecFactoryPool;
        private volatile CodecFactory _codecFactory;

        private CodecFactory CodecFactory
        {
            get
            {
                if (_codecFactory == null)
                {
                    Interlocked.CompareExchange(ref _codecFactoryPool, new MemoryStreamPool(), null);
                    Interlocked.CompareExchange(ref _codecFactory, new CodecFactory(Host, _codecFactoryPool), null);
                }
                Host.Assert(_codecFactory != null);
                return _codecFactory;
            }
        }
        private static VersionInfo GetTermManagerVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TERM MAN",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Generalize to multiple types beyond text
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: TermManagerLoaderSignature,
                loaderAssemblyName: typeof(ValueToKeyMappingTransformer).Assembly.FullName);
        }

        private readonly TermMap[] _unboundMaps;
        private readonly bool[] _textMetadata;
        private const string RegistrationName = "Term";

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        private string TestIsKnownDataKind(ColumnType type)
        {
            VectorType vectorType = type as VectorType;
            ColumnType itemType = vectorType?.ItemType ?? type;

            if (itemType is KeyType || itemType.IsStandardScalar())
                return null;
            return "standard type or a vector of standard type";
        }

        private ColInfo[] CreateInfos(Schema inputSchema)
        {
            Host.AssertValue(inputSchema);
            var infos = new ColInfo[ColumnPairs.Length];
            for (int i = 0; i < ColumnPairs.Length; i++)
            {
                if (!inputSchema.TryGetColumnIndex(ColumnPairs[i].input, out int colSrc))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[i].input);
                var type = inputSchema[colSrc].Type;
                string reason = TestIsKnownDataKind(type);
                if (reason != null)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[i].input, reason, type.ToString());
                infos[i] = new ColInfo(ColumnPairs[i].output, ColumnPairs[i].input, type);
            }
            return infos;
        }

        internal ValueToKeyMappingTransformer(IHostEnvironment env, IDataView input,
            params ColumnInfo[] columns) :
            this(env, input, columns, null, null, null)
        { }

        internal ValueToKeyMappingTransformer(IHostEnvironment env, IDataView input,
            ColumnInfo[] columns,
            string file = null, string termsColumn = null,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            using (var ch = Host.Start("Training"))
            {
                var infos = CreateInfos(input.Schema);
                _unboundMaps = Train(Host, ch, infos, file, termsColumn, loaderFactory, columns, input);
                _textMetadata = new bool[_unboundMaps.Length];
                for (int iinfo = 0; iinfo < columns.Length; ++iinfo)
                {
                    _textMetadata[iinfo] = columns[iinfo].TextKeyValues;
                }
                ch.Assert(_unboundMaps.Length == columns.Length);
            }
        }

        [BestFriend]
        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));
            var cols = new ColumnInfo[args.Column.Length];
            using (var ch = env.Start("ValidateArgs"))
            {
                if ((args.Term != null || !string.IsNullOrEmpty(args.Terms)) &&
                  (!string.IsNullOrWhiteSpace(args.DataFile) || args.Loader != null ||
                      !string.IsNullOrWhiteSpace(args.TermsColumn)))
                {
                    ch.Warning("Explicit term list specified. Data file arguments will be ignored");
                }
                if (!Enum.IsDefined(typeof(SortOrder), args.Sort))
                    throw ch.ExceptUserArg(nameof(args.Sort), "Undefined sorting criteria '{0}' detected", args.Sort);

                for (int i = 0; i < cols.Length; i++)
                {
                    var item = args.Column[i];
                    var sortOrder = item.Sort ?? args.Sort;
                    if (!Enum.IsDefined(typeof(SortOrder), sortOrder))
                        throw env.ExceptUserArg(nameof(args.Sort), "Undefined sorting criteria '{0}' detected for column '{1}'", sortOrder, item.Name);

                    cols[i] = new ColumnInfo(item.Source ?? item.Name,
                        item.Name,
                        item.MaxNumTerms ?? args.MaxNumTerms,
                        sortOrder,
                        item.Term,
                        item.TextKeyValues ?? args.TextKeyValues);
                    cols[i].Terms = item.Terms ?? args.Terms;
                };
            }
            return new ValueToKeyMappingTransformer(env, input, cols, args.DataFile, args.TermsColumn, args.Loader).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static ValueToKeyMappingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);

            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new ValueToKeyMappingTransformer(host, ctx);
        }

        private ValueToKeyMappingTransformer(IHost host, ModelLoadContext ctx)
           : base(host, ctx)
        {
            var columnsLength = ColumnPairs.Length;

            if (ctx.Header.ModelVerWritten >= VerNonTextTypesSupported)
                _textMetadata = ctx.Reader.ReadBoolArray(columnsLength);
            else
                _textMetadata = new bool[columnsLength]; // No need to set in this case. They're all text.

            const string dir = "Vocabulary";
            var termMap = new TermMap[columnsLength];
            bool b = ctx.TryProcessSubModel(dir,
            c =>
            {
                // *** Binary format ***
                // int: number of term maps (should equal number of columns)
                // for each term map:
                //   byte: code identifying the term map type (0 text, 1 codec)
                //   <data>: type specific format, see TermMap save/load methods

                host.CheckValue(c, nameof(ctx));
                c.CheckAtModel(GetTermManagerVersionInfo());
                int cmap = c.Reader.ReadInt32();
                host.CheckDecode(cmap == columnsLength);
                if (c.Header.ModelVerWritten >= VerManagerNonTextTypesSupported)
                {
                    for (int i = 0; i < columnsLength; ++i)
                        termMap[i] = TermMap.Load(c, host, CodecFactory);
                }
                else
                {
                    for (int i = 0; i < columnsLength; ++i)
                        termMap[i] = TermMap.TextImpl.Create(c, host);
                }
            });
#pragma warning disable MSML_NoMessagesForLoadContext // Vaguely useful.
            if (!b)
                throw host.ExceptDecode("Missing {0} model", dir);
#pragma warning restore MSML_NoMessagesForLoadContext
            _unboundMaps = termMap;
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        /// <summary>
        /// Utility method to create the file-based <see cref="TermMap"/>.
        /// </summary>
        private static TermMap CreateFileTermMap(IHostEnvironment env, IChannel ch, string file, string termsColumn,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory, Builder bldr)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(env);
            ch.Assert(!string.IsNullOrWhiteSpace(file));
            ch.AssertValue(bldr);

            // First column using the file.
            string src = termsColumn;
            IMultiStreamSource fileSource = new MultiFileSource(file);

            // If the user manually specifies a loader, or this is already a pre-processed binary
            // file, then we assume the user knows what they're doing and do not attempt to convert
            // to the desired type ourselves.
            bool autoConvert = false;
            IDataView termData;
            if (loaderFactory != null)
            {
                termData = loaderFactory.CreateComponent(env, fileSource);
            }
            else
            {
                // Determine the default loader from the extension.
                var ext = Path.GetExtension(file);
                bool isBinary = string.Equals(ext, ".idv", StringComparison.OrdinalIgnoreCase);
                bool isTranspose = string.Equals(ext, ".tdv", StringComparison.OrdinalIgnoreCase);
                if (isBinary || isTranspose)
                {
                    ch.Assert(isBinary != isTranspose);
                    ch.CheckUserArg(!string.IsNullOrWhiteSpace(src), nameof(termsColumn),
                        "Must be specified");
                    if (isBinary)
                        termData = new BinaryLoader(env, new BinaryLoader.Arguments(), fileSource);
                    else
                    {
                        ch.Assert(isTranspose);
                        termData = new TransposeLoader(env, new TransposeLoader.Arguments(), fileSource);
                    }
                }
                else
                {
                    if (!string.IsNullOrWhiteSpace(src))
                    {
                        ch.Warning(
                            "{0} should not be specified when default loader is TextLoader. Ignoring {0}={1}",
                            nameof(Arguments.TermsColumn), src);
                    }
                    termData = new TextLoader(env,
                        columns: new[] { new TextLoader.Column("Term", DataKind.TX, 0) },
                        dataSample: fileSource)
                        .Read(fileSource);
                    src = "Term";
                    autoConvert = true;
                }
            }
            ch.AssertNonEmpty(src);

            int colSrc;
            if (!termData.Schema.TryGetColumnIndex(src, out colSrc))
                throw ch.ExceptUserArg(nameof(termsColumn), "Unknown column '{0}'", src);
            var typeSrc = termData.Schema[colSrc].Type;
            if (!autoConvert && !typeSrc.Equals(bldr.ItemType))
                throw ch.ExceptUserArg(nameof(termsColumn), "Must be of type '{0}' but was '{1}'", bldr.ItemType, typeSrc);

            using (var cursor = termData.GetRowCursor(termData.Schema[colSrc]))
            using (var pch = env.StartProgressChannel("Building term dictionary from file"))
            {
                var header = new ProgressHeader(new[] { "Total Terms" }, new[] { "examples" });
                var trainer = Trainer.Create(cursor, colSrc, autoConvert, int.MaxValue, bldr);
                double rowCount = termData.GetRowCount() ?? double.NaN;
                long rowCur = 0;
                pch.SetHeader(header,
                    e =>
                    {
                        e.SetProgress(0, rowCur, rowCount);
                        // Purely feedback for the user. That the other thread might be
                        // working in the background is not a problem.
                        e.SetMetric(0, trainer.Count);
                    });
                while (cursor.MoveNext() && trainer.ProcessRow())
                    rowCur++;
                if (trainer.Count == 0)
                    ch.Warning("Term map loaded from file resulted in an empty map.");
                pch.Checkpoint(trainer.Count, rowCur);
                return trainer.Finish();
            }
        }

        /// <summary>
        /// This builds the <see cref="TermMap"/> instances per column.
        /// </summary>
        private static TermMap[] Train(IHostEnvironment env, IChannel ch, ColInfo[] infos,
            string file, string termsColumn,
            IComponentFactory<IMultiStreamSource, IDataLoader> loaderFactory, ColumnInfo[] columns, IDataView trainingData)
        {
            Contracts.AssertValue(env);
            env.AssertValue(ch);
            ch.AssertValue(infos);
            ch.AssertValue(columns);
            ch.AssertValue(trainingData);

            TermMap termsFromFile = null;
            var termMap = new TermMap[infos.Length];
            int[] lims = new int[infos.Length];
            int trainsNeeded = 0;
            HashSet<int> toTrain = null;

            for (int iinfo = 0; iinfo < infos.Length; iinfo++)
            {
                // First check whether we have a terms argument, and handle it appropriately.
                var terms = columns[iinfo].Terms.AsMemory();
                var termsArray = columns[iinfo].Term;

                terms = ReadOnlyMemoryUtils.TrimSpaces(terms);
                if (!terms.IsEmpty || (termsArray != null && termsArray.Length > 0))
                {
                    // We have terms! Pass it in.
                    var sortOrder = columns[iinfo].Sort;
                    var bldr = Builder.Create(infos[iinfo].TypeSrc, sortOrder);
                    if (!terms.IsEmpty)
                        bldr.ParseAddTermArg(ref terms, ch);
                    else
                        bldr.ParseAddTermArg(termsArray, ch);
                    termMap[iinfo] = bldr.Finish();
                }
                else if (!string.IsNullOrWhiteSpace(file))
                {
                    // First column using this file.
                    if (termsFromFile == null)
                    {
                        var bldr = Builder.Create(infos[iinfo].TypeSrc, columns[iinfo].Sort);
                        termsFromFile = CreateFileTermMap(env, ch, file, termsColumn, loaderFactory, bldr);
                    }
                    if (!termsFromFile.ItemType.Equals(infos[iinfo].TypeSrc.GetItemType()))
                    {
                        // We have no current plans to support re-interpretation based on different column
                        // type, not only because it's unclear what realistic customer use-cases for such
                        // a complicated feature would be, and also because it's difficult to see how we
                        // can logically reconcile "reinterpretation" for different types with the resulting
                        // data view having an actual type.
                        throw ch.ExceptUserArg(nameof(file), "Data file terms loaded as type '{0}' but mismatches column '{1}' item type '{2}'",
                            termsFromFile.ItemType, infos[iinfo].Name, infos[iinfo].TypeSrc.GetItemType());
                    }
                    termMap[iinfo] = termsFromFile;
                }
                else
                {
                    // Auto train this column. Leave the term map null for now, but set the lim appropriately.
                    lims[iinfo] = columns[iinfo].MaxNumTerms;
                    ch.CheckUserArg(lims[iinfo] > 0, nameof(Column.MaxNumTerms), "Must be positive");
                    Contracts.Check(trainingData.Schema.TryGetColumnIndex(infos[iinfo].Source, out int colIndex));
                    Utils.Add(ref toTrain, colIndex);
                    ++trainsNeeded;
                }
            }

            ch.Assert((Utils.Size(toTrain) == 0) == (trainsNeeded == 0));
            ch.Assert(Utils.Size(toTrain) <= trainsNeeded);
            if (trainsNeeded > 0)
            {
                Trainer[] trainer = new Trainer[trainsNeeded];
                int[] trainerInfo = new int[trainsNeeded];
                // Open the cursor, then instantiate the trainers.
                int itrainer;
                using (var cursor = trainingData.GetRowCursor(trainingData.Schema.Where(c => toTrain.Contains(c.Index))))
                using (var pch = env.StartProgressChannel("Building term dictionary"))
                {
                    long rowCur = 0;
                    double rowCount = trainingData.GetRowCount() ?? double.NaN;
                    var header = new ProgressHeader(new[] { "Total Terms" }, new[] { "examples" });

                    itrainer = 0;
                    for (int iinfo = 0; iinfo < infos.Length; ++iinfo)
                    {
                        if (termMap[iinfo] != null)
                            continue;
                        var bldr = Builder.Create(infos[iinfo].TypeSrc, columns[iinfo].Sort);
                        trainerInfo[itrainer] = iinfo;
                        trainingData.Schema.TryGetColumnIndex(infos[iinfo].Source, out int colIndex);
                        trainer[itrainer++] = Trainer.Create(cursor, colIndex, false, lims[iinfo], bldr);
                    }
                    ch.Assert(itrainer == trainer.Length);
                    pch.SetHeader(header,
                        e =>
                        {
                            e.SetProgress(0, rowCur, rowCount);
                            // Purely feedback for the user. That the other thread might be
                            // working in the background is not a problem.
                            e.SetMetric(0, trainer.Sum(t => t.Count));
                        });

                    // The [0,tmin) trainers are finished.
                    int tmin = 0;
                    // We might exit early if all trainers reach their maximum.
                    while (tmin < trainer.Length && cursor.MoveNext())
                    {
                        rowCur++;
                        for (int t = tmin; t < trainer.Length; ++t)
                        {
                            if (!trainer[t].ProcessRow())
                            {
                                Utils.Swap(ref trainerInfo[t], ref trainerInfo[tmin]);
                                Utils.Swap(ref trainer[t], ref trainer[tmin++]);
                            }
                        }
                    }

                    pch.Checkpoint(trainer.Sum(t => t.Count), rowCur);
                }
                for (itrainer = 0; itrainer < trainer.Length; ++itrainer)
                {
                    int iinfo = trainerInfo[itrainer];
                    ch.Assert(termMap[iinfo] == null);
                    if (trainer[itrainer].Count == 0)
                        ch.Warning("Term map for output column '{0}' contains no entries.", infos[iinfo].Name);
                    termMap[iinfo] = trainer[itrainer].Finish();
                    // Allow the intermediate structures in the trainer and builder to be released as we iterate
                    // over the columns, as the Finish operation can potentially result in the allocation of
                    // additional structures.
                    trainer[itrainer] = null;
                }
                ch.Assert(termMap.All(tm => tm != null));
                ch.Assert(termMap.Zip(infos, (tm, info) => tm.ItemType.Equals(info.TypeSrc.GetItemType())).All(x => x));
            }

            return termMap;
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            SaveColumns(ctx);

            Host.Assert(_unboundMaps.Length == _textMetadata.Length);
            Host.Assert(_textMetadata.Length == ColumnPairs.Length);
            ctx.Writer.WriteBoolBytesNoCount(_textMetadata);

            // REVIEW: Should we do separate sub models for each dictionary?
            const string dir = "Vocabulary";
            ctx.SaveSubModel(dir,
                c =>
                {
                    // *** Binary format ***
                    // int: number of term maps (should equal number of columns)
                    // for each term map:
                    //   byte: code identifying the term map type (0 text, 1 codec)
                    //   <data>: type specific format, see TermMap save/load methods

                    Host.CheckValue(c, nameof(ctx));
                    c.CheckAtModel();
                    c.SetVersionInfo(GetTermManagerVersionInfo());
                    c.Writer.Write(_unboundMaps.Length);
                    foreach (var term in _unboundMaps)
                        term.Save(c, Host, CodecFactory);

                    c.SaveTextStream("Terms.txt",
                        writer =>
                        {
                            foreach (var map in _unboundMaps)
                                map.WriteTextTerms(writer);
                        });
                });
        }

        [BestFriend]
        internal TermMap GetTermMap(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < _unboundMaps.Length);
            return _unboundMaps[iinfo];
        }

        private protected override IRowMapper MakeRowMapper(Schema schema)
          => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx, ISaveAsPfa
        {
            private readonly ColumnType[] _types;
            private readonly ValueToKeyMappingTransformer _parent;
            private readonly ColInfo[] _infos;
            private readonly BoundTermMap[] _termMap;

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public bool CanSavePfa => true;

            public Mapper(ValueToKeyMappingTransformer parent, Schema inputSchema)
               : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _infos = _parent.CreateInfos(inputSchema);
                _types = new ColumnType[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    var type = _infos[i].TypeSrc;
                    KeyType keyType = _parent._unboundMaps[i].OutputType;
                    ColumnType colType;
                    if (type is VectorType vectorType)
                        colType = new VectorType(keyType, vectorType);
                    else
                        colType = keyType;
                    _types[i] = colType;
                }
                _termMap = new BoundTermMap[_parent.ColumnPairs.Length];
                for (int iinfo = 0; iinfo < _parent.ColumnPairs.Length; ++iinfo)
                {
                    _termMap[iinfo] = Bind(Host, inputSchema, _parent._unboundMaps[iinfo], _infos, _parent._textMetadata, iinfo);
                }
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_parent.ColumnPairs.Length];
                for (int i = 0; i < _parent.ColumnPairs.Length; i++)
                {
                    InputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].input, out int colIndex);
                    Host.Assert(colIndex >= 0);
                    var builder = new MetadataBuilder();
                    _termMap[i].AddMetadata(builder);

                    builder.Add(InputSchema[colIndex].Metadata, name => name == MetadataUtils.Kinds.SlotNames);
                    result[i] = new Schema.DetachedColumn(_parent.ColumnPairs[i].output, _types[i], builder.GetMetadata());
                }
                return result;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                disposer = null;
                var type = _termMap[iinfo].Map.OutputType;
                return Utils.MarshalInvoke(MakeGetter<int>, type.RawType, input, iinfo);
            }

            private Delegate MakeGetter<T>(Row row, int src) => _termMap[src].GetMappingGetter(row);

            private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
            {
                if (!(info.TypeSrc.GetItemType() is TextType))
                    return false;

                var terms = default(VBuffer<ReadOnlyMemory<char>>);
                TermMap<ReadOnlyMemory<char>> map = (TermMap<ReadOnlyMemory<char>>)_termMap[iinfo].Map;
                map.GetTerms(ref terms);
                string opType = "LabelEncoder";
                var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                node.AddAttribute("classes_strings", terms.DenseValues());
                node.AddAttribute("default_int64", -1);
                //default_string needs to be an empty string but there is a BUG in Lotus that
                //throws a validation error when default_string is empty. As a work around, set
                //default_string to a space.
                node.AddAttribute("default_string", " ");
                return true;
            }

            public void SaveAsOnnx(OnnxContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                for (int iinfo = 0; iinfo < _infos.Length; ++iinfo)
                {
                    ColInfo info = _infos[iinfo];
                    string sourceColumnName = info.Source;
                    if (!ctx.ContainsColumn(sourceColumnName))
                    {
                        ctx.RemoveColumn(info.Name, false);
                        continue;
                    }

                    if (!SaveAsOnnxCore(ctx, iinfo, info, ctx.GetVariableName(sourceColumnName),
                        ctx.AddIntermediateVariable(_types[iinfo], info.Name)))
                    {
                        ctx.RemoveColumn(info.Name, true);
                    }
                }
            }

            public void SaveAsPfa(BoundPfaContext ctx)
            {
                Host.CheckValue(ctx, nameof(ctx));

                var toHide = new List<string>();
                var toDeclare = new List<KeyValuePair<string, JToken>>();

                for (int iinfo = 0; iinfo < _infos.Length; ++iinfo)
                {
                    var info = _infos[iinfo];
                    var srcName = info.Source;
                    string srcToken = ctx.TokenOrNullForName(srcName);
                    if (srcToken == null)
                    {
                        toHide.Add(info.Name);
                        continue;
                    }
                    var result = SaveAsPfaCore(ctx, iinfo, info, srcToken);
                    if (result == null)
                    {
                        toHide.Add(info.Name);
                        continue;
                    }
                    toDeclare.Add(new KeyValuePair<string, JToken>(info.Name, result));
                }
                ctx.Hide(toHide.ToArray());
                ctx.DeclareVar(toDeclare.ToArray());
            }

            private JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
            {
                Contracts.AssertValue(ctx);
                Contracts.Assert(0 <= iinfo && iinfo < _infos.Length);
                Contracts.Assert(_infos[iinfo] == info);
                Contracts.AssertValue(srcToken);
                //Contracts.Assert(CanSavePfa);

                VectorType vectorType = info.TypeSrc as VectorType;
                ColumnType itemType = vectorType?.ItemType ?? info.TypeSrc;
                if (!(itemType is TextType))
                    return null;
                var terms = default(VBuffer<ReadOnlyMemory<char>>);
                TermMap<ReadOnlyMemory<char>> map = (TermMap<ReadOnlyMemory<char>>)_termMap[iinfo].Map;
                map.GetTerms(ref terms);
                var jsonMap = new JObject();
                foreach (var kv in terms.Items())
                    jsonMap[kv.Value.ToString()] = kv.Key;
                string cellName = ctx.DeclareCell(
                    "TermMap", PfaUtils.Type.Map(PfaUtils.Type.Int), jsonMap);
                JObject cellRef = PfaUtils.Cell(cellName);

                if (vectorType != null)
                {
                    var funcName = ctx.GetFreeFunctionName("mapTerm");
                    ctx.Pfa.AddFunc(funcName, new JArray(PfaUtils.Param("term", PfaUtils.Type.String)),
                        PfaUtils.Type.Int, PfaUtils.If(PfaUtils.Call("map.containsKey", cellRef, "term"), PfaUtils.Index(cellRef, "term"), -1));
                    var funcRef = PfaUtils.FuncRef("u." + funcName);
                    return PfaUtils.Call("a.map", srcToken, funcRef);
                }
                return PfaUtils.If(PfaUtils.Call("map.containsKey", cellRef, srcToken), PfaUtils.Index(cellRef, srcToken), -1);
            }
        }
    }
}
