// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static Microsoft.ML.Runtime.Data.TermTransform;

[assembly: LoadableClass(TermTransform.Summary, typeof(IDataTransform), typeof(TermTransform),
    typeof(TermTransform.Arguments), typeof(SignatureDataTransform),
    TermTransform.UserName, "Term", "AutoLabel", "TermTransform", "AutoLabelTransform", DocName = "transform/TermTransform.md")]

[assembly: LoadableClass(TermTransform.Summary, typeof(IDataView), typeof(TermTransform), null, typeof(SignatureLoadDataTransform),
    TermTransform.UserName, TermTransform.LoaderSignature)]

[assembly: LoadableClass(TermTransform.Summary, typeof(TermTransform), null, typeof(SignatureLoadModel),
    TermTransform.UserName, TermTransform.LoaderSignature)]

[assembly: LoadableClass(TermTransform.Summary, typeof(TermRowMapper), null, typeof(SignatureLoadRowMapper),
    TermTransform.UserName, TermRowMapper.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    // TermTransform builds up term vocabularies (dictionaries).
    // Notes:
    // * Each column builds/uses exactly one "vocabulary" (dictionary).
    // * Output columns are KeyType-valued.
    // * The Key value is the one-based index of the item in the dictionary.
    // * Not found is assigned the value zero.
    /// <include file='doc.xml' path='doc/members/member[@name="TextToKey"]/*' />
    public sealed partial class TermTransform : ITransformer, ICanSaveModel
    {
        internal readonly IHost Host;

        private ColInfo[] _infos;

        public sealed class ColInfo
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

        public const string Summary = "Converts input values (words, numbers, etc.) to index in a dictionary.";
        public const string UserName = "Term Transform";
        public const string LoaderSignature = "TermTransform";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TERMTRNF",
                // verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Dropped sizeof(Float)
                verWrittenCur: 0x00010003, // Generalize to multiple types beyond text
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const uint VerNonTextTypesSupported = 0x00010003;
        private const uint VerManagerNonTextTypesSupported = 0x00010002;

        public const string TermManagerLoaderSignature = "TermManager";
        internal static VersionInfo GetTermManagerVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TERM MAN",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Generalize to multiple types beyond text
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: TermManagerLoaderSignature);
        }

        private (string Source, string Name)[] _columns;
        private readonly TermMap[] _unboundMaps;
        private readonly bool[] _textMetadata;
        private const string RegistrationName = "Term";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be transformed. If this is null '<paramref name="name"/>' will be used.</param>
        /// <param name="maxNumTerms">Maximum number of terms to keep per column when auto-training.</param>
        /// <param name="sort">How items should be ordered when vectorized. By default, they will be in the order encountered.
        /// If by value items are sorted according to their default comparison, e.g., text sorting will be case sensitive (e.g., 'A' then 'Z' then 'a').</param>
        public TermTransform(IHostEnvironment env,
            IDataView input,
            string name,
            string source = null,
            int maxNumTerms = Defaults.MaxNumTerms,
            SortOrder sort = Defaults.Sort)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source ?? name, Name = name } }, MaxNumTerms = maxNumTerms, Sort = sort }, input)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public TermTransform(IHostEnvironment env, Arguments args, IDataView input)
            : this(env, args, Contracts.CheckRef(args, nameof(args)).Column, input)
        {
        }

        public void CreateInfos(SourceNameColumnBase[] column, ISchema input)
        {
            Host.CheckUserArg(Utils.Size(column) > 0, nameof(column));
            Host.AssertValue(input);
            /*host.AssertValueOrNull(transInput);
            host.AssertValueOrNull(testType);*/

            _infos = new ColInfo[column.Length];
            for (int i = 0; i < column.Length; i++)
            {
                var item = column[i];
                Host.CheckUserArg(item.TrySanitize(), nameof(OneToOneColumn.Name), "Invalid new column name");

                int colSrc;
                if (!input.TryGetColumnIndex(item.Source, out colSrc))
                    throw Host.ExceptUserArg(nameof(OneToOneColumn.Source), "Source column '{0}' not found", item.Source);

                var type = input.GetColumnType(colSrc);
                /*if (testType != null)
                {
                    string reason = testType(type);
                    if (reason != null)
                        throw host.ExceptUserArg(nameof(OneToOneColumn.Source), InvalidTypeErrorFormat, item.Source, type, reason);
                }*/

                //var slotType = transInput == null ? null : transInput.GetSlotType(colSrc);
                _infos[i] = new ColInfo(item.Name, item.Source, type);
            }
        }

        /// <summary>
        /// Public constructor for compositional forms.
        /// </summary>
        public TermTransform(IHostEnvironment env, ArgumentsBase args, ColumnBase[] column, IDataView input)
        {
            Host = env.Register(nameof(TermTransform));
            Host.CheckValue(args, nameof(args));
            CreateInfos(column, input.Schema);
            Host.AssertNonEmpty(_infos);
            Host.Assert(_infos.Length == Utils.Size(column));
            _columns = column.Select(x => (x.Source, x.Name)).ToArray();
            using (var ch = Host.Start("Training"))
            {
                _unboundMaps = Train(Host, ch, _infos, args, column, input);
                _textMetadata = new bool[_unboundMaps.Length];
                for (int iinfo = 0; iinfo < column.Length; ++iinfo)
                {
                    _textMetadata[iinfo] = column[iinfo].TextKeyValues ?? args.TextKeyValues;
                }
                ch.Assert(_unboundMaps.Length == column.Length);
                ch.Done();
            }
        }

        public static IDataTransform Create(IHostEnvironment env, ArgumentsBase args, ColumnBase[] column, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            var transformer = new TermTransform(env, args, column, input);
            return transformer.CreateRowToRowMapper(input);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            var transformer = new TermTransform(env, args, input);
            return transformer.CreateRowToRowMapper(input);
        }

        public static IDataView Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(input, nameof(input));
            var transformer = Create(env, ctx);
            return transformer.Transform(input);
        }

        public static TermTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            env.AssertValue(ctx);
            LoadObjectsFromContext(env, ctx, out (string Source, string Name)[] columns, out bool[] textMetadata, out TermMap[] termMap);
            return new TermTransform(env, columns, textMetadata, termMap);
        }

        internal static void LoadObjectsFromContext(IHostEnvironment env, ModelLoadContext ctx,
            out (string Source, string Name)[] columns, out bool[] textMetadata, out TermMap[] boundMap, bool ignoreVersioning = false)
        {
            // *** Binary format ***
            // for each term map:
            //   bool(byte): whether this column should present key value metadata as text
            var length = ctx.Reader.ReadInt32();
            env.Assert(length > 0);
            columns = new (string Source, string Name)[length];
            for (int i = 0; i < length; i++)
            {
                columns[i].Name = ctx.LoadNonEmptyString();
                columns[i].Source = ctx.LoadNonEmptyString();
            }
            env.Assert(length > 0);
            if (ignoreVersioning)
                textMetadata = ctx.Reader.ReadBoolArray(length);
            else
            {
                if (ctx.Header.ModelVerWritten >= VerNonTextTypesSupported)
                    textMetadata = ctx.Reader.ReadBoolArray(length);
                else
                    textMetadata = new bool[length]; // No need to set in this case. They're all text.
            }

            const string dir = "Vocabulary";
            var termMap = new TermMap[length];
            bool b = ctx.TryProcessSubModel(dir,
            c =>
{
    // *** Binary format ***
    // int: number of term maps (should equal number of columns)
    // for each term map:
    //   byte: code identifying the term map type (0 text, 1 codec)
    //   <data>: type specific format, see TermMap save/load methods

    env.CheckValue(c, nameof(ctx));
    c.CheckAtModel(GetTermManagerVersionInfo());
    int cmap = c.Reader.ReadInt32();
    env.CheckDecode(cmap == length);
    if (c.Header.ModelVerWritten >= VerManagerNonTextTypesSupported)
    {
        for (int i = 0; i < length; ++i)
            termMap[i] = TermMap.Load(c, env);
    }
    else
    {
        for (int i = 0; i < length; ++i)
            termMap[i] = TermMap.TextImpl.Create(c, env);
    }
});
#pragma warning disable MSML_NoMessagesForLoadContext // Vaguely useful.
            if (!b)
                throw env.ExceptDecode("Missing {0} model", dir);
#pragma warning restore MSML_NoMessagesForLoadContext
            boundMap = termMap;
        }

        private static string TestIsKnownDataKind(ColumnType type)
        {
            if (type.ItemType.RawKind != default(DataKind) && (type.IsVector || type.IsPrimitive))
                return null;
            return "Expected standard type or a vector of standard type";
        }

        /// <summary>
        /// Utility method to create the file-based <see cref="TermMap"/> if the <see cref="ArgumentsBase.DataFile"/>
        /// argument of <paramref name="args"/> was present.
        /// </summary>
        private static TermMap CreateFileTermMap(IHostEnvironment env, IChannel ch, ArgumentsBase args, Builder bldr)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(env);
            ch.AssertValue(args);
            ch.Assert(!string.IsNullOrWhiteSpace(args.DataFile));
            ch.AssertValue(bldr);

            string file = args.DataFile;
            // First column using the file.
            string src = args.TermsColumn;
            IMultiStreamSource fileSource = new MultiFileSource(file);

            var loaderFactory = args.Loader;
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
                    ch.CheckUserArg(!string.IsNullOrWhiteSpace(src), nameof(args.TermsColumn),
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
                    termData = TextLoader.ReadFile(env,
                        new TextLoader.Arguments()
                        {
                            Separator = "tab",
                            Column = new[] { new TextLoader.Column("Term", DataKind.TX, 0) }
                        },
                        fileSource);
                    src = "Term";
                    autoConvert = true;
                }
            }
            ch.AssertNonEmpty(src);

            int colSrc;
            if (!termData.Schema.TryGetColumnIndex(src, out colSrc))
                throw ch.ExceptUserArg(nameof(args.TermsColumn), "Unknown column '{0}'", src);
            var typeSrc = termData.Schema.GetColumnType(colSrc);
            if (!autoConvert && !typeSrc.Equals(bldr.ItemType))
                throw ch.ExceptUserArg(nameof(args.TermsColumn), "Must be of type '{0}' but was '{1}'", bldr.ItemType, typeSrc);

            using (var cursor = termData.GetRowCursor(col => col == colSrc))
            using (var pch = env.StartProgressChannel("Building term dictionary from file"))
            {
                var header = new ProgressHeader(new[] { "Total Terms" }, new[] { "examples" });
                var trainer = Trainer.Create(cursor, colSrc, autoConvert, int.MaxValue, bldr);
                double rowCount = termData.GetRowCount(true) ?? double.NaN;
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
            ArgumentsBase args, ColumnBase[] column, IDataView trainingData)
        {
            Contracts.AssertValue(env);
            env.AssertValue(ch);
            ch.AssertValue(infos);
            ch.AssertValue(args);
            ch.AssertValue(column);
            ch.AssertValue(trainingData);

            if ((args.Term != null || !string.IsNullOrEmpty(args.Terms)) &&
                (!string.IsNullOrWhiteSpace(args.DataFile) || args.Loader != null ||
                    !string.IsNullOrWhiteSpace(args.TermsColumn)))
            {
                ch.Warning("Explicit term list specified. Data file arguments will be ignored");
            }

            if (!Enum.IsDefined(typeof(SortOrder), args.Sort))
                throw ch.ExceptUserArg(nameof(args.Sort), "Undefined sorting criteria '{0}' detected", args.Sort);

            TermMap termsFromFile = null;
            var termMap = new TermMap[infos.Length];
            int[] lims = new int[infos.Length];
            int trainsNeeded = 0;
            HashSet<int> toTrain = null;

            for (int iinfo = 0; iinfo < infos.Length; iinfo++)
            {
                // First check whether we have a terms argument, and handle it appropriately.
                var terms = new DvText(column[iinfo].Terms);
                var termsArray = column[iinfo].Term;
                if (!terms.HasChars && termsArray == null)
                {
                    terms = new DvText(args.Terms);
                    termsArray = args.Term;
                }

                terms = terms.Trim();
                if (terms.HasChars || (termsArray != null && termsArray.Length > 0))
                {
                    // We have terms! Pass it in.
                    var sortOrder = column[iinfo].Sort ?? args.Sort;
                    if (!Enum.IsDefined(typeof(SortOrder), sortOrder))
                        throw ch.ExceptUserArg(nameof(args.Sort), "Undefined sorting criteria '{0}' detected for column '{1}'", sortOrder, infos[iinfo].Name);

                    var bldr = Builder.Create(infos[iinfo].TypeSrc, sortOrder);
                    if (terms.HasChars)
                        bldr.ParseAddTermArg(ref terms, ch);
                    else
                        bldr.ParseAddTermArg(termsArray, ch);
                    termMap[iinfo] = bldr.Finish();
                }
                else if (!string.IsNullOrWhiteSpace(args.DataFile))
                {
                    // First column using this file.
                    if (termsFromFile == null)
                    {
                        var bldr = Builder.Create(infos[iinfo].TypeSrc, column[iinfo].Sort ?? args.Sort);
                        termsFromFile = CreateFileTermMap(env, ch, args, bldr);
                    }
                    if (!termsFromFile.ItemType.Equals(infos[iinfo].TypeSrc.ItemType))
                    {
                        // We have no current plans to support re-interpretation based on different column
                        // type, not only because it's unclear what realistic customer use-cases for such
                        // a complicated feature would be, and also because it's difficult to see how we
                        // can logically reconcile "reinterpretation" for different types with the resulting
                        // data view having an actual type.
                        throw ch.ExceptUserArg(nameof(args.DataFile), "Data file terms loaded as type '{0}' but mismatches column '{1}' item type '{2}'",
                            termsFromFile.ItemType, infos[iinfo].Name, infos[iinfo].TypeSrc.ItemType);
                    }
                    termMap[iinfo] = termsFromFile;
                }
                else
                {
                    // Auto train this column. Leave the term map null for now, but set the lim appropriately.
                    lims[iinfo] = column[iinfo].MaxNumTerms ?? args.MaxNumTerms;
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
                using (var cursor = trainingData.GetRowCursor(toTrain.Contains))
                using (var pch = env.StartProgressChannel("Building term dictionary"))
                {
                    long rowCur = 0;
                    double rowCount = trainingData.GetRowCount(true) ?? double.NaN;
                    var header = new ProgressHeader(new[] { "Total Terms" }, new[] { "examples" });

                    itrainer = 0;
                    for (int iinfo = 0; iinfo < infos.Length; ++iinfo)
                    {
                        if (termMap[iinfo] != null)
                            continue;
                        var bldr = Builder.Create(infos[iinfo].TypeSrc, column[iinfo].Sort ?? args.Sort);
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
                ch.Assert(termMap.Zip(infos, (tm, info) => tm.ItemType.Equals(info.TypeSrc.ItemType)).All(x => x));
            }

            return termMap;
        }

        private TermTransform(IHost host, ModelLoadContext ctx, IDataView input)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // for each term map:
            //   bool(byte): whether this column should present key value metadata as text
            int cinfo = _infos.Length;
            Host.Assert(cinfo > 0);

            if (ctx.Header.ModelVerWritten >= VerNonTextTypesSupported)
                _textMetadata = ctx.Reader.ReadBoolArray(cinfo);
            else
                _textMetadata = new bool[cinfo]; // No need to set in this case. They're all text.

            const string dir = "Vocabulary";
            TermMap[] termMap = new TermMap[cinfo];
            bool b = ctx.TryProcessSubModel(dir,
                c =>
                {
                    // *** Binary format ***
                    // int: number of term maps (should equal number of columns)
                    // for each term map:
                    //   byte: code identifying the term map type (0 text, 1 codec)
                    //   <data>: type specific format, see TermMap save/load methods

                    Host.CheckValue(c, nameof(ctx));
                    c.CheckAtModel(GetTermManagerVersionInfo());
                    int cmap = c.Reader.ReadInt32();
                    Host.CheckDecode(cmap == cinfo);
                    if (c.Header.ModelVerWritten >= VerManagerNonTextTypesSupported)
                    {
                        for (int i = 0; i < cinfo; ++i)
                            termMap[i] = TermMap.Load(c, host);
                    }
                    else
                    {
                        for (int i = 0; i < cinfo; ++i)
                            termMap[i] = TermMap.TextImpl.Create(c, host);
                    }
                });
#pragma warning disable MSML_NoMessagesForLoadContext // Vaguely useful.
            if (!b)
                throw Host.ExceptDecode("Missing {0} model", dir);
#pragma warning restore MSML_NoMessagesForLoadContext
        }

        internal TermTransform(IHostEnvironment env, (string Source, string Name)[] columns, bool[] textMetadata, TermMap[] termMap)
        {
            Host = env.Register(nameof(TermTransform));
            _columns = columns;
            _textMetadata = textMetadata;
            _unboundMaps = termMap;
        }

        internal static void Save(ModelSaveContext ctx, IHostEnvironment host, (string source, string name)[] columns, TermMap[] termMap, bool[] textMetadata)
        {
            // *** Binary format ***
            // for each term map:
            //   bool(byte): whether this column should present key value metadata as text
            ctx.Writer.Write(columns.Length);
            foreach (var info in columns)
            {
                ctx.SaveNonEmptyString(info.name);
                ctx.SaveNonEmptyString(info.source);
            }

            host.Assert(termMap.Length == columns.Length);
            host.Assert(textMetadata.Length == columns.Length);
            ctx.Writer.WriteBoolBytesNoCount(textMetadata, textMetadata.Length);

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

                    host.CheckValue(c, nameof(ctx));
                    c.CheckAtModel();
                    c.SetVersionInfo(GetTermManagerVersionInfo());
                    c.Writer.Write(termMap.Length);
                    foreach (var term in termMap)
                        term.Save(c, host);

                    c.SaveTextStream("Terms.txt",
                        writer =>
                        {
                            foreach (var map in termMap)
                                map.WriteTextTerms(writer);
                        });
                });
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            Save(ctx, Host, _infos.Select(x => (x.Source, x.Name)).ToArray(), _unboundMaps, _textMetadata);
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            // Validate schema.
            return Transform(new EmptyDataView(Host, inputSchema)).Schema;
        }

        private RowToRowMapperTransform CreateRowToRowMapper(IDataView input)
        {
            var mapper = new TermRowMapper(Host, input.Schema, _columns, _textMetadata, _unboundMaps);
            return new RowToRowMapperTransform(Host, input, mapper);
        }

        public IDataView Transform(IDataView input)
        {
            return CreateRowToRowMapper(input);
        }

        //IVAN: FIGURE OUT HOW IT WILL WORK IN NEW WORLD
        private JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
        {
            Contracts.AssertValue(ctx);
            Contracts.Assert(0 <= iinfo && iinfo < _infos.Length);
            Contracts.Assert(_infos[iinfo] == info);
            Contracts.AssertValue(srcToken);
            //Contracts.Assert(CanSavePfa);

            if (!info.TypeSrc.ItemType.IsText)
                return null;
            var terms = default(VBuffer<DvText>);
            TermMap<DvText> map = (TermMap<DvText>)_unboundMaps[iinfo];
            map.GetTerms(ref terms);
            var jsonMap = new JObject();
            foreach (var kv in terms.Items())
                jsonMap[kv.Value.ToString()] = kv.Key;
            string cellName = ctx.DeclareCell(
                "TermMap", PfaUtils.Type.Map(PfaUtils.Type.Int), jsonMap);
            JObject cellRef = PfaUtils.Cell(cellName);

            if (info.TypeSrc.IsVector)
            {
                var funcName = ctx.GetFreeFunctionName("mapTerm");
                ctx.Pfa.AddFunc(funcName, new JArray(PfaUtils.Param("term", PfaUtils.Type.String)),
                    PfaUtils.Type.Int, PfaUtils.If(PfaUtils.Call("map.containsKey", cellRef, "term"), PfaUtils.Index(cellRef, "term"), -1));
                var funcRef = PfaUtils.FuncRef("u." + funcName);
                return PfaUtils.Call("a.map", srcToken, funcRef);
            }
            return PfaUtils.If(PfaUtils.Call("map.containsKey", cellRef, srcToken), PfaUtils.Index(cellRef, srcToken), -1);
        }

        //IVAN: FIGURE OUT HOW IT WILL WORK IN NEW WORLD
        private bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
        {
            if (!info.TypeSrc.ItemType.IsText)
                return false;

            var terms = default(VBuffer<DvText>);
            TermMap<DvText> map = (TermMap<DvText>)_unboundMaps[iinfo];
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

    }

    internal sealed class TermRowMapper : IRowMapper
    {
        internal readonly ISchema Schema;
        private readonly Dictionary<int, int> _colNewToOldMapping;
        private readonly (string Source, string Name)[] _columns;
        internal readonly IHost Host;
        public const string LoaderSignature = "TermRowMapper";

        public ColInfo[] Infos;

        private readonly BoundTermMap[] _termMap;
        internal readonly bool[] TextMetadata;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TERMROWM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public static TermRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            LoadObjectsFromContext(env, ctx, out (string Source, string Name)[] columns, out bool[] textMetadata, out TermMap[] termMap, true);
            return new TermRowMapper(env, schema, columns, textMetadata, termMap);
        }

        internal TermRowMapper(IHostEnvironment env, ISchema schema, (string source, string name)[] columns, bool[] textMetadata, TermMap[] unboundMaps)
        {
            Host = env.Register(LoaderSignature);
            Host.CheckValue(schema, nameof(schema));
            Host.CheckValue(columns, nameof(columns));
            Schema = schema;
            _columns = columns;
            _colNewToOldMapping = new Dictionary<int, int>();
            for (int i = 0; i < columns.Length; i++)
            {
                if (!Schema.TryGetColumnIndex(columns[i].source, out int colIndex))
                {
                    throw Host.ExceptParam(nameof(schema), $"{columns[i].source} not found in {nameof(schema)}");
                }
                _colNewToOldMapping.Add(i, colIndex);
            }
            Infos = new ColInfo[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var item = columns[i];
                int colSrc;
                if (!schema.TryGetColumnIndex(item.source, out colSrc))
                    throw Host.ExceptUserArg(nameof(OneToOneColumn.Source), "Source column '{0}' not found", item.source);

                var type = schema.GetColumnType(colSrc);
                /*if (testType != null)
                {
                    string reason = testType(type);
                    if (reason != null)
                        throw host.ExceptUserArg(nameof(OneToOneColumn.Source), InvalidTypeErrorFormat, item.Source, type, reason);
                }*/

                //var slotType = transInput == null ? null : transInput.GetSlotType(colSrc);
                Infos[i] = new ColInfo(item.name, item.source, type);
            }
            TextMetadata = textMetadata;
            _termMap = new BoundTermMap[unboundMaps.Length];
            for (int iinfo = 0; iinfo < columns.Length; ++iinfo)
            {
                _termMap[iinfo] = unboundMaps[iinfo].Bind(this, iinfo);
            }
        }

        public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
        {
            Host.Assert(input.Schema == Schema);
            var result = new Delegate[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                if (!activeOutput(i))
                    continue;
                var type = _termMap[i].Map.OutputType;
                result[i] = Utils.MarshalInvoke(MakeGetter<int>, type.RawType, input, i);
            }
            disposer = null;
            return result;
        }

        private Delegate MakeGetter<T>(IRow row, int src) => _termMap[src].GetMappingGetter(row);

        public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
        {
            var active = new bool[Schema.ColumnCount];
            foreach (var pair in _colNewToOldMapping)
                if (activeOutput(pair.Key))
                    active[pair.Value] = true;
            return col => active[col];
        }

        public RowMapperColumnInfo[] GetOutputColumns()
        {
            var result = new RowMapperColumnInfo[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                Schema.TryGetColumnIndex(Infos[i].Source, out int colIndex);
                var colMetaInfo = new ColumnMetadataInfo(Infos[i].Name);
                _termMap[i].AddMetadata(colMetaInfo);
                var colType = _termMap[i].Map.OutputType;
                result[i] = new RowMapperColumnInfo(Infos[i].Name, colType, colMetaInfo);
            }
            return result;
        }

        private int AddMetaGetter<T>(ColumnMetadataInfo colMetaInfo, ISchema schema, string kind, ColumnType ct, Dictionary<int, int> colMap)
        {
            MetadataUtils.MetadataGetter<T> getter = (int col, ref T dst) =>
            {
                var originalCol = colMap[col];
                schema.GetMetadata<T>(kind, originalCol, ref dst);
            };
            var info = new MetadataInfo<T>(ct, getter);
            colMetaInfo.Add(kind, info);
            return 0;
        }

        public void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            TermTransform.Save(ctx, Host, _columns, _termMap.Select(x => x.Map).ToArray(), TextMetadata);
        }
    }
}
