// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Runtime.Model.Pfa;
using Newtonsoft.Json.Linq;

[assembly: LoadableClass(TermTransform.Summary, typeof(TermTransform), typeof(TermTransform.Arguments), typeof(SignatureDataTransform),
    TermTransform.UserName, "Term", "AutoLabel", "TermTransform", "AutoLabelTransform", DocName = "transform/TermTransform.md")]

[assembly: LoadableClass(TermTransform.Summary, typeof(TermTransform), null, typeof(SignatureLoadDataTransform),
    TermTransform.UserName, TermTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{

    // TermTransform builds up term vocabularies (dictionaries).
    // Notes:
    // * Each column builds/uses exactly one "vocabulary" (dictionary).
    // * Output columns are KeyType-valued.
    // * The Key value is the one-based index of the item in the dictionary.
    // * Not found is assigned the value zero.
    /// <include file='doc.xml' path='doc/members/member[@name="TextToKey"]/*' />
    public sealed partial class TermTransform : OneToOneTransformBase, ITransformTemplate
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
                "If by value items are sorted according to their default comparison, e.g., text sorting will be case sensitive (e.g., 'A' then 'Z' then 'a').")]
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

        private static class Defaults
        {
            public const int MaxNumTerms = 1000000;
            public const SortOrder Sort = SortOrder.Occurrence;
        }

        public abstract class ArgumentsBase : TransformInputBase
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of terms to keep per column when auto-training", ShortName = "max", SortOrder = 5)]
            public int MaxNumTerms = Defaults.MaxNumTerms;

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
                "If by value items are sorted according to their default comparison, e.g., text sorting will be case sensitive (e.g., 'A' then 'Z' then 'a').", SortOrder = 113)]
            public SortOrder Sort = Defaults.Sort;

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
        private static VersionInfo GetTermManagerVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TERM MAN",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Generalize to multiple types beyond text
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010001,
                loaderSignature: TermManagerLoaderSignature);
        }

        // These are parallel to Infos.
        private readonly ColumnType[] _types;
        private readonly BoundTermMap[] _termMap;
        private readonly bool[] _textMetadata;

        private const string RegistrationName = "Term";

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

        public override bool CanSavePfa => true;
        public override bool CanSaveOnnx => true;

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
            : this(args, Contracts.CheckRef(args, nameof(args)).Column, env, input)
        {
        }

        /// <summary>
        /// Re-apply constructor.
        /// </summary>
        private TermTransform(IHostEnvironment env, TermTransform transform, IDataView newSource)
            : base(env, RegistrationName, transform, newSource, TestIsKnownDataKind)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == transform.Infos.Length);

            _textMetadata = transform._textMetadata;
            _termMap = new BoundTermMap[Infos.Length];
            for (int iinfo = 0; iinfo < Infos.Length; ++iinfo)
            {
                TermMap map = transform._termMap[iinfo].Map;
                if (!map.ItemType.Equals(Infos[iinfo].TypeSrc.ItemType))
                {
                    // Column with the same name, but different types.
                    throw Host.Except(
                        "For column '{0}', term map was trained on items of type '{1}' but being applied to type '{2}'",
                        Infos[iinfo].Name, map.ItemType, Infos[iinfo].TypeSrc.ItemType);
                }
                _termMap[iinfo] = map.Bind(this, iinfo);
            }
            _types = ComputeTypesAndMetadata();
        }

        public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            return new TermTransform(env, this, newSource);
        }

        /// <summary>
        /// Public constructor for compositional forms.
        /// </summary>
        public TermTransform(ArgumentsBase args, ColumnBase[] column, IHostEnvironment env, IDataView input)
            : base(env, RegistrationName, column, input, TestIsKnownDataKind)
        {
            Host.CheckValue(args, nameof(args));
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(column));

            using (var ch = Host.Start("Training"))
            {
                TermMap[] unboundMaps = Train(Host, ch, Infos, args, column, Source);
                ch.Assert(unboundMaps.Length == Infos.Length);
                _textMetadata = new bool[unboundMaps.Length];
                _termMap = new BoundTermMap[unboundMaps.Length];
                for (int iinfo = 0; iinfo < Infos.Length; ++iinfo)
                {
                    _textMetadata[iinfo] = column[iinfo].TextKeyValues ?? args.TextKeyValues;
                    _termMap[iinfo] = unboundMaps[iinfo].Bind(this, iinfo);
                }
                _types = ComputeTypesAndMetadata();
                ch.Done();
            }
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
            IDataLoader loader;
            if (loaderFactory != null)
            {
                loader = loaderFactory.CreateComponent(env, fileSource);
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
                        loader = new BinaryLoader(env, new BinaryLoader.Arguments(), fileSource);
                    else
                    {
                        ch.Assert(isTranspose);
                        loader = new TransposeLoader(env, new TransposeLoader.Arguments(), fileSource);
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
                    loader = new TextLoader(env,
                        new TextLoader.Arguments()
                        {
                            Separator = "tab",
                            Column = new[]
                            {
                                new TextLoader.Column()
                                {
                                    Name ="Term",
                                    Type = DataKind.TX,
                                    KeyRange = new KeyRange() { Min = 0 }
                                }
                            }
                        },
                        fileSource);
                    src = "Term";
                    autoConvert = true;
                }
            }
            ch.AssertNonEmpty(src);

            int colSrc;
            if (!loader.Schema.TryGetColumnIndex(src, out colSrc))
                throw ch.ExceptUserArg(nameof(args.TermsColumn), "Unknown column '{0}'", src);
            var typeSrc = loader.Schema.GetColumnType(colSrc);
            if (!autoConvert && !typeSrc.Equals(bldr.ItemType))
                throw ch.ExceptUserArg(nameof(args.TermsColumn), "Must be of type '{0}' but was '{1}'", bldr.ItemType, typeSrc);

            using (var cursor = loader.GetRowCursor(col => col == colSrc))
            using (var pch = env.StartProgressChannel("Building term dictionary from file"))
            {
                var header = new ProgressHeader(new[] { "Total Terms" }, new[] { "examples" });
                var trainer = Trainer.Create(cursor, colSrc, autoConvert, int.MaxValue, bldr);
                double rowCount = loader.GetRowCount(true) ?? double.NaN;
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
                    if(terms.HasChars)
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
                    Utils.Add(ref toTrain, infos[iinfo].Source);
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
                        trainer[itrainer++] = Trainer.Create(cursor, infos[iinfo].Source, false, lims[iinfo], bldr);
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

        // Computes the types of the columns.
        private ColumnType[] ComputeTypesAndMetadata()
        {
            Contracts.Assert(Utils.Size(Infos) > 0);
            Contracts.Assert(Utils.Size(Infos) == Utils.Size(_termMap));

            var md = Metadata;
            var types = new ColumnType[Infos.Length];
            for (int iinfo = 0; iinfo < types.Length; iinfo++)
            {
                Contracts.Assert(types[iinfo] == null);

                var info = Infos[iinfo];
                KeyType keyType = _termMap[iinfo].Map.OutputType;
                Host.Assert(keyType.KeyCount > 0);
                if (info.TypeSrc.IsVector)
                    types[iinfo] = new VectorType(keyType, info.TypeSrc.AsVector);
                else
                    types[iinfo] = keyType;

                // Inherit slot names from source.
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, info.Source, MetadataUtils.Kinds.SlotNames))
                {
                    // Add key values metadata. It is legal to not add anything, in which case
                    // this builder performs no operations except passing slot names.
                    _termMap[iinfo].AddMetadata(bldr);
                }
            }
            md.Seal();
            return types;
        }

        private TermTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, TestIsKnownDataKind)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // for each term map:
            //   bool(byte): whether this column should present key value metadata as text

            int cinfo = Infos.Length;
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
                            termMap[i] = TermMap.Load(c, host, this);
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
            _termMap = new BoundTermMap[cinfo];
            for (int i = 0; i < cinfo; ++i)
                _termMap[i] = termMap[i].Bind(this, i);

            _types = ComputeTypesAndMetadata();
        }

        public static TermTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(input, nameof(input));
            env.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            return h.Apply("Loading Model", ch => new TermTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // for each term map:
            //   bool(byte): whether this column should present key value metadata as text
            SaveBase(ctx);

            Host.Assert(_termMap.Length == Infos.Length);
            Host.Assert(_textMetadata.Length == Infos.Length);
            ctx.Writer.WriteBoolBytesNoCount(_textMetadata, _textMetadata.Length);

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
                    c.Writer.Write(_termMap.Length);
                    foreach (var term in _termMap)
                        term.Map.Save(c, this);

                    c.SaveTextStream("Terms.txt",
                        writer =>
                        {
                            foreach (var map in _termMap)
                                map.WriteTextTerms(writer);
                        });
                });
        }

        protected override JToken SaveAsPfaCore(BoundPfaContext ctx, int iinfo, ColInfo info, JToken srcToken)
        {
            Contracts.AssertValue(ctx);
            Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
            Contracts.Assert(Infos[iinfo] == info);
            Contracts.AssertValue(srcToken);
            Contracts.Assert(CanSavePfa);

            if (!info.TypeSrc.ItemType.IsText)
                return null;
            var terms = default(VBuffer<DvText>);
            TermMap<DvText> map = (TermMap<DvText>)_termMap[iinfo].Map;
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

        protected override bool SaveAsOnnxCore(OnnxContext ctx, int iinfo, ColInfo info, string srcVariableName, string dstVariableName)
        {
            if (!info.TypeSrc.ItemType.IsText)
                return false;

            var terms = default(VBuffer<DvText>);
            TermMap<DvText> map = (TermMap<DvText>)_termMap[iinfo].Map;
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

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < _types.Length);
            var type = _types[iinfo];
            Host.Assert(type != null);
            return type;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);
            disposer = null;

            return _termMap[iinfo].GetMappingGetter(input);
        }
    }
}
