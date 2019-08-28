//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(LineLoader.Summary, typeof(LineLoader), typeof(LineLoader.Arguments), typeof(SignatureDataLoader),
    "", "LineLoader", "Line", DocName = "loader/LineLoader.md")]

[assembly: LoadableClass(LineLoader.Summary, typeof(LineLoader), null, typeof(SignatureLoadDataLoader),
    "Line Data View Loader", LineLoader.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A loader that presents the lines of a text file just as a single column of raw <see cref="ReadOnlyMemory{T}"/> of <see cref="char"/>,
    /// with no processing of comments, headers, and whatnot.
    /// </summary>
    public sealed class LineLoader : ILegacyDataLoader
    {
        public sealed class Arguments
        {
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "The name of the single output column", ShortName = "n")]
            public string Name = "Text";
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema { get; }

        private readonly IHost _host;
        private readonly string _name;
        private readonly IMultiStreamSource _files;

        internal const string Summary = "Loads text files as a dataview consisting of a single column containing the line.";

        public const string LoaderSignature = "LineLoader";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LINELOAD",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LineLoader).Assembly.FullName);
        }

        public LineLoader(IHostEnvironment env, Arguments args, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.CheckValue(args, nameof(args));
            _host.CheckValue(files, nameof(files));

            _host.CheckUserArg(IsValidName(args.Name), nameof(args.Name), "Cannot be null or whitespace only");
            _name = args.Name;
            _files = files;
            Schema = CreateSchema(_host, _name);
        }

        private LineLoader(IHost host, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.AssertValue(host);
            _host = host;
            _host.AssertValue(ctx);
            _host.AssertValue(files);

            // *** Binary format **
            // int: Id of the text column name (cannot be null or whitespace)

            _name = ctx.LoadString();
            _host.CheckDecode(IsValidName(_name));
            _files = files;
            Schema = CreateSchema(_host, _name);
        }

        private static LineLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(files, nameof(files));

            IHost h = env.Register(LoaderSignature);
            return h.Apply("Loading Model",
                ch => new LineLoader(h, ctx, files));
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format **
            // int: Id of the text column name (cannot be null or whitespace)

            _host.Assert(IsValidName(_name));
            ctx.SaveString(_name);
        }

        private static DataViewSchema CreateSchema(IExceptionContext ectx, string name)
        {
            var bldr = new DataViewSchema.Builder();
            bldr.AddColumn(name, TextDataViewType.Instance);
            return bldr.ToSchema();
        }

        private static bool IsValidName(string name)
        {
            return !string.IsNullOrWhiteSpace(name);
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
            private readonly LineLoader _parent;
            private readonly bool _isActive;
            private int _fileIdx;
            private TextReader _currReader;
            private ReadOnlyMemory<char> _text;
            private ValueGetter<ReadOnlyMemory<char>> _getter;

            public override DataViewSchema Schema { get { return _parent.Schema; } }

            public override long Batch { get { return 0; } }

            public Cursor(LineLoader parent, bool isActive)
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

            protected override void Dispose(bool disposing)
            {
                if (_currReader != null)
                {
                    _currReader.Dispose();
                    _currReader = null;
                }
                base.Dispose(disposing);
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

            protected override bool MoveNextCore()
            {
                Ch.AssertValue(_currReader);
                Ch.Assert(-1 <= _fileIdx && _fileIdx < _parent._files.Count);

                for (; ; )
                {
                    var line = _currReader.ReadLine();
                    if (line != null)
                    {
                        if (_isActive)
                            _text = line.AsMemory();
                        return true;
                    }
                    if (++_fileIdx == _parent._files.Count)
                        return false;
                    _currReader = _parent._files.OpenTextReader(_fileIdx);
                }
            }

            public override bool IsColumnActive(DataViewSchema.Column col)
            {
                Ch.CheckParam(col.Index == 0, nameof(col));
                return _isActive;
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column col)
            {
                Ch.CheckParam(col.Index == 0, nameof(col));
                Ch.CheckParam(_isActive, nameof(col), "requested column not active");
                ValueGetter<TValue> getter = _getter as ValueGetter<TValue>;
                if (getter == null)
                    throw Ch.Except("Invalid TValue: '{0}'", typeof(TValue));
                return getter;
            }

            private void Getter(ref ReadOnlyMemory<char> val)
            {
                Ch.Check(IsGood, "cannot call getter with cursor in its current state");
                Ch.Assert(_isActive);
                val = _text;
            }
        }
    }
}
