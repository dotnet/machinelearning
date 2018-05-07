// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Web;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(SimplePartitionedPathParser.Summary, typeof(SimplePartitionedPathParser), typeof(SimplePartitionedPathParser.Arguments), typeof(PartitionedPathParser),
    SimplePartitionedPathParser.UserName, SimplePartitionedPathParser.LoadName, SimplePartitionedPathParser.ShortName)]
[assembly: LoadableClass(ParquetPartitionedPathParser.Summary, typeof(ParquetPartitionedPathParser), null, typeof(PartitionedPathParser),
    ParquetPartitionedPathParser.UserName, ParquetPartitionedPathParser.LoadName, ParquetPartitionedPathParser.ShortName)]

// This is for deserialization
[assembly: LoadableClass(SimplePartitionedPathParser.Summary, typeof(SimplePartitionedPathParser), null, typeof(SignatureLoadModel),
    SimplePartitionedPathParser.UserName, SimplePartitionedPathParser.LoadName, SimplePartitionedPathParser.ShortName)]
[assembly: LoadableClass(ParquetPartitionedPathParser.Summary, typeof(ParquetPartitionedPathParser), null, typeof(SignatureLoadModel),
    ParquetPartitionedPathParser.UserName, ParquetPartitionedPathParser.LoadName, ParquetPartitionedPathParser.ShortName)]

[assembly: EntryPointModule(typeof(SimplePartitionedPathParser.Arguments))]
[assembly: EntryPointModule(typeof(ParquetPartitionedPathParserFactory))]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Delegate signature for a partitioned path parser.
    /// </summary>
    public delegate void PartitionedPathParser();

    /// <summary>
    /// Supports extracting column names and values from a path string.
    /// </summary>
    public interface IPartitionedPathParser
    {
        /// <summary>
        /// Extract the column definitions from a file path.
        /// </summary>
        /// <param name="path">The file path.</param>
        /// <returns>The resulting column definitions.</returns>
        /// <exception cref="FormatException">Thrown when parsing fails.</exception>
        IEnumerable<PartitionedFileLoader.Column> ParseColumns(string path);

        /// <summary>
        /// Extract the column values from a file path.
        /// </summary>
        /// <param name="path">The file path.</param>
        /// <returns>The resulting column values.</returns>
        /// <exception cref="FormatException">Thrown when parsing fails.</exception>
        IEnumerable<string> ParseValues(string path);
    }

    [TlcModule.ComponentKind("PartitionedPathParser")]
    public interface IPartitionedPathParserFactory : IComponentFactory<IPartitionedPathParser>
    {
        new IPartitionedPathParser CreateComponent(IHostEnvironment env);
    }

    public sealed class SimplePartitionedPathParser : IPartitionedPathParser, ICanSaveModel
    {
        internal const string Summary = "A simple parser that extracts directory names as column values. Column names are defined as arguments.";
        internal const string UserName = "Simple Partitioned Path Parser";
        public const string LoadName = "SimplePathParser";
        public const string ShortName = "SmplPP";

        [TlcModule.Component(Name = SimplePartitionedPathParser.LoadName, FriendlyName = SimplePartitionedPathParser.UserName,
            Desc = SimplePartitionedPathParser.Summary, Alias = SimplePartitionedPathParser.ShortName)]
        public class Arguments : IPartitionedPathParserFactory
        {
            [Argument(ArgumentType.Multiple, HelpText = "Column definitions used to override the Partitioned Path Parser. Expected with the format name:type:numeric-source, e.g. col=MyFeature:R4:1",
                ShortName = "col", SortOrder = 1)]
            public Microsoft.ML.Runtime.Data.PartitionedFileLoader.Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Data type of each column.")]
            public DataKind? Type = DataKind.Text;

            public IPartitionedPathParser CreateComponent(IHostEnvironment env) => new SimplePartitionedPathParser(env, this);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SMPLPARS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName);
        }

        private IHost _host;
        private PartitionedFileLoader.Column[] _columns;

        public SimplePartitionedPathParser(IHostEnvironment env, Arguments args)
        {
            _host = env.Register(LoadName);

            _columns = args.Columns;
            foreach (var col in _columns)
            {
                if (!col.Type.HasValue)
                {
                    col.Type = args.Type.HasValue ? args.Type : DataKind.Text;
                }
            }
        }

        private SimplePartitionedPathParser(IHost host, ModelLoadContext ctx)
        {
            Contracts.AssertValue(host);
            _host = host;
            _host.AssertValue(ctx);

            // ** Binary format **
            // int: number of columns
            // foreach column:
            //   string: column representation

            int numColumns = ctx.Reader.ReadInt32();
            _host.CheckDecode(numColumns >= 0);

            _columns = new PartitionedFileLoader.Column[numColumns];
            for (int i = 0; i < numColumns; i++)
            {
                var column = PartitionedFileLoader.Column.Parse(ctx.LoadString());
                _host.CheckDecode(column != null);
                _columns[i] = column;
            }
        }

        public static SimplePartitionedPathParser Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            IHost host = env.Register(LoadName);
            ctx.CheckAtModel(GetVersionInfo());

            return host.Apply("Loading Parser",
                ch => new SimplePartitionedPathParser(host, ctx));
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());

            // ** Binary format **
            // int: number of columns
            // foreach column:
            //   string: column representation

            ctx.Writer.Write(_columns.Length);
            StringBuilder sb = new StringBuilder();
            foreach (var col in _columns)
            {
                sb.Clear();
                _host.Check(col.TryUnparse(sb));
                ctx.SaveString(sb.ToString());
            }
        }

        public IEnumerable<PartitionedFileLoader.Column> ParseColumns(string path)
        {
            Contracts.AssertNonEmpty(path);

            // Verify that path matches the columns expectations.
            var values = ParseValues(path);
            foreach (var col in _columns)
            {
                if (col.Source < 0 || col.Source >= values.Count())
                {
                    throw new FormatException($"Column definition {col} is outside the bounds of path {path}.");
                }
            }

            return _columns;
        }

        public IEnumerable<string> ParseValues(string path)
        {
            Contracts.AssertNonEmpty(path);

            var dirs = Utils.SplitDirectories(path);
            return dirs.Take(dirs.Count() - 1); // Ignore last directory which is the file name.
        }
    }

    [TlcModule.Component(Name = ParquetPartitionedPathParser.LoadName, FriendlyName = ParquetPartitionedPathParser.UserName,
        Desc = ParquetPartitionedPathParser.Summary, Alias = ParquetPartitionedPathParser.ShortName)]
    public class ParquetPartitionedPathParserFactory : IPartitionedPathParserFactory
    {
        public IPartitionedPathParser CreateComponent(IHostEnvironment env) => new ParquetPartitionedPathParser();
    }

    public sealed class ParquetPartitionedPathParser : IPartitionedPathParser, ICanSaveModel
    {
        internal const string Summary = "Extract name/value pairs from Parquet formatted directory names. Example path: Year=2018/Month=12/data1.parquet";
        internal const string UserName = "Parquet Partitioned Path Parser";
        public const string LoadName = "ParquetPathParser";
        public const string ShortName = "ParqPP";

        private IHost _host;
        private PartitionedFileLoader.Column[] _columns;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "PARQPARS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoadName);
        }

        public ParquetPartitionedPathParser()
        {
            _columns = new PartitionedFileLoader.Column[0];
        }

        private ParquetPartitionedPathParser(IHost host, ModelLoadContext ctx)
        {
            Contracts.AssertValue(host);
            _host = host;
            _host.AssertValue(ctx);

            // ** Binary format **
            // int: number of columns
            // foreach column:
            //   string: column representation

            int numColumns = ctx.Reader.ReadInt32();
            _host.CheckDecode(numColumns >= 0);

            _columns = new PartitionedFileLoader.Column[numColumns];
            for (int i = 0; i < numColumns; i++)
            {
                var column = PartitionedFileLoader.Column.Parse(ctx.LoadString());
                _host.CheckDecode(column != null);
                _columns[i] = column;
            }
        }

        public static ParquetPartitionedPathParser Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            IHost host = env.Register(LoadName);
            ctx.CheckAtModel(GetVersionInfo());

            return host.Apply("Loading Parser",
                ch => new ParquetPartitionedPathParser(host, ctx));
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.SetVersionInfo(GetVersionInfo());

            // ** Binary format **
            // int: number of columns
            // foreach column:
            //   string: column representation

            ctx.Writer.Write(_columns.Length);
            StringBuilder sb = new StringBuilder();
            foreach (var col in _columns)
            {
                sb.Clear();
                _host.Check(col.TryUnparse(sb));
                ctx.SaveString(sb.ToString());
            };
        }

        public IEnumerable<PartitionedFileLoader.Column> ParseColumns(string path)
        {
            if (!TryParseNames(path, out List<string> names))
            {
                throw new FormatException($"Failed to parse names from path {path}. Expected directory names with the format 'Name=Value'.");
            }

            _columns = new PartitionedFileLoader.Column[names.Count];
            for (int i = 0; i < names.Count; i++)
            {
                _columns[i] = new PartitionedFileLoader.Column()
                {
                    Name = names[i],
                    Source = i,
                    Type = DataKind.Text
                };
            }

            return _columns;
        }

        public IEnumerable<string> ParseValues(string path)
        {
            if (!TryParseValues(path, out List<string> values))
            {
                throw new FormatException($"Failed to parse names from path {path}. Expected directory names with the format 'Name=Value'.");
            }

            if (values.Count != _columns.Length)
            {
                throw new FormatException($"The extracted value count of {values.Count} does not match the expected Column count of {_columns.Length} for path {path}");
            }

            return values;
        }

        public bool TryParseNames(string path, out List<string> names)
        {
            return TryParseNamesAndValues(path, out names, out List<string> values);
        }

        public bool TryParseValues(string path, out List<string> values)
        {
            return TryParseNamesAndValues(path, out List<string> names, out values);
        }

        public bool TryParseNamesAndValues(string path, out List<string> names, out List<string> values)
        {
            names = null;
            values = null;

            if (string.IsNullOrEmpty(path))
            {
                return false;
            }

            var dirs = Utils.SplitDirectories(path);
            dirs = dirs.Take(dirs.Count() - 1); // Ignore last directory which is the file name.

            names = new List<string>(dirs.Count());
            values = new List<string>(dirs.Count());

            foreach (var dir in dirs)
            {
                if (!TryParseNameValueFromDir(dir, out string name, out string value))
                {
                    return false;
                }

                names.Add(name);
                values.Add(value);
            }

            return true;
        }

        /// <summary>
        /// Parse the name/value pair from a partitioned directory name.
        /// </summary>
        /// <param name="dir">The directory name.</param>
        /// <param name="name">The resulting name.</param>
        /// <param name="value">The resulting value.</param>
        /// <returns>true if the parsing was successfull.</returns>
        private static bool TryParseNameValueFromDir(string dir, out string name, out string value)
        {
            const char nameValueSeparator = '=';

            name = null;
            value = null;

            if (string.IsNullOrEmpty(dir))
            {
                return false;
            }

            var nameValue = dir.Split(nameValueSeparator);
            if (nameValue.Length != 2)
            {
                return false;
            }

            name = nameValue[0];
            value = HttpUtility.UrlDecode(nameValue[1]);

            return true;
        }
    }
}
