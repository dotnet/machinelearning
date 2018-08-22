// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(CopyColumnsTransform), typeof(CopyColumnsTransform.Arguments), typeof(SignatureDataTransform),
    CopyColumnsTransform.UserName, "CopyColumns", "CopyColumnsTransform", CopyColumnsTransform.ShortName, DocName = "transform/CopyColumnsTransform.md")]

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(CopyColumnsTransform), null, typeof(SignatureLoadDataTransform),
    CopyColumnsTransform.UserName, CopyColumnsTransform.LoaderSignature)]

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(CopyColumnsTransformer), null, typeof(SignatureLoadModel),
    CopyColumnsTransform.UserName, CopyColumnsTransformer.LoaderSignature)]

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(CopyColumnsRowMapper), null, typeof(SignatureLoadRowMapper),
    CopyColumnsTransform.UserName, CopyColumnsRowMapper.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class CopyColumnsTransform : OneToOneTransformBase
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
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        public const string Summary = "Copy a source column to a new column.";
        public const string UserName = "Copy Columns Transform";
        public const string ShortName = "Copy";

        public const string LoaderSignature = "CopyTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "COPYCOLT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "CopyColumns";

        /// <summary>
        /// Convenience constructor for public facing API.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="input">Input <see cref="IDataView"/>. This is the output from previous transform or loader.</param>
        /// <param name="name">Name of the output column.</param>
        /// <param name="source">Name of the column to be copied.</param>
        public CopyColumnsTransform(IHostEnvironment env, IDataView input, string name, string source)
            : this(env, new Arguments() { Column = new[] { new Column() { Source = source, Name = name } } }, input)
        {
        }

        public CopyColumnsTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, env.CheckRef(args, nameof(args)).Column, input, null)
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));
            SetMetadata();
        }

        private CopyColumnsTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, null)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <base>

            Host.AssertNonEmpty(Infos);
            SetMetadata();
        }

        public static CopyColumnsTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(input, nameof(input));

            // *** Binary format ***
            // <handled in ctors>
            var h = env.Register(RegistrationName);
            return h.Apply("Loading Model", ch => new CopyColumnsTransform(h, ctx, input));
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>

            SaveBase(ctx);
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return Infos[iinfo].TypeSrc;
        }

        private void SetMetadata()
        {
            var md = Metadata;
            for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
            {
                // REVIEW: Should we filter out score set metadata or any others?
                using (var bldr = md.BuildMetadata(iinfo, Source.Schema, Infos[iinfo].Source))
                {
                    // No metadata to add.
                }
            }
            md.Seal();
        }

        protected override bool WantParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate);
            // Parallel doesn't matter to this transform.
            return false;
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            disposer = null;
            int col = Infos[iinfo].Source;
            var typeSrc = input.Schema.GetColumnType(col);

            Func<int, ValueGetter<int>> del = input.GetGetter<int>;
            var methodInfo = del.GetMethodInfo().GetGenericMethodDefinition().MakeGenericMethod(typeSrc.RawType);
            return (Delegate)methodInfo.Invoke(input, new object[] { col });
        }
    }

    public sealed class CopyColumnsEstimator : IEstimator<CopyColumnsTransformer>
    {
        private readonly (string Source, string Name)[] _columns;
        private readonly IHost _host;

        public CopyColumnsEstimator(IHostEnvironment env, string input, string output)
        {
            Contracts.CheckNonWhiteSpace(input, nameof(input));
            Contracts.CheckNonWhiteSpace(output, nameof(output));
            _columns = new (string, string)[1] { (input, output) };
            _host = env.Register("CopyColumnsEstimator");
        }

        public CopyColumnsEstimator(IHostEnvironment env, (string Source, string Name)[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            var newNames = new HashSet<string>();
            foreach (var column in columns)
            {
                if (newNames.Contains(column.Name))
                    throw Contracts.ExceptUserArg(nameof(columns), $"New column {column.Name} specified multiple times");
                newNames.Add(column.Name);
            }
            _columns = columns;
            _host = env.Register("CopyColumnsEstimator");
        }

        public CopyColumnsTransformer Fit(IDataView input)
        {
            // invoke schema validation.
            GetOutputSchema(SchemaShape.Create(input.Schema));
            return new CopyColumnsTransformer(_host, _columns);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Contracts.CheckValue(inputSchema, nameof(inputSchema));
            Contracts.CheckValue(inputSchema.Columns, nameof(inputSchema.Columns));
            var originDic = inputSchema.Columns.ToDictionary(x => x.Name);
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach ((string source, string name) pair in _columns)
            {
                if (originDic.ContainsKey(pair.source))
                {
                    var col = originDic[pair.source].CloneWithNewName(pair.name);
                    resultDic[pair.name] = col;
                }
                else
                {
                    throw Contracts.ExceptUserArg(nameof(inputSchema), $"{pair.source} not found in {nameof(inputSchema)}");
                }
            }
            var result = new SchemaShape(originDic.Values.ToArray());
            return result;
        }
    }

    public sealed class CopyColumnsTransformer : ITransformer, ICanSaveModel
    {
        private readonly (string Source, string Name)[] _columns;
        private readonly IHost _host;

        public CopyColumnsTransformer(IHostEnvironment env, (string Source, string Name)[] columns)
        {
            _columns = columns;
            _host = env.Register("CopyColumnsTransformer");
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            //Validate schema.
            return Transform(new EmptyDataView(_host, inputSchema)).Schema;
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "COPYCOLT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public const string LoaderSignature = "CopyTransform";

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name
            ctx.Writer.Write(_columns.Length);
            foreach (var column in _columns)
            {
                ctx.SaveNonEmptyString(column.Name);
                ctx.SaveNonEmptyString(column.Source);
            }
        }
        public static CopyColumnsTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var lenght = ctx.Reader.ReadInt32();
            var columns = new (string Source, string Name)[lenght];
            for (int i = 0; i < lenght; i++)
            {
                columns[i].Name = ctx.LoadNonEmptyString();
                columns[i].Source = ctx.LoadNonEmptyString();
            }
            return new CopyColumnsTransformer(env, columns);
        }

        public IDataView Transform(IDataView input)
        {
            var mapper = new CopyColumnsRowMapper(_host, input.Schema, _columns);
            return new RowToRowMapperTransform(_host, input, mapper);
        }

    }
    public class CopyColumnsRowMapper : IRowMapper
    {
        private readonly ISchema _schema;
        private readonly HashSet<int> _originalColumnSources;
        private (string Source, string Name)[] _columns;
        private readonly IHost _host;
        public const string LoaderSignature = "CopyColumnsRowMapper";

        public CopyColumnsRowMapper(IHostEnvironment env, ISchema schema, (string Source, string Name)[] columns)
        {
            _host = env.Register(LoaderSignature);
            env.CheckValue(schema, nameof(schema));
            env.CheckValue(columns, nameof(columns));
            _schema = schema;
            _columns = columns;
            _originalColumnSources = new HashSet<int>();
            var sources = new HashSet<string>();
            foreach (var source in columns.Select(x => x.Source))
                sources.Add(source);
            for (int i = 0; i < _schema.ColumnCount; i++)
                if (sources.Contains(_schema.GetColumnName(i)))
                    _originalColumnSources.Add(i);
        }

        public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
        {
            var result = new Delegate[_columns.Length];
            int i = 0;
            foreach (var column in _columns)
            {
                // validate activeOutput.
                input.Schema.TryGetColumnIndex(column.Source, out int colIndex);
                var type = input.Schema.GetColumnType(colIndex);

                result[i++] = Utils.MarshalInvoke(MakeGetter<int>, type.RawType, input, colIndex);
            }
            disposer = null;
            return result;
        }

        private Delegate MakeGetter<T>(IRow row, int src) => row.GetGetter<T>(src);

        public Func<int, bool> GetDependencies(Func<int, bool> activeOutput) => (col) => (activeOutput(col) && _originalColumnSources.Contains(col));

        public RowMapperColumnInfo[] GetOutputColumns()
        {
            var result = new RowMapperColumnInfo[_columns.Length];
            // Do I need return all columns or only output???
            for (int i = 0; i < _columns.Length; i++)
            {
                _schema.TryGetColumnIndex(_columns[i].Source, out int colIndex);
                // how to get metadata?
                result[i] = new RowMapperColumnInfo(_columns[i].Name, _schema.GetColumnType(colIndex), null);
            }
            return result;
        }

        public static CopyColumnsRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var lenght = ctx.Reader.ReadInt32();
            var columns = new (string Source, string Name)[lenght];
            for (int i = 0; i < lenght; i++)
            {
                columns[i].Name = ctx.LoadNonEmptyString();
                columns[i].Source = ctx.LoadNonEmptyString();
            }
            return new CopyColumnsRowMapper(env, schema, columns);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "COPYROWM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            ctx.Writer.Write(_columns.Length);
            foreach (var column in _columns)
            {
                ctx.SaveNonEmptyString(column.Name);
                ctx.SaveNonEmptyString(column.Source);
            }
        }
    }
}
