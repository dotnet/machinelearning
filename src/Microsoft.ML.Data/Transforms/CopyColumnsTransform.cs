// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(IDataTransform), typeof(CopyColumnsTransform),
    typeof(CopyColumnsTransform.Arguments), typeof(SignatureDataTransform),
    CopyColumnsTransform.UserName, "CopyColumns", "CopyColumnsTransform", CopyColumnsTransform.ShortName,
    DocName = "transform/CopyColumnsTransformer.md")]

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(IDataView), typeof(CopyColumnsTransform), null, typeof(SignatureLoadDataTransform),
    CopyColumnsTransform.UserName, CopyColumnsTransform.LoaderSignature)]

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(CopyColumnsTransform), null, typeof(SignatureLoadModel),
    CopyColumnsTransform.UserName, CopyColumnsTransform.LoaderSignature)]

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(CopyColumnsRowMapper), null, typeof(SignatureLoadRowMapper),
    CopyColumnsTransform.UserName, CopyColumnsRowMapper.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class CopyColumnsEstimator : IEstimator<CopyColumnsTransform>
    {
        private readonly (string Source, string Name)[] _columns;
        private readonly IHost _host;

        public CopyColumnsEstimator(IHostEnvironment env, string input, string output) :
            this(env, (input, output))
        {
        }

        public CopyColumnsEstimator(IHostEnvironment env, params (string source, string name)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(CopyColumnsEstimator));
            _host.CheckValue(columns, nameof(columns));
            var newNames = new HashSet<string>();
            foreach (var column in columns)
            {
                if (!newNames.Add(column.name))
                    throw Contracts.ExceptUserArg(nameof(columns), $"New column {column.name} specified multiple times");
            }
            _columns = columns;
        }

        public CopyColumnsTransform Fit(IDataView input)
        {
            // Invoke schema validation.
            GetOutputSchema(SchemaShape.Create(input.Schema));
            return new CopyColumnsTransform(_host, _columns);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var column in _columns)
            {
                var originalColumn = inputSchema.FindColumn(column.Source);
                if (originalColumn != null)
                {
                    var col = new SchemaShape.Column(column.Name, originalColumn.Kind, originalColumn.ItemKind, originalColumn.IsKey, originalColumn.MetadataKinds);
                    resultDic[column.Name] = col;
                }
                else
                {
                    throw _host.ExceptParam(nameof(inputSchema), $"{column.Source} not found in {nameof(inputSchema)}");
                }
            }
            return new SchemaShape(resultDic.Values.ToArray());
        }
    }

    public sealed class CopyColumnsTransform : ITransformer, ICanSaveModel
    {
        private readonly (string Source, string Name)[] _columns;
        private readonly IHost _host;
        public const string LoaderSignature = "CopyTransform";
        private const string RegistrationName = "CopyColumns";
        public const string Summary = "Copy a source column to a new column.";
        public const string UserName = "Copy Columns Transform";
        public const string ShortName = "Copy";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "COPYCOLT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public CopyColumnsTransform(IHostEnvironment env, params (string source, string name)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(columns, nameof(columns));
            _columns = columns;
        }

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

        public static IDataView Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            env.CheckValue(input, nameof(input));
            var transformer = Create(env, ctx);
            return transformer.Transform(input);
        }

        public static CopyColumnsTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   string: output column name
            //   string: input column name

            var length = ctx.Reader.ReadInt32();
            var columns = new (string Source, string Name)[length];
            for (int i = 0; i < length; i++)
            {
                columns[i].Name = ctx.LoadNonEmptyString();
                columns[i].Source = ctx.LoadNonEmptyString();
            }
            return new CopyColumnsTransform(env, columns);
        }

        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            var transformer = new CopyColumnsTransform(env, args.Column.Select(x => (x.Source, x.Name)).ToArray());
            return transformer.CreateRowToRowMapper(input);
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            // Validate schema.
            return Transform(new EmptyDataView(_host, inputSchema)).Schema;
        }

        public void Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   string: output column name
            //   string: input column name
            ctx.Writer.Write(_columns.Length);
            foreach (var column in _columns)
            {
                ctx.SaveNonEmptyString(column.Name);
                ctx.SaveNonEmptyString(column.Source);
            }
        }

        private RowToRowMapperTransform CreateRowToRowMapper(IDataView input)
        {
            var mapper = new CopyColumnsRowMapper(_host, input.Schema, _columns);
            return new RowToRowMapperTransform(_host, input, mapper);
        }

        public IDataView Transform(IDataView input)
        {
            return CreateRowToRowMapper(input);
        }
    }

    internal sealed class CopyColumnsRowMapper : IRowMapper
    {
        private readonly ISchema _schema;
        private readonly Dictionary<int, int> _colNewToOldMapping;
        private readonly (string Source, string Name)[] _columns;
        private readonly IHost _host;
        public const string LoaderSignature = "CopyColumnsRowMapper";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "COPYROWM",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public static CopyColumnsRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   string: output column name
            //   string: input column name

            var length = ctx.Reader.ReadInt32();
            var columns = new (string Source, string Name)[length];
            for (int i = 0; i < length; i++)
            {
                columns[i].Name = ctx.LoadNonEmptyString();
                columns[i].Source = ctx.LoadNonEmptyString();
            }
            return new CopyColumnsRowMapper(env, schema, columns);
        }

        public CopyColumnsRowMapper(IHostEnvironment env, ISchema schema, (string source, string name)[] columns)
        {
            _host = env.Register(LoaderSignature);
            env.CheckValue(schema, nameof(schema));
            env.CheckValue(columns, nameof(columns));
            _schema = schema;
            _columns = columns;
            _colNewToOldMapping = new Dictionary<int, int>();
            for (int i = 0; i < _columns.Length; i++)
            {
                if (!_schema.TryGetColumnIndex(_columns[i].Source, out int colIndex))
                {
                    throw _host.ExceptParam(nameof(schema), $"{_columns[i].Source} not found in {nameof(schema)}");
                }
                _colNewToOldMapping.Add(i, colIndex);
            }
        }

        public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
        {
            _host.Assert(input.Schema == _schema);
            var result = new Delegate[_columns.Length];
            for (int i = 0; i < _columns.Length; i++)
            {
                if (!activeOutput(i))
                    continue;
                input.Schema.TryGetColumnIndex(_columns[i].Source, out int colIndex);
                var type = input.Schema.GetColumnType(colIndex);
                result[i] = Utils.MarshalInvoke(MakeGetter<int>, type.RawType, input, colIndex);
            }
            disposer = null;
            return result;
        }

        private Delegate MakeGetter<T>(IRow row, int src) => row.GetGetter<T>(src);

        public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
        {
            var active = new bool[_schema.ColumnCount];
            foreach (var pair in _colNewToOldMapping)
                if (activeOutput(pair.Key))
                    active[pair.Value] = true;
            return col => active[col];
        }

        public RowMapperColumnInfo[] GetOutputColumns()
        {
            var result = new RowMapperColumnInfo[_columns.Length];
            for (int i = 0; i < _columns.Length; i++)
            {
                _schema.TryGetColumnIndex(_columns[i].Source, out int colIndex);
                //REVIEW: Metadata need to be switched to IRow instead of ColumMetadataInfo
                var colMetaInfo = new ColumnMetadataInfo(_columns[i].Name);
                var types = _schema.GetMetadataTypes(colIndex);
                var colType = _schema.GetColumnType(colIndex);
                foreach (var type in types)
                {
                    Utils.MarshalInvoke(AddMetaGetter<int>, type.Value.RawType, colMetaInfo, _schema, type.Key, type.Value, _colNewToOldMapping);
                }
                result[i] = new RowMapperColumnInfo(_columns[i].Name, colType, colMetaInfo);
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
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   string: output column name
            //   string: input column name

            ctx.Writer.Write(_columns.Length);
            foreach (var column in _columns)
            {
                ctx.SaveNonEmptyString(column.Name);
                ctx.SaveNonEmptyString(column.Source);
            }
        }
    }
}
