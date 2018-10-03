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

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(IDataTransform), typeof(CopyColumnsTransform), null, typeof(SignatureLoadDataTransform),
    CopyColumnsTransform.UserName, CopyColumnsTransform.LoaderSignature)]

[assembly: LoadableClass(CopyColumnsTransform.Summary, typeof(CopyColumnsTransform), null, typeof(SignatureLoadModel),
    CopyColumnsTransform.UserName, CopyColumnsTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(CopyColumnsTransform), null, typeof(SignatureLoadRowMapper),
    CopyColumnsTransform.UserName, CopyColumnsTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    public sealed class CopyColumnsEstimator : TrivialEstimator<CopyColumnsTransform>
    {
        public CopyColumnsEstimator(IHostEnvironment env, string input, string output) :
            this(env, (input, output))
        {
        }

        public CopyColumnsEstimator(IHostEnvironment env, params (string source, string name)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CopyColumnsEstimator)), new CopyColumnsTransform(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var resultDic = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var column in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(column.Source, out var originalColumn))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", column.Source);
                var col = new SchemaShape.Column(column.Name, originalColumn.Kind, originalColumn.ItemType, originalColumn.IsKey, originalColumn.Metadata);
                resultDic[column.Name] = col;
            }
            return new SchemaShape(resultDic.Values);
        }
    }

    public sealed class CopyColumnsTransform : OneToOneTransformerBase
    {
        public const string LoaderSignature = "CopyTransform";
        private const string RegistrationName = "CopyColumns";
        public const string Summary = "Copy a source column to a new column.";
        public const string UserName = "Copy Columns Transform";
        public const string ShortName = "Copy";

        public IReadOnlyCollection<(string Source, string Name)> Columns => ColumnPairs.AsReadOnly();

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "COPYCOLT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(CopyColumnsTransform).Assembly.FullName);
        }

        public CopyColumnsTransform(IHostEnvironment env, params (string source, string name)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(CopyColumnsTransform)), columns)
        {
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

        /// <summary>
        /// Factory method to create from arguments
        /// </summary>
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));

            var transformer = new CopyColumnsTransform(env, args.Column.Select(x => (x.Source, x.Name)).ToArray());
            return transformer.MakeDataTransform(input);
        }

        /// <summary>
        /// Factory method for SignatureLoadModel.
        /// </summary>
        private static CopyColumnsTransform Create(IHostEnvironment env, ModelLoadContext ctx)
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

        /// <summary>
        /// Factory method for SignatureLoadDataTransform.
        /// </summary>
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        /// <summary>
        /// Factory method for SignatureLoadRowMapper.
        /// </summary>
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   string: output (Name) column name
            //   string: input (Source) column name
            ctx.Writer.Write(ColumnPairs.Length);
            foreach (var column in ColumnPairs)
            {
                ctx.SaveNonEmptyString(column.output);
                ctx.SaveNonEmptyString(column.input);
            }
        }

        protected override IRowMapper MakeRowMapper(ISchema inputSchema)
            => new Mapper(this, inputSchema, ColumnPairs);

        private sealed class Mapper : MapperBase
        {
            private readonly ISchema _schema;
            private readonly (string Source, string Name)[] _columns;

            public Mapper(CopyColumnsTransform parent, ISchema inputSchema, (string Source, string Name)[] columns)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _schema = inputSchema;
                _columns = columns;
            }

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _columns.Length);
                disposer = null;

                input.Schema.TryGetColumnIndex(_columns[iinfo].Source, out int colIndex);
                var type = input.Schema.GetColumnType(colIndex);
                return Utils.MarshalInvoke(MakeGetter<int>, type.RawType,  input, colIndex);
            }

            private Delegate MakeGetter<T>(IRow input, int iinfo)
                => input.GetGetter<T>(iinfo);

            public override RowMapperColumnInfo[] GetOutputColumns()
            {
                var result = new RowMapperColumnInfo[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    _schema.TryGetColumnIndex(_columns[i].Source, out int colIndex);
                    var colType = _schema.GetColumnType(colIndex);
                    var meta = new RowColumnUtils.MetadataRow(_schema, colIndex, x => true);
                    result[i] = new RowMapperColumnInfo(_columns[i].Name, colType, meta);
                }
                return result;
            }
        }
    }
}
