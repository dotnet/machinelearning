// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Model.Onnx;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ColumnCopyingTransformer.Summary, typeof(IDataTransform), typeof(ColumnCopyingTransformer),
    typeof(ColumnCopyingTransformer.Arguments), typeof(SignatureDataTransform),
    ColumnCopyingTransformer.UserName, "CopyColumns", "CopyColumnsTransform", ColumnCopyingTransformer.ShortName,
    DocName = "transform/CopyColumnsTransformer.md")]

[assembly: LoadableClass(ColumnCopyingTransformer.Summary, typeof(IDataTransform), typeof(ColumnCopyingTransformer), null, typeof(SignatureLoadDataTransform),
    ColumnCopyingTransformer.UserName, ColumnCopyingTransformer.LoaderSignature)]

[assembly: LoadableClass(ColumnCopyingTransformer.Summary, typeof(ColumnCopyingTransformer), null, typeof(SignatureLoadModel),
    ColumnCopyingTransformer.UserName, ColumnCopyingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ColumnCopyingTransformer), null, typeof(SignatureLoadRowMapper),
    ColumnCopyingTransformer.UserName, ColumnCopyingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    public sealed class ColumnCopyingEstimator : TrivialEstimator<ColumnCopyingTransformer>
    {
        public ColumnCopyingEstimator(IHostEnvironment env, string input, string output) :
            this(env, (input, output))
        {
        }

        public ColumnCopyingEstimator(IHostEnvironment env, params (string source, string name)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ColumnCopyingEstimator)), new ColumnCopyingTransformer(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var resultDic = inputSchema.ToDictionary(x => x.Name);
            foreach (var (Source, Name) in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(Source, out var originalColumn))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Source);
                var col = new SchemaShape.Column(Name, originalColumn.Kind, originalColumn.ItemType, originalColumn.IsKey, originalColumn.Metadata);
                resultDic[Name] = col;
            }
            return new SchemaShape(resultDic.Values);
        }
    }

    public sealed class ColumnCopyingTransformer : OneToOneTransformerBase
    {
        public const string LoaderSignature = "CopyTransform";
        internal const string Summary = "Copy a source column to a new column.";
        internal const string UserName = "Copy Columns Transform";
        internal const string ShortName = "Copy";

        public IReadOnlyCollection<(string Source, string Name)> Columns => ColumnPairs.AsReadOnly();

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "COPYCOLT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ColumnCopyingTransformer).Assembly.FullName);
        }

        public ColumnCopyingTransformer(IHostEnvironment env, params (string source, string name)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ColumnCopyingTransformer)), columns)
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

        // Factory method to create from arguments
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));

            var transformer = new ColumnCopyingTransformer(env, args.Column.Select(x => (x.Source, x.Name)).ToArray());
            return transformer.MakeDataTransform(input);
        }

        // Factory method for SignatureLoadModel.
        private static ColumnCopyingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
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
            return new ColumnCopyingTransformer(env, columns);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        public override void Save(ModelSaveContext ctx)
        {
            ctx.SetVersionInfo(GetVersionInfo());
            SaveColumns(ctx);
        }

        private protected override IRowMapper MakeRowMapper(Schema inputSchema)
            => new Mapper(this, inputSchema, ColumnPairs);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private readonly Schema _schema;
            private readonly (string Source, string Name)[] _columns;

            public bool CanSaveOnnx(OnnxContext ctx) => ctx.GetOnnxVersion() == OnnxVersion.Experimental;

            internal Mapper(ColumnCopyingTransformer parent, Schema inputSchema, (string Source, string Name)[] columns)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _schema = inputSchema;
                _columns = columns;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _columns.Length);
                disposer = null;

                Delegate MakeGetter<T>(Row row, int index)
                    => input.GetGetter<T>(index);

                input.Schema.TryGetColumnIndex(_columns[iinfo].Source, out int colIndex);
                var type = input.Schema[colIndex].Type;
                return Utils.MarshalInvoke(MakeGetter<int>, type.RawType, input, colIndex);
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new Schema.DetachedColumn[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    var srcCol = _schema[_columns[i].Source];
                    result[i] = new Schema.DetachedColumn(_columns[i].Name, srcCol.Type, srcCol.Metadata);
                }
                return result;
            }

            public void SaveAsOnnx(OnnxContext ctx)
            {
                var opType = "CSharp";

                foreach (var column in _columns)
                {
                    var srcVariableName = ctx.GetVariableName(column.Source);
                    _schema.TryGetColumnIndex(column.Source, out int colIndex);
                    var dstVariableName = ctx.AddIntermediateVariable(_schema[colIndex].Type, column.Name);
                    var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                    node.AddAttribute("type", LoaderSignature);
                }
            }
        }
    }
}
