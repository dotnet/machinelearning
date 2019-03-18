// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ColumnCopyingTransformer.Summary, typeof(IDataTransform), typeof(ColumnCopyingTransformer),
    typeof(ColumnCopyingTransformer.Options), typeof(SignatureDataTransform),
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
    /// <summary>
    /// <see cref="ColumnCopyingEstimator"/> copies the input column to another column named as specified in the parameters of the transformation.
    /// </summary>
    public sealed class ColumnCopyingEstimator : TrivialEstimator<ColumnCopyingTransformer>
    {
        [BestFriend]
        internal ColumnCopyingEstimator(IHostEnvironment env, string outputColumnName, string inputColumnName) :
            this(env, (outputColumnName, inputColumnName))
        {
        }

        [BestFriend]
        internal ColumnCopyingEstimator(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ColumnCopyingEstimator)), new ColumnCopyingTransformer(env, columns))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            var resultDic = inputSchema.ToDictionary(x => x.Name);
            foreach (var (outputColumnName, inputColumnName) in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(inputColumnName, out var originalColumn))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName);
                var col = new SchemaShape.Column(outputColumnName, originalColumn.Kind, originalColumn.ItemType, originalColumn.IsKey, originalColumn.Annotations);
                resultDic[outputColumnName] = col;
            }
            return new SchemaShape(resultDic.Values);
        }
    }

    public sealed class ColumnCopyingTransformer : OneToOneTransformerBase
    {
        [BestFriend]
        internal const string LoaderSignature = "CopyTransform";
        internal const string Summary = "Copy a source column to a new column.";
        internal const string UserName = "Copy Columns Transform";
        internal const string ShortName = "Copy";

        /// <summary>
        /// Names of output and input column pairs on which the transformation is applied.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

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

        internal ColumnCopyingTransformer(IHostEnvironment env, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ColumnCopyingTransformer)), columns)
        {
        }

        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;
        }

        // Factory method corresponding to SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));

            var transformer = new ColumnCopyingTransformer(env, options.Columns.Select(x => (x.Name, x.Source)).ToArray());
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
            var columns = new (string outputColumnName, string inputColumnName)[length];
            for (int i = 0; i < length; i++)
            {
                columns[i].outputColumnName = ctx.LoadNonEmptyString();
                columns[i].inputColumnName = ctx.LoadNonEmptyString();
            }
            return new ColumnCopyingTransformer(env, columns);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            ctx.SetVersionInfo(GetVersionInfo());
            SaveColumns(ctx);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema)
            => new Mapper(this, inputSchema, ColumnPairs);

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private readonly DataViewSchema _schema;
            private readonly (string outputColumnName, string inputColumnName)[] _columns;

            public bool CanSaveOnnx(OnnxContext ctx) => ctx.GetOnnxVersion() == OnnxVersion.Experimental;

            internal Mapper(ColumnCopyingTransformer parent, DataViewSchema inputSchema, (string outputColumnName, string inputColumnName)[] columns)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _schema = inputSchema;
                _columns = columns;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _columns.Length);
                disposer = null;

                Delegate MakeGetter<T>(DataViewRow row, int index)
                    => input.GetGetter<T>(input.Schema[index]);

                input.Schema.TryGetColumnIndex(_columns[iinfo].inputColumnName, out int colIndex);
                var type = input.Schema[colIndex].Type;
                return Utils.MarshalInvoke(MakeGetter<int>, type.RawType, input, colIndex);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                var result = new DataViewSchema.DetachedColumn[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    var srcCol = _schema[_columns[i].inputColumnName];
                    result[i] = new DataViewSchema.DetachedColumn(_columns[i].outputColumnName, srcCol.Type, srcCol.Annotations);
                }
                return result;
            }

            public void SaveAsOnnx(OnnxContext ctx)
            {
                var opType = "CSharp";

                foreach (var column in _columns)
                {
                    var srcVariableName = ctx.GetVariableName(column.inputColumnName);
                    _schema.TryGetColumnIndex(column.inputColumnName, out int colIndex);
                    var dstVariableName = ctx.AddIntermediateVariable(_schema[colIndex].Type, column.outputColumnName);
                    var node = ctx.CreateNode(opType, srcVariableName, dstVariableName, ctx.GetNodeName(opType));
                    node.AddAttribute("type", LoaderSignature);
                }
            }
        }
    }
}
