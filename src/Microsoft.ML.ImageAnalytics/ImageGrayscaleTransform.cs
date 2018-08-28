// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;

[assembly: LoadableClass(ImageGrayscaleTransform.Summary, typeof(IDataTransform), typeof(ImageGrayscaleTransform), typeof(ImageGrayscaleTransform.Arguments), typeof(SignatureDataTransform),
    ImageGrayscaleTransform.UserName, "ImageGrayscaleTransform", "ImageGrayscale")]

[assembly: LoadableClass(ImageGrayscaleTransform.Summary, typeof(IDataTransform), typeof(ImageGrayscaleTransform), null, typeof(SignatureLoadDataTransform),
    ImageGrayscaleTransform.UserName, ImageGrayscaleTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageGrayscaleTransform), null, typeof(SignatureLoadModel),
    ImageGrayscaleTransform.UserName, ImageGrayscaleTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageGrayscaleTransform.Mapper), null, typeof(SignatureLoadRowMapper),
    ImageGrayscaleTransform.UserName, ImageGrayscaleTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    // REVIEW: Should it be separate transform or part of ImageResizerTransform?
    /// <summary>
    /// Transform which takes one or many columns of <see cref="ImageType"/> type in IDataView and
    /// convert them to greyscale representation of the same image.
    /// </summary>
    public sealed class ImageGrayscaleTransform : ITransformer, ICanSaveModel
    {
        public sealed class Column : OneToOneColumn
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

        public class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        internal const string Summary = "Convert image into grayscale.";

        internal const string UserName = "Image Greyscale Transform";
        public const string LoaderSignature = "ImageGrayscaleTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGGRAYT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "ImageGrayscale";
        private readonly IHost _host;
        private readonly (string input, string output)[] _columns;

        public (string input, string output)[] Columns => _columns;

        public ImageGrayscaleTransform(IHostEnvironment env, params (string input, string output)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(columns, nameof(columns));

            _columns = columns.ToArray();
        }

        // Factory method for SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            var transformer = new ImageGrayscaleTransform(env, args.Column.Select(x => (x.Source ?? x.Name, x.Name)).ToArray());
            return new RowToRowMapperTransform(env, input, transformer.MakeRowMapper(input.Schema));
        }

        public ImageGrayscaleTransform(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name

            int n = ctx.Reader.ReadInt32();
            _columns = new (string input, string output)[n];
            for (int i = 0; i < n; i++)
            {
                string output = ctx.LoadNonEmptyString();
                string input = ctx.LoadNonEmptyString();
                _columns[i] = (input, output);
            }
        }

        // Factory method for SignatureLoadDataTransform.
        public static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            var transformer = new ImageGrayscaleTransform(env, ctx);
            return new RowToRowMapperTransform(env, input, transformer.MakeRowMapper(input.Schema));
        }

        public void Save(ModelSaveContext ctx) => SaveContents(_host, ctx, _columns);

        private static void SaveContents(IHostEnvironment env, ModelSaveContext ctx, (string input, string output)[] columns)
        {
            Contracts.AssertValue(env);
            env.CheckValue(ctx, nameof(ctx));
            Contracts.AssertValue(columns);

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name

            ctx.Writer.Write(columns.Length);
            for (int i = 0; i < columns.Length; i++)
            {
                ctx.SaveNonEmptyString(columns[i].output);
                ctx.SaveNonEmptyString(columns[i].input);
            }
        }

        private static readonly ColorMatrix _grayscaleColorMatrix = new ColorMatrix(
                new float[][]
                {
                    new float[] {.3f, .3f, .3f, 0, 0},
                    new float[] {.59f, .59f, .59f, 0, 0},
                    new float[] {.11f, .11f, .11f, 0, 0},
                    new float[] {0, 0, 0, 1, 0},
                    new float[] {0, 0, 0, 0, 1}
                });

        private IRowMapper MakeRowMapper(ISchema schema)
            => new Mapper(_host, _columns, schema);

        private static void CheckInput(IExceptionContext ctx, ISchema inputSchema, string input, out int srcCol)
        {
            Contracts.AssertValueOrNull(ctx);
            Contracts.AssertValue(inputSchema);
            Contracts.AssertNonEmpty(input);

            if (!inputSchema.TryGetColumnIndex(input, out srcCol))
                throw ctx.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
            if (!(inputSchema.GetColumnType(srcCol) is ImageType))
                throw ctx.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "image", inputSchema.GetColumnType(srcCol).ToString());
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            // Check that all the input columns are present and are images.
            foreach (var column in _columns)
                CheckInput(_host, inputSchema, column.input, out int col);

            return Transform(new EmptyDataView(_host, inputSchema)).Schema;
        }

        public IDataView Transform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            var mapper = MakeRowMapper(input.Schema);
            return new RowToRowMapperTransform(_host, input, mapper);
        }

        internal sealed class Mapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly (string input, string output)[] _columns;
            private readonly ISchema _inputSchema;
            private readonly Dictionary<int, int> _colMapNewToOld;

            public Mapper(IHostEnvironment env, (string input, string output)[] columns, ISchema inputSchema)
            {
                Contracts.AssertValue(env);
                _host = env.Register(nameof(Mapper));
                _host.AssertValue(columns);
                _host.AssertValue(inputSchema);

                _colMapNewToOld = new Dictionary<int, int>();
                for (int i = 0; i < columns.Length; i++)
                {
                    CheckInput(_host, inputSchema, columns[i].input, out int srcCol);
                    _colMapNewToOld.Add(i, srcCol);
                }
                _columns = columns;
                _inputSchema = inputSchema;
            }

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                var active = new bool[_inputSchema.ColumnCount];
                foreach (var pair in _colMapNewToOld)
                    if (activeOutput(pair.Key))
                        active[pair.Value] = true;
                return col => active[col];
            }

            public RowMapperColumnInfo[] GetOutputColumns()
                => _columns.Select((x, idx) => new RowMapperColumnInfo(x.output, _inputSchema.GetColumnType(_colMapNewToOld[idx]), null)).ToArray();

            public void Save(ModelSaveContext ctx) => SaveContents(_host, ctx, _columns);

            public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                env.CheckValue(inputSchema, nameof(inputSchema));
                var transformer = new ImageGrayscaleTransform(env, ctx);
                return transformer.MakeRowMapper(inputSchema);
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                _host.Assert(input.Schema == _inputSchema);
                var result = new Delegate[_columns.Length];
                var disposers = new Action[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    int srcCol = _colMapNewToOld[i];
                    result[i] = MakeGetter(input, i, out disposers[i]);
                }
                disposer = () =>
                {
                    foreach (var act in disposers)
                        act();
                };
                return result;
            }

            private Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                _host.AssertValue(input);
                _host.Assert(0 <= iinfo && iinfo < _columns.Length);

                var src = default(Bitmap);
                var getSrc = input.GetGetter<Bitmap>(_colMapNewToOld[iinfo]);

                disposer =
                    () =>
                    {
                        if (src != null)
                        {
                            src.Dispose();
                            src = null;
                        }
                    };

                ValueGetter<Bitmap> del =
                    (ref Bitmap dst) =>
                    {
                        if (dst != null)
                            dst.Dispose();

                        getSrc(ref src);
                        if (src == null || src.Height <= 0 || src.Width <= 0)
                            return;

                        dst = new Bitmap(src.Width, src.Height);
                        ImageAttributes attributes = new ImageAttributes();
                        attributes.SetColorMatrix(_grayscaleColorMatrix);
                        var srcRectangle = new Rectangle(0, 0, src.Width, src.Height);
                        using (var g = Graphics.FromImage(dst))
                        {
                            g.DrawImage(src, srcRectangle, 0, 0, src.Width, src.Height, GraphicsUnit.Pixel, attributes);
                        }
                        Contracts.Assert(dst.Width == src.Width && dst.Height == src.Height);
                    };

                return del;
            }
        }
    }

    public sealed class ImageGrayscaleEstimator: TrivialEstimator<ImageGrayscaleTransform>
    {
        public ImageGrayscaleEstimator(IHostEnvironment env, params (string input, string output)[] columns)
            :base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageGrayscaleEstimator)), new ImageGrayscaleTransform(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                var col = inputSchema.FindColumn(colInfo.input);

                if (col == null)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input);
                if (!(col.ItemType is ImageType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.input, new ImageType().ToString(), col.GetTypeString());

                result[colInfo.output] = new SchemaShape.Column(colInfo.output, col.Kind, col.ItemType, col.IsKey, col.MetadataKinds);
            }

            return new SchemaShape(result.Values);
        }
    }
}
