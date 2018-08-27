// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Core.Data;
using System.Collections.Generic;
using System.Linq;

[assembly: LoadableClass(ImageLoaderTransform.Summary, typeof(IDataTransform), typeof(ImageLoaderTransform), typeof(ImageLoaderTransform.Arguments), typeof(SignatureDataTransform),
    ImageLoaderTransform.UserName, "ImageLoaderTransform", "ImageLoader")]

[assembly: LoadableClass(ImageLoaderTransform.Summary, typeof(IDataTransform), typeof(ImageLoaderTransform), null, typeof(SignatureLoadDataTransform),
   ImageLoaderTransform.UserName, ImageLoaderTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageLoaderTransform), null, typeof(SignatureLoadModel), "", ImageLoaderTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageLoaderTransform.Mapper), null, typeof(SignatureLoadRowMapper), "", ImageLoaderTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    public abstract class TrivialEstimator<TTransformer> : IEstimator<TTransformer>
        where TTransformer : class, ITransformer
    {
        protected readonly IHost Host;
        protected readonly TTransformer Transformer;

        protected TrivialEstimator(IHost host, TTransformer transformer)
        {
            Contracts.AssertValue(host);

            Host = host;
            Host.CheckValue(transformer, nameof(transformer));
            Transformer = transformer;
        }

        public TTransformer Fit(IDataView input) => Transformer;

        public abstract SchemaShape GetOutputSchema(SchemaShape inputSchema);
    }

    /// <summary>
    /// Transform which takes one or many columns of type <see cref="DvText"/> and loads them as <see cref="ImageType"/>
    /// </summary>
    public sealed class ImageLoaderTransform : ITransformer, ICanSaveModel
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
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)",
                ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Folder where to search for images", ShortName = "folder")]
            public string ImageFolder;
        }

        internal const string Summary = "Load images from files.";
        internal const string UserName = "Image Loader Transform";
        public const string LoaderSignature = "ImageLoaderTransform";

        private readonly string _imageFolder;
        private readonly (string input, string output)[] _columns;
        private readonly IHost _host;

        public IReadOnlyCollection<(string input, string output)> Columns => _columns.AsReadOnly();

        public ImageLoaderTransform(IHostEnvironment env, string imageFolder, params (string input, string output)[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(ImageLoaderTransform));
            _host.CheckValueOrNull(imageFolder);
            _host.CheckValue(columns, nameof(columns));

            _imageFolder = imageFolder;

            var newNames = new HashSet<string>();
            foreach (var column in columns)
            {
                _host.CheckNonEmpty(column.input, nameof(columns));
                _host.CheckNonEmpty(column.output, nameof(columns));

                if (!newNames.Add(column.output))
                    throw Contracts.ExceptParam(nameof(columns), $"Output column '{column.output}' specified multiple times");
            }
            _columns = columns;
        }

        public static ImageLoaderTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name
            // int: id of image folder

            int n = ctx.Reader.ReadInt32();
            var columns = new (string input, string output)[n];
            for (int i = 0; i < n; i++)
            {
                string output = ctx.LoadNonEmptyString();
                string input = ctx.LoadNonEmptyString();
                columns[i] = (input, output);
            }

            string imageFolder = ctx.LoadStringOrNull();

            return new ImageLoaderTransform(env, imageFolder, columns);
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            // Check that all the input columns are present and are scalar texts.
            foreach (var (input, output) in _columns)
                CheckInput(_host, inputSchema, input, out int col);

            return Transform(new EmptyDataView(_host, inputSchema)).Schema;
        }

        private static void CheckInput(IExceptionContext ctx, ISchema inputSchema, string input, out int srcCol)
        {
            Contracts.AssertValueOrNull(ctx);
            Contracts.AssertValue(inputSchema);
            Contracts.AssertNonEmpty(input);

            if (!inputSchema.TryGetColumnIndex(input, out srcCol))
                throw ctx.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
            if (!inputSchema.GetColumnType(srcCol).IsText)
                throw ctx.ExceptSchemaMismatch(nameof(inputSchema), "input", input, TextType.Instance.ToString(), inputSchema.GetColumnType(srcCol).ToString());
        }

        public IDataView Transform(IDataView input) => CreateDataTransform(input);

        public void Save(ModelSaveContext ctx) => SaveContents(ctx, _imageFolder, _columns);

        private static void SaveContents(ModelSaveContext ctx, string imageFolder, (string input, string output)[] columns)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name
            // int: id of image folder

            ctx.Writer.Write(columns.Length);
            foreach (var (input, output) in columns)
            {
                ctx.SaveNonEmptyString(output);
                ctx.SaveNonEmptyString(input);
            }
            ctx.SaveStringOrNull(imageFolder);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGLOADR",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature);
        }

        public static IDataTransform Create(IHostEnvironment env, ImageLoaderTransform.Arguments args, IDataView data)
        {
            return new ImageLoaderTransform(env, args.ImageFolder, args.Column.Select(x => (x.Source ?? x.Name, x.Name)).ToArray())
                .CreateDataTransform(data);
        }

        private IDataTransform CreateDataTransform(IDataView input)
        {
            _host.CheckValue(input, nameof(input));

            var mapper = new Mapper(_host, _imageFolder, _columns, input.Schema);
            return new RowToRowMapperTransform(_host, input, mapper);
        }

        internal sealed class Mapper : IRowMapper
        {
            private readonly IHost _host;
            private readonly string _imageFolder;
            private readonly (string input, string output)[] _columns;
            private readonly Dictionary<int, int> _colMapNewToOld;
            private readonly ISchema _inputSchema;
            private readonly ImageType _imageType;

            public Mapper(IHostEnvironment env, string imageFolder, (string input, string output)[] columns, ISchema schema)
            {
                Contracts.CheckValue(env, nameof(env));
                _host = env.Register(nameof(Mapper));
                _host.CheckValueOrNull(imageFolder);
                _host.CheckValue(columns, nameof(columns));
                _host.CheckValue(schema, nameof(schema));

                _colMapNewToOld = new Dictionary<int, int>();
                for (int i = 0; i < columns.Length; i++)
                {
                    CheckInput(_host, schema, columns[i].input, out int srcCol);
                    _colMapNewToOld.Add(i, srcCol);
                }

                _imageFolder = imageFolder;
                _columns = columns;
                _inputSchema = schema;
                _imageType = new ImageType();
            }

            public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema schema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                env.CheckValue(schema, nameof(schema));

                var xf = ImageLoaderTransform.Create(env, ctx);
                return new Mapper(env, xf._imageFolder, xf._columns, schema);
            }

            public Delegate[] CreateGetters(IRow input, Func<int, bool> activeOutput, out Action disposer)
            {
                _host.Assert(input.Schema == _inputSchema);
                var result = new Delegate[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    if (!activeOutput(i))
                        continue;
                    int srcCol = _colMapNewToOld[i];
                    result[i] = MakeGetter(input, i);
                }
                disposer = null;
                return result;
            }

            private Delegate MakeGetter(IRow input, int iinfo)
            {
                _host.AssertValue(input);
                _host.Assert(0 <= iinfo && iinfo < _columns.Length);

                var getSrc = input.GetGetter<DvText>(_colMapNewToOld[iinfo]);
                DvText src = default;
                ValueGetter<Bitmap> del =
                    (ref Bitmap dst) =>
                    {
                        if (dst != null)
                        {
                            dst.Dispose();
                            dst = null;
                        }

                        getSrc(ref src);

                        if (src.Length > 0)
                        {
                            // Catch exceptions and pass null through. Should also log failures...
                            try
                            {
                                string path = src.ToString();
                                if (!string.IsNullOrWhiteSpace(_imageFolder))
                                    path = Path.Combine(_imageFolder, path);
                                dst = new Bitmap(path);
                            }
                            catch (Exception)
                            {
                                // REVIEW: We catch everything since the documentation for new Bitmap(string)
                                // appears to be incorrect. When the file isn't found, it throws an ArgumentException,
                                // while the documentation says FileNotFoundException. Not sure what it will throw
                                // in other cases, like corrupted file, etc.

                                // REVIEW : Log failures.
                                dst = null;
                            }
                        }
                    };
                return del;
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
                => _columns.Select(x => new RowMapperColumnInfo(x.output, _imageType, null)).ToArray();

            public void Save(ModelSaveContext ctx) => SaveContents(ctx, _imageFolder, _columns);
        }
    }

    public sealed class ImageLoaderEstimator : TrivialEstimator<ImageLoaderTransform>
    {
        private readonly ImageType _imageType;

        public ImageLoaderEstimator(IHostEnvironment env, string imageFolder, params (string input, string output)[] columns)
            : this(env, new ImageLoaderTransform(env, imageFolder, columns))
        {
        }

        public ImageLoaderEstimator(IHostEnvironment env, ImageLoaderTransform transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoaderEstimator)), transformer)
        {
            _imageType = new ImageType();
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var (input, output) in Transformer.Columns)
            {
                var col = inputSchema.FindColumn(input);

                if (col == null)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (!col.ItemType.IsText || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, TextType.Instance.ToString(), col.GetTypeString());

                result[output] = new SchemaShape.Column(output, SchemaShape.Column.VectorKind.Scalar, _imageType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
