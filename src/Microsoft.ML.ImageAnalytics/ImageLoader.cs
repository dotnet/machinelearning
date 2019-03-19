// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;

[assembly: LoadableClass(ImageLoadingTransformer.Summary, typeof(IDataTransform), typeof(ImageLoadingTransformer), typeof(ImageLoadingTransformer.Options), typeof(SignatureDataTransform),
    ImageLoadingTransformer.UserName, "ImageLoaderTransform", "ImageLoader")]

[assembly: LoadableClass(ImageLoadingTransformer.Summary, typeof(IDataTransform), typeof(ImageLoadingTransformer), null, typeof(SignatureLoadDataTransform),
   ImageLoadingTransformer.UserName, ImageLoadingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageLoadingTransformer), null, typeof(SignatureLoadModel), "", ImageLoadingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageLoadingTransformer), null, typeof(SignatureLoadRowMapper), "", ImageLoadingTransformer.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// The <see cref="ITransformer"/> produced by fitting an <see cref="IDataView"/> to an <see cref="ImageLoadingEstimator"/>.
    /// </summary>
    /// <remarks>
    /// Calling <see cref="ITransformer.Transform(IDataView)"/> that loads images from the disk.
    /// Loading is the first step of almost every pipeline that does image processing, and further analysis on images.
    /// The images to load need to be in the formats supported by <see cref = "System.Drawing.Bitmap" />.
    /// For end-to-end image processing pipelines, and scenarios in your applications, see the
    /// <a href="https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started"> examples in the machinelearning-samples github repository.</a>
    /// <seealso cref = "ImageEstimatorsCatalog"/>
    /// </remarks>
    public sealed class ImageLoadingTransformer : OneToOneTransformerBase
    {
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
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Folder where to search for images", ShortName = "folder")]
            public string ImageFolder;
        }

        internal const string Summary = "Load images from files.";
        internal const string UserName = "Image Loader Transform";
        internal const string LoaderSignature = "ImageLoaderTransform";

        /// <summary>
        /// The folder to load the images from.
        /// </summary>
        public readonly string ImageFolder;

        /// <summary>
        /// The columns passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Initializes a new instance of <see cref="ImageLoadingTransformer"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingTransformer(IHostEnvironment env, string imageFolder = null, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingTransformer)), columns)
        {
            ImageFolder = imageFolder;
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView data)
        {
            return new ImageLoadingTransformer(env, options.ImageFolder, options.Columns.Select(x => (x.Name, x.Source ?? x.Name)).ToArray())
                .MakeDataTransform(data);
        }

        // Factory method for SignatureLoadModel.
        private static ImageLoadingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel(GetVersionInfo());
            return new ImageLoadingTransformer(env.Register(nameof(ImageLoadingTransformer)), ctx);
        }

        private ImageLoadingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>
            // int: id of image folder

            ImageFolder = ctx.LoadStringOrNull();
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema[srcCol].Type is TextDataViewType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, TextDataViewType.Instance.ToString(), inputSchema[srcCol].Type.ToString());
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // int: id of image folder

            base.SaveColumns(ctx);
            ctx.SaveStringOrNull(ImageFolder);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGLOADR",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImageLoadingTransformer).Assembly.FullName);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ImageLoadingTransformer _parent;
            private readonly ImageType _imageType;

            public Mapper(ImageLoadingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _imageType = new ImageType();
                _parent = parent;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                disposer = null;
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                ReadOnlyMemory<char> src = default;
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
                                if (!string.IsNullOrWhiteSpace(_parent.ImageFolder))
                                    path = Path.Combine(_parent.ImageFolder, path);
                                dst = new Bitmap(path);
                            }
                            catch (Exception)
                            {
                                // REVIEW: We catch everything since the documentation for new Bitmap(string)
                                // appears to be incorrect. When the file isn't found, it throws an ArgumentException,
                                // while the documentation says FileNotFoundException. Not sure what it will throw
                                // in other cases, like corrupted file, etc.
                                throw Host.Except($"Image {src.ToString()} was not found.");
                            }

                            // Check for an incorrect pixel format which indicates the loading failed
                            if (dst.PixelFormat == System.Drawing.Imaging.PixelFormat.DontCare)
                                throw Host.Except($"Failed to load image {src.ToString()}.");
                        }
                    };
                return del;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select(x => new DataViewSchema.DetachedColumn(x.outputColumnName, _imageType, null)).ToArray();
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> that loads images from the disk.
    /// </summary>
    /// <remarks>
    /// Calling <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> in this estimator, produces an <see cref="ImageLoadingTransformer"/>.
    /// Loading is the first step of almost every pipeline that does image processing, and further analysis on images.
    /// The images to load need to be in the formats supported by <see cref = "System.Drawing.Bitmap" />.
    /// For end-to-end image processing pipelines, and scenarios in your applications, see the
    /// <a href = "https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started"> examples in the machinelearning-samples github repository.</a>
    /// <seealso cref="ImageEstimatorsCatalog" />
    /// </remarks>
    public sealed class ImageLoadingEstimator : TrivialEstimator<ImageLoadingTransformer>
    {
        private readonly ImageType _imageType;

        /// <summary>
        /// Load images in memory.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingEstimator(IHostEnvironment env, string imageFolder, params (string outputColumnName, string inputColumnName)[] columns)
            : this(env, new ImageLoadingTransformer(env, imageFolder, columns))
        {
        }

        internal ImageLoadingEstimator(IHostEnvironment env, ImageLoadingTransformer transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingEstimator)), transformer)
        {
            _imageType = new ImageType();
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var (outputColumnName, inputColumnName) in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName);
                if (!(col.ItemType is TextDataViewType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName, TextDataViewType.Instance.ToString(), col.GetTypeString());

                result[outputColumnName] = new SchemaShape.Column(outputColumnName, SchemaShape.Column.VectorKind.Scalar, _imageType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
