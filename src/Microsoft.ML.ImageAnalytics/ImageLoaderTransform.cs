// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;

[assembly: LoadableClass(ImageLoaderTransformer.Summary, typeof(IDataTransform), typeof(ImageLoaderTransformer), typeof(ImageLoaderTransformer.Arguments), typeof(SignatureDataTransform),
    ImageLoaderTransformer.UserName, "ImageLoaderTransform", "ImageLoader")]

[assembly: LoadableClass(ImageLoaderTransformer.Summary, typeof(IDataTransform), typeof(ImageLoaderTransformer), null, typeof(SignatureLoadDataTransform),
   ImageLoaderTransformer.UserName, ImageLoaderTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageLoaderTransformer), null, typeof(SignatureLoadModel), "", ImageLoaderTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageLoaderTransformer), null, typeof(SignatureLoadRowMapper), "", ImageLoaderTransformer.LoaderSignature)]

namespace Microsoft.ML.ImageAnalytics
{
    /// <summary>
    /// Transform which takes one or many columns of type ReadOnlyMemory and loads them as <see cref="ImageType"/>
    /// </summary>
    public sealed class ImageLoaderTransformer : OneToOneTransformerBase
    {
        public sealed class Column : OneToOneColumn
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

        public sealed class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Folder where to search for images", ShortName = "folder")]
            public string ImageFolder;
        }

        internal const string Summary = "Load images from files.";
        internal const string UserName = "Image Loader Transform";
        internal const string LoaderSignature = "ImageLoaderTransform";

        public readonly string ImageFolder;

        public IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Load images in memory.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        public ImageLoaderTransformer(IHostEnvironment env, string imageFolder = null, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoaderTransformer)), columns)
        {
            ImageFolder = imageFolder;
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView data)
        {
            return new ImageLoaderTransformer(env, args.ImageFolder, args.Columns.Select(x => (x.Name, x.Source ?? x.Name)).ToArray())
                .MakeDataTransform(data);
        }

        // Factory method for SignatureLoadModel.
        private static ImageLoaderTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel(GetVersionInfo());
            return new ImageLoaderTransformer(env.Register(nameof(ImageLoaderTransformer)), ctx);
        }

        private ImageLoaderTransformer(IHost host, ModelLoadContext ctx)
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
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        protected override void CheckInputColumn(Schema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema[srcCol].Type is TextType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, TextType.Instance.ToString(), inputSchema[srcCol].Type.ToString());
        }

        public override void Save(ModelSaveContext ctx)
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
                loaderAssemblyName: typeof(ImageLoaderTransformer).Assembly.FullName);
        }

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ImageLoaderTransformer _parent;
            private readonly ImageType _imageType;

            public Mapper(ImageLoaderTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _imageType = new ImageType();
                _parent = parent;
            }

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                disposer = null;
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(ColMapNewToOld[iinfo]);
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

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select(x => new Schema.DetachedColumn(x.outputColumnName, _imageType, null)).ToArray();
        }
    }

    /// <summary>
    /// Load images in memory.
    /// </summary>
    public sealed class ImageLoadingEstimator : TrivialEstimator<ImageLoaderTransformer>
    {
        private readonly ImageType _imageType;

        /// <summary>
        /// Load images in memory.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        public ImageLoadingEstimator(IHostEnvironment env, string imageFolder, params (string outputColumnName, string inputColumnName)[] columns)
            : this(env, new ImageLoaderTransformer(env, imageFolder, columns))
        {
        }

        private ImageLoadingEstimator(IHostEnvironment env, ImageLoaderTransformer transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingEstimator)), transformer)
        {
            _imageType = new ImageType();
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var (outputColumnName, inputColumnName) in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName);
                if (!(col.ItemType is TextType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName, TextType.Instance.ToString(), col.GetTypeString());

                result[outputColumnName] = new SchemaShape.Column(outputColumnName, SchemaShape.Column.VectorKind.Scalar, _imageType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
