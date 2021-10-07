// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
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
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="ImageLoadingEstimator"/>.
    /// </summary>
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
        /// The flag for DataViewType for the image. If Type true, it is a VectorDataView of bytes else it is an ImageDataView type.
        /// If no options are specified, it defaults to false for ImageDataView type.
        /// </summary>
        private readonly bool _useImageType;

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
            : this(env, imageFolder, type: true, columns)
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="ImageLoadingTransformer"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="type">Image type flag - true for ImageDataViewType or false for VectorDataViewType. Defaults to true i.e. ImageDataViewType if not specified.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingTransformer(IHostEnvironment env, string imageFolder = null, bool type = true, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingTransformer)), columns)
        {
            // Throws ArgumentException if given imageFolder path is invalid. Note: imageFolder may be null or empty in this case.
            if (String.IsNullOrEmpty(imageFolder))
                ImageFolder = null;
            else
            {
                if (Directory.Exists(imageFolder))
                    ImageFolder = Path.GetFullPath(imageFolder);
                else
                    throw new ArgumentException(String.Format("Directory \"{0}\" does not exist.", imageFolder));
            }
            _useImageType = type;
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
            if (ctx.Header.ModelVerWritten >= 0x00010003) // do a version check
                _useImageType = ctx.Reader.ReadBoolean();
            else
                _useImageType = true; // It is an ImageDataViewType
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
            ctx.Writer.Write(_useImageType);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGLOADR",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010003, // Added support for output type as byte array
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImageLoadingTransformer).Assembly.FullName);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema, _useImageType);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ImageLoadingTransformer _parent;
            private readonly bool _type;

            public Mapper(ImageLoadingTransformer parent, DataViewSchema inputSchema, bool type)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _type = type;
                _parent = parent;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                // Check for the type of Image, if true load images as ImageDataViewType else load images as VBuffer<bytes>
                if (_type)
                    return MakeGetterImageDataViewType(input, iinfo, activeOutput, out disposer);
                else
                    return MakeGetterVectorDataViewByteType(input, iinfo, activeOutput, out disposer);
            }

            private Delegate MakeGetterImageDataViewType(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
                var lastImage = default(Bitmap);

                disposer = () =>
                {
                    if (lastImage != null)
                    {
                        lastImage.Dispose();
                        lastImage = null;
                    }
                };

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
                            string path = src.ToString();
                            if (!string.IsNullOrWhiteSpace(_parent.ImageFolder))
                                path = Path.Combine(_parent.ImageFolder, path);

                            // to avoid locking file, use the construct below to load bitmap
                            var bytes = File.ReadAllBytes(path);
                            var ms = new MemoryStream(bytes);
                            dst = (Bitmap)Image.FromStream(ms);
                            dst.Tag = path;

                            // Check for an incorrect pixel format which indicates the loading failed
                            if (dst.PixelFormat == System.Drawing.Imaging.PixelFormat.DontCare)
                                throw Host.Except($"Failed to load image {src.ToString()}.");
                        }

                        lastImage = dst;
                    };

                return del;
            }

            private Delegate MakeGetterVectorDataViewByteType(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                disposer = null;
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                ReadOnlyMemory<char> src = default;
                ValueGetter<VBuffer<byte>> del =
                    (ref VBuffer<byte> dst) =>
                    {
                        getSrc(ref src);
                        if (src.Length > 0)
                        {
                            string path = src.ToString();
                            if (!string.IsNullOrWhiteSpace(_parent.ImageFolder))
                                path = Path.Combine(_parent.ImageFolder, path);
                            if (!TryLoadDataIntoBuffer(path, ref dst))
                            {
                                var editor = VBufferEditor.Create(ref dst, 0); //Empty Image
                                dst = editor.Commit();
                            }
                        }
                        else
                        {
                            var editor = VBufferEditor.Create(ref dst, 0);
                            dst = editor.Commit();
                        }

                    };

                return del;
            }

            private static bool TryLoadDataIntoBuffer(string path, ref VBuffer<byte> imgData)
            {
                int count = -1;
                int bytesread = -1;
                // bufferSize == 1 used to avoid unnecessary buffer in FileStream
                using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1))
                {
                    long fileLength = fs.Length;
                    if (fileLength > int.MaxValue)
                        throw new IOException($"File {path} too big to open.");
                    else if (fileLength == 0)
                    {
                        byte[] imageBuffer;
                        // Some file systems (e.g. procfs on Linux) return 0 for length even when there's content.
                        // Thus we need to assume 0 doesn't mean empty.
                        imageBuffer = File.ReadAllBytes(path);
                        count = imageBuffer.Length;
                        imgData = new VBuffer<byte>(count, imageBuffer);
                        return (count > 0);
                    }

                    count = (int)fileLength;
                    var editor = VBufferEditor.Create(ref imgData, count);
                    bytesread = ReadToEnd(fs, editor.Values);
                    imgData = editor.Commit();
                    return (count > 0);
                }

            }

            private static int ReadToEnd(System.IO.Stream stream, Span<byte> bufferSpan)
            {
                int totalBytesRead = 0;
                int bytesRead;

                int chunksize = 4096; // Most optimal size for buffer, friendly to CPU's L1 cache
                var bufferPool = ArrayPool<byte>.Shared;
                byte[] readBuffer = bufferPool.Rent(chunksize);

                try
                {
                    var readBufferSpan = readBuffer.AsSpan();
                    var srcSpan = readBufferSpan;
                    while ((bytesRead = stream.Read(readBuffer, 0, readBuffer.Length)) > 0)
                    {
                        if (bytesRead != srcSpan.Length)
                            srcSpan = readBufferSpan.Slice(0, bytesRead);
                        var dstSpan = bufferSpan.Slice(totalBytesRead, bytesRead);
                        Contract.Assert(srcSpan.Length == dstSpan.Length);
                        srcSpan.CopyTo(dstSpan);
                        totalBytesRead += bytesRead;
                    }
                }
                finally
                {

                    // don't use the reference to the buffer after returning it!
                    bufferPool.Return(readBuffer);
                }

                return totalBytesRead;
            }

            private DataViewType GetDataViewType()
            {
                if (_type)
                    return new ImageDataViewType();
                else
                    return new VectorDataViewType(NumberDataViewType.Byte);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select(x => new DataViewSchema.DetachedColumn(x.outputColumnName, GetDataViewType(), null)).ToArray();
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> for the <see cref="ImageLoadingTransformer"/>.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | [Text](<xref:Microsoft.ML.Data.TextDataViewType>) |
    /// | Output column data type | <xref:System.Drawing.Bitmap> |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.ImageAnalytics |
    /// | Exportable to ONNX | No |
    ///
    /// The resulting <xref:Microsoft.ML.Data.ImageLoadingTransformer> creates a new column, named as specified in the output column name parameters, and
    /// loads in it images specified in the input column.
    /// Loading is the first step of almost every pipeline that does image processing, and further analysis on images.
    /// The images to load need to be in the formats supported by <xref:System.Drawing.Bitmap>.
    /// For end-to-end image processing pipelines, and scenarios in your applications, see the
    /// [examples](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started) in the machinelearning-samples github repository.</a>
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="ImageEstimatorsCatalog.LoadImages(TransformsCatalog, string, string, string)" />

    public sealed class ImageLoadingEstimator : TrivialEstimator<ImageLoadingTransformer>
    {
        private readonly bool _type;
        /// <summary>
        /// Load images in memory.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingEstimator(IHostEnvironment env, string imageFolder, params (string outputColumnName, string inputColumnName)[] columns)
            : this(env, new ImageLoadingTransformer(env, imageFolder, true, columns), true)
        {
        }

        /// <summary>
        /// Load images in memory.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="type">Image type flag - true for  ImageDataViewType or false for VectorDataView. Defaults to true i.e ImageDataViewType if not specified.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingEstimator(IHostEnvironment env, string imageFolder, bool type = true, params (string outputColumnName, string inputColumnName)[] columns)
            : this(env, new ImageLoadingTransformer(env, imageFolder, type, columns), type)
        {
        }

        internal ImageLoadingEstimator(IHostEnvironment env, ImageLoadingTransformer transformer, bool type = true)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingEstimator)), transformer)
        {
            _type = type;
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

                if (_type)
                    result[outputColumnName] = new SchemaShape.Column(outputColumnName, SchemaShape.Column.VectorKind.Scalar, new ImageDataViewType(), false);
                else
                    result[outputColumnName] = new SchemaShape.Column(outputColumnName, SchemaShape.Column.VectorKind.VariableVector, NumberDataViewType.Byte, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
