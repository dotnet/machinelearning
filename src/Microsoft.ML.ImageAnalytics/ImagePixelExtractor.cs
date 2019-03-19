// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Image;

[assembly: LoadableClass(ImagePixelExtractingTransformer.Summary, typeof(IDataTransform), typeof(ImagePixelExtractingTransformer), typeof(ImagePixelExtractingTransformer.Options), typeof(SignatureDataTransform),
    ImagePixelExtractingTransformer.UserName, "ImagePixelExtractorTransform", "ImagePixelExtractor")]

[assembly: LoadableClass(ImagePixelExtractingTransformer.Summary, typeof(IDataTransform), typeof(ImagePixelExtractingTransformer), null, typeof(SignatureLoadDataTransform),
    ImagePixelExtractingTransformer.UserName, ImagePixelExtractingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImagePixelExtractingTransformer), null, typeof(SignatureLoadModel),
    ImagePixelExtractingTransformer.UserName, ImagePixelExtractingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImagePixelExtractingTransformer), null, typeof(SignatureLoadRowMapper),
    ImagePixelExtractingTransformer.UserName, ImagePixelExtractingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Image
{
    /// <summary>
    /// <see cref="ITransformer"/> produced by fitting the <see cref="IDataView"/> to an <see cref="ImagePixelExtractingEstimator" /> .
    /// </summary>
    /// <remarks>
    ///  During the transformation, the columns of <see cref="ImageType"/> are converted them into a vector representing the image pixels
    ///  than can be further used as features by the algorithms added to the pipeline.
    /// <seealso cref="ImageEstimatorsCatalog.ExtractPixels(TransformsCatalog, ImagePixelExtractingEstimator.ColumnOptions[])" />
    /// <seealso cref="ImageEstimatorsCatalog.ExtractPixels(TransformsCatalog, string, string, ImagePixelExtractingEstimator.ColorBits, ImagePixelExtractingEstimator.ColorsOrder, bool, float, float, bool)" />
    /// <seealso cref="ImageEstimatorsCatalog"/>
    /// </remarks>
    public sealed class ImagePixelExtractingTransformer : OneToOneTransformerBase
    {
        [BestFriend]
        internal class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool? UseAlpha;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool? UseRed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool? UseGreen;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool? UseBlue;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Order of channels")]
            public ImagePixelExtractingEstimator.ColorsOrder? Order;

            // REVIEW: Consider turning this into an enum that allows for pixel, line, or planar interleaving.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in specified order")]
            public bool? Interleave;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to convert to floating point", ShortName = "conv")]
            public bool? Convert;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Offset (pre-scale)")]
            public Single? Offset;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale factor")]
            public Single? Scale;

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
                if (UseAlpha != null || UseRed != null || UseGreen != null || UseBlue != null || Convert != null ||
                    Offset != null || Scale != null || Interleave != null || Order != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        [BestFriend]
        internal class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool UseAlpha = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool UseRed = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool UseGreen = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool UseBlue = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Order of colors.")]
            public ImagePixelExtractingEstimator.ColorsOrder Order = ImagePixelExtractingEstimator.Defaults.Order;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in specified order")]
            public bool Interleave = ImagePixelExtractingEstimator.Defaults.Interleave;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to convert to floating point", ShortName = "conv")]
            public bool Convert = ImagePixelExtractingEstimator.Defaults.Convert;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Offset (pre-scale)")]
            public Single? Offset;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale factor")]
            public Single? Scale;
        }

        internal const string Summary = "Extract color plane(s) from an image. Options include scaling, offset and conversion to floating point.";
        internal const string UserName = "Image Pixel Extractor Transform";
        internal const string LoaderSignature = "ImagePixelExtractor";

        internal const uint BeforeOrderVersion = 0x00010002;
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGPXEXT",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verWrittenCur: 0x00010003, // Add pixel order
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImagePixelExtractingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "ImagePixelExtractor";

        private readonly ImagePixelExtractingEstimator.ColumnOptions[] _columns;

        /// <summary>
        /// The columns passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<ImagePixelExtractingEstimator.ColumnOptions> Columns => _columns.AsReadOnly();

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="colorsToExtract">What colors to extract.</param>
        /// <param name="orderOfExtraction">In which order to extract colors from pixel.</param>
        /// <param name="interleavePixelColors">Whether to interleave the pixels colors, meaning keep them in the <paramref name="orderOfExtraction"/> order, or leave them in the plannar form:
        /// all the values for one color for all pixels, then all the values for another color and so on.</param>
        /// <param name="offsetImage">Offset each pixel's color value by this amount. Applied to color value first.</param>
        /// <param name="scaleImage">Scale each pixel's color value by this amount. Applied to color value second.</param>
        /// <param name="outputAsFloatArray">Output array as float array. If false, output as byte array and ignores <paramref name="offsetImage"/> and <paramref name="scaleImage"/>.</param>
        internal ImagePixelExtractingTransformer(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            ImagePixelExtractingEstimator.ColorBits colorsToExtract = ImagePixelExtractingEstimator.Defaults.Colors,
            ImagePixelExtractingEstimator.ColorsOrder orderOfExtraction = ImagePixelExtractingEstimator.Defaults.Order,
            bool interleavePixelColors = ImagePixelExtractingEstimator.Defaults.Interleave,
            float offsetImage = ImagePixelExtractingEstimator.Defaults.Offset,
            float scaleImage = ImagePixelExtractingEstimator.Defaults.Scale,
            bool outputAsFloatArray = ImagePixelExtractingEstimator.Defaults.Convert)
            : this(env, new ImagePixelExtractingEstimator.ColumnOptions(outputColumnName, inputColumnName, colorsToExtract, orderOfExtraction, interleavePixelColors, offsetImage, scaleImage, outputAsFloatArray))
        {
        }

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columns">Describes the parameters of pixel extraction for each column pair.</param>
        internal ImagePixelExtractingTransformer(IHostEnvironment env, params ImagePixelExtractingEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(ImagePixelExtractingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            env.CheckValue(options.Columns, nameof(options.Columns));

            var columns = new ImagePixelExtractingEstimator.ColumnOptions[options.Columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var item = options.Columns[i];
                columns[i] = new ImagePixelExtractingEstimator.ColumnOptions(item, options);
            }

            var transformer = new ImagePixelExtractingTransformer(env, columns);
            return new RowToRowMapperTransform(env, input, transformer.MakeRowMapper(input.Schema), transformer.MakeRowMapper);
        }

        // Factory method for SignatureLoadModel.
        private static ImagePixelExtractingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new ImagePixelExtractingTransformer(host, ctx);
        }

        private ImagePixelExtractingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>

            // for each added column
            //   ColumnOptions

            _columns = new ImagePixelExtractingEstimator.ColumnOptions[ColumnPairs.Length];
            for (int i = 0; i < _columns.Length; i++)
                _columns[i] = new ImagePixelExtractingEstimator.ColumnOptions(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>

            // for each added column
            //   ColumnOptions

            base.SaveColumns(ctx);

            foreach (var info in _columns)
                info.Save(ctx);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var inputColName = _columns[col].InputColumnName;
            var imageType = inputSchema[srcCol].Type as ImageType;
            if (imageType == null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColName, "image", inputSchema[srcCol].Type.ToString());
            if (imageType.Height <= 0 || imageType.Width <= 0)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColName, "known-size image", "unknown-size image");
            if ((long)imageType.Height * imageType.Width > int.MaxValue / 4)
                throw Host.Except("Image dimensions are too large");
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ImagePixelExtractingTransformer _parent;
            private readonly VectorType[] _types;

            public Mapper(ImagePixelExtractingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = ConstructTypes();
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent._columns.Select((x, idx) => new DataViewSchema.DetachedColumn(x.Name, _types[idx], null)).ToArray();

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent._columns.Length);

                if (_parent._columns[iinfo].OutputAsFloatArray)
                    return GetGetterCore<Single>(input, iinfo, out disposer);
                return GetGetterCore<byte>(input, iinfo, out disposer);
            }

            //REVIEW Rewrite it to where TValue : IConvertible
            private ValueGetter<VBuffer<TValue>> GetGetterCore<TValue>(DataViewRow input, int iinfo, out Action disposer)
                where TValue : struct
            {
                var type = _types[iinfo];
                var dims = type.Dimensions;
                Contracts.Assert(dims.Length == 3);

                var ex = _parent._columns[iinfo];

                int planes = ex.InterleavePixelColors ? dims[2] : dims[0];
                int height = ex.InterleavePixelColors ? dims[0] : dims[1];
                int width = ex.InterleavePixelColors ? dims[1] : dims[2];

                int size = type.Size;
                Contracts.Assert(size > 0);
                Contracts.Assert(size == planes * height * width);
                int cpix = height * width;

                var getSrc = input.GetGetter<Bitmap>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(Bitmap);

                disposer =
                    () =>
                    {
                        if (src != null)
                        {
                            src.Dispose();
                            src = null;
                        }
                    };

                return
                    (ref VBuffer<TValue> dst) =>
                    {
                        getSrc(ref src);
                        Contracts.AssertValueOrNull(src);

                        if (src == null)
                        {
                            VBufferUtils.Resize(ref dst, size, 0);
                            return;
                        }

                        Host.Check(src.PixelFormat == System.Drawing.Imaging.PixelFormat.Format32bppArgb
                            || src.PixelFormat == System.Drawing.Imaging.PixelFormat.Format24bppRgb,
                            "Transform only supports pixel formats Format24bppRgb and Format32bppArgb");
                        Host.Check(src.Height == height && src.Width == width);

                        var editor = VBufferEditor.Create(ref dst, size);
                        var values = editor.Values;

                        float offset = ex.OffsetImage;
                        float scale = ex.ScaleImage;
                        Contracts.Assert(scale != 0);

                        // REVIEW: split the getter into 2 specialized getters, one for float case and one for byte case.
                        Span<float> vf = typeof(TValue) == typeof(float) ? MemoryMarshal.Cast<TValue, float>(editor.Values) : default;
                        Span<byte> vb = typeof(TValue) == typeof(byte) ? MemoryMarshal.Cast<TValue, byte>(editor.Values) : default;
                        Contracts.Assert(!vf.IsEmpty || !vb.IsEmpty);
                        bool needScale = offset != 0 || scale != 1;
                        Contracts.Assert(!needScale || !vf.IsEmpty);

                        ImagePixelExtractingEstimator.GetOrder(ex.OrderOfExtraction, ex.ColorsToExtract, out int a, out int r, out int b, out int g);

                        int h = height;
                        int w = width;

                        if (ex.InterleavePixelColors)
                        {
                            int idst = 0;
                            for (int y = 0; y < h; ++y)
                                for (int x = 0; x < w; x++)
                                {
                                    var pb = src.GetPixel(x, y);
                                    if (!vb.IsEmpty)
                                    {
                                        if (a != -1) { vb[idst + a] = pb.A; }
                                        if (r != -1) { vb[idst + r] = pb.R; }
                                        if (g != -1) { vb[idst + g] = pb.G; }
                                        if (b != -1) { vb[idst + b] = pb.B; }
                                    }
                                    else if (!needScale)
                                    {
                                        if (a != -1) { vf[idst + a] = pb.A; }
                                        if (r != -1) { vf[idst + r] = pb.R; }
                                        if (g != -1) { vf[idst + g] = pb.G; }
                                        if (b != -1) { vf[idst + b] = pb.B; }
                                    }
                                    else
                                    {

                                        if (a != -1) { vf[idst + a] = (pb.A - offset) * scale; }
                                        if (r != -1) { vf[idst + r] = (pb.R - offset) * scale; }
                                        if (g != -1) { vf[idst + g] = (pb.G - offset) * scale; }
                                        if (b != -1) { vf[idst + b] = (pb.B - offset) * scale; }
                                    }
                                    idst += ex.Planes;
                                }
                            Contracts.Assert(idst == size);
                        }
                        else
                        {
                            int idstMin = 0;
                            for (int y = 0; y < h; ++y)
                            {
                                int idst = idstMin + y * w;
                                for (int x = 0; x < w; x++, idst++)
                                {
                                    if (!vb.IsEmpty)
                                    {
                                        var pb = src.GetPixel(x, y);
                                        if (a != -1) vb[idst + cpix * a] = pb.A;
                                        if (r != -1) vb[idst + cpix * r] = pb.R;
                                        if (g != -1) vb[idst + cpix * g] = pb.G;
                                        if (b != -1) vb[idst + cpix * b] = pb.B;
                                    }
                                    else if (!needScale)
                                    {
                                        var pb = src.GetPixel(x, y);
                                        if (a != -1) vf[idst + cpix * a] = pb.A;
                                        if (r != -1) vf[idst + cpix * r] = pb.R;
                                        if (g != -1) vf[idst + cpix * g] = pb.G;
                                        if (b != -1) vf[idst + cpix * b] = pb.B;
                                    }
                                    else
                                    {
                                        var pb = src.GetPixel(x, y);
                                        if (a != -1) vf[idst + cpix * a] = (pb.A - offset) * scale;
                                        if (r != -1) vf[idst + cpix * r] = (pb.R - offset) * scale;
                                        if (g != -1) vf[idst + cpix * g] = (pb.G - offset) * scale;
                                        if (b != -1) vf[idst + cpix * b] = (pb.B - offset) * scale;
                                    }
                                }
                            }
                        }

                        dst = editor.Commit();
                    };
            }

            private VectorType[] ConstructTypes()
            {
                var types = new VectorType[_parent._columns.Length];
                for (int i = 0; i < _parent._columns.Length; i++)
                {
                    var column = _parent._columns[i];
                    Contracts.Assert(column.Planes > 0);

                    var type = InputSchema[ColMapNewToOld[i]].Type as ImageType;
                    Contracts.Assert(type != null);

                    int height = type.Height;
                    int width = type.Width;
                    Contracts.Assert(height > 0);
                    Contracts.Assert(width > 0);
                    Contracts.Assert((long)height * width <= int.MaxValue / 4);

                    if (column.InterleavePixelColors)
                        types[i] = new VectorType(column.OutputAsFloatArray ? NumberDataViewType.Single : NumberDataViewType.Byte, height, width, column.Planes);
                    else
                        types[i] = new VectorType(column.OutputAsFloatArray ? NumberDataViewType.Single : NumberDataViewType.Byte, column.Planes, height, width);
                }
                return types;
            }
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> that extracts the pixels of the image into a vector.
    /// This vector can further be used as a feature to the algorithms.
    /// </summary>
    /// <remarks>
    /// Calling <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> in this estimator, produces an <see cref="ImagePixelExtractingTransformer"/>.
    /// <seealso cref="ImageEstimatorsCatalog.ExtractPixels(TransformsCatalog, ImagePixelExtractingEstimator.ColumnOptions[])" />
    /// <seealso cref="ImageEstimatorsCatalog.ExtractPixels(TransformsCatalog, string, string, ColorBits, ColorsOrder, bool, float, float, bool)" />
    /// <seealso cref="ImageEstimatorsCatalog"/>
    /// </remarks>
    public sealed class ImagePixelExtractingEstimator : TrivialEstimator<ImagePixelExtractingTransformer>
    {
        [BestFriend]
        internal static class Defaults
        {
            public const ColorsOrder Order = ColorsOrder.ARGB;
            public const ColorBits Colors = ColorBits.Rgb;
            public const bool Interleave = false;
            public const bool Convert = true;
            public const float Scale = 1f;
            public const float Offset = 0f;
        }
        /// <summary>
        /// Which color channels are extracted. Note that these values are serialized so should not be modified.
        /// </summary>
        [Flags]
        public enum ColorBits : byte
        {
            Alpha = 0x01,
            Red = 0x02,
            Green = 0x04,
            Blue = 0x08,

            Rgb = Red | Green | Blue,
            All = Alpha | Red | Green | Blue
        }

        public enum ColorsOrder : byte
        {
#pragma warning disable MSML_GeneralName // This name should be PascalCased
            ARGB = 1,
            ARBG = 2,
            ABRG = 3,
            ABGR = 4,
            AGRB = 5,
            AGBR = 6
#pragma warning restore MSML_GeneralName // This name should be PascalCased
        }

        internal static void GetOrder(ColorsOrder order, ColorBits colors, out int a, out int r, out int b, out int g)
        {
            var str = order.ToString().ToLowerInvariant();
            a = -1;
            r = -1;
            b = -1;
            g = -1;
            int pos = 0;
            for (int i = 0; i < str.Length; i++)
            {

                switch (str[i])
                {
                    case 'a':
                        if ((colors & ColorBits.Alpha) != 0)
                            a = pos++;
                        break;
                    case 'r':
                        if ((colors & ColorBits.Red) != 0)
                            r = pos++;
                        break;
                    case 'b':
                        if ((colors & ColorBits.Blue) != 0)
                            b = pos++;
                        break;
                    case 'g':
                        if ((colors & ColorBits.Green) != 0)
                            g = pos++;
                        break;
                }
            }
        }

        /// <summary>
        /// Describes how the transformer handles one image pixel extraction column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>Name of the column resulting from the transformation of <see cref="InputColumnName"/>.</summary>
            public readonly string Name;

            /// <summary>Name of column to transform.</summary>
            public readonly string InputColumnName;

            /// <summary>The colors to extract.</summary>
            public readonly ColorBits ColorsToExtract;

            /// <summary>The order in which to extract color values from pixel.</summary>
            public readonly ColorsOrder OrderOfExtraction;

            /// <summary>Offset pixel's color value by this amount. Applied to color value first.</summary>
            public readonly float OffsetImage;

            /// <summary>Scale pixel's color value by this amount. Applied to color value second.</summary>
            public readonly float ScaleImage;

            /// <summary>
            /// Whether to interleave the pixels colors, meaning keep them in the <see cref="OrderOfExtraction"/> order, or leave them in the plannar form:
            /// all the values for one color for all pixels, then all the values for another color and so on.
            /// </summary>
            public readonly bool InterleavePixelColors;

            /// <summary>Output array as float array. If false, output as byte array and ignores <see cref="OffsetImage"/> and <see cref="ScaleImage"/> .</summary>
            public readonly bool OutputAsFloatArray;

            internal readonly byte Planes;

            internal ColumnOptions(ImagePixelExtractingTransformer.Column item, ImagePixelExtractingTransformer.Options options)
            {
                Contracts.CheckValue(item, nameof(item));
                Contracts.CheckValue(options, nameof(options));

                Name = item.Name;
                InputColumnName = item.Source ?? item.Name;
                if (item.UseAlpha ?? options.UseAlpha) { ColorsToExtract |= ColorBits.Alpha; Planes++; }
                if (item.UseRed ?? options.UseRed) { ColorsToExtract |= ColorBits.Red; Planes++; }
                if (item.UseGreen ?? options.UseGreen) { ColorsToExtract |= ColorBits.Green; Planes++; }
                if (item.UseBlue ?? options.UseBlue) { ColorsToExtract |= ColorBits.Blue; Planes++; }
                Contracts.CheckUserArg(Planes > 0, nameof(item.UseRed), "Need to use at least one color plane");

                OrderOfExtraction = item.Order ?? options.Order;
                InterleavePixelColors = item.Interleave ?? options.Interleave;

                OutputAsFloatArray = item.Convert ?? options.Convert;
                if (!OutputAsFloatArray)
                {
                    OffsetImage = Defaults.Offset;
                    ScaleImage = Defaults.Scale;
                }
                else
                {
                    OffsetImage = item.Offset ?? options.Offset ?? Defaults.Offset;
                    ScaleImage = item.Scale ?? options.Scale ?? Defaults.Scale;
                    Contracts.CheckUserArg(FloatUtils.IsFinite(OffsetImage), nameof(item.Offset));
                    Contracts.CheckUserArg(FloatUtils.IsFiniteNonZero(ScaleImage), nameof(item.Scale));
                }
            }

            /// <summary>
            /// Describes how the transformer handles one input-output column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="colorsToExtract">What colors to extract.</param>
            /// <param name="orderOfExtraction">In which order to extract colors from pixel.</param>
            /// <param name="interleavePixelColors">Whether to interleave the pixels, meaning keep them in the <paramref name="orderOfExtraction"/> order, or leave them in the plannar form:
            /// all the values for one color for all pixels, then all the values for another color and so on.</param>
            /// <param name="offsetImage">Offset each pixel's color value by this amount. Applied to color value before <paramref name="scaleImage"/>.</param>
            /// <param name="scaleImage">Scale each pixel's color value by this amount. Applied to color value after <paramref name="offsetImage"/>.</param>
            /// <param name="outputAsFloatArray">Output array as float array. If false, output as byte array and ignores <paramref name="offsetImage"/> and <paramref name="scaleImage"/>.</param>
            public ColumnOptions(string name,
                string inputColumnName = null,
                ColorBits colorsToExtract = Defaults.Colors,
                ColorsOrder orderOfExtraction = Defaults.Order,
                bool interleavePixelColors = Defaults.Interleave,
                float offsetImage = Defaults.Offset,
                float scaleImage = Defaults.Scale,
                bool outputAsFloatArray = Defaults.Convert)
            {
                Contracts.CheckNonWhiteSpace(name, nameof(name));

                Name = name;
                InputColumnName = inputColumnName ?? name;
                ColorsToExtract = colorsToExtract;
                OrderOfExtraction = orderOfExtraction;
                if ((ColorsToExtract & ColorBits.Alpha) == ColorBits.Alpha) Planes++;
                if ((ColorsToExtract & ColorBits.Red) == ColorBits.Red) Planes++;
                if ((ColorsToExtract & ColorBits.Green) == ColorBits.Green) Planes++;
                if ((ColorsToExtract & ColorBits.Blue) == ColorBits.Blue) Planes++;
                Contracts.CheckParam(Planes > 0, nameof(colorsToExtract), "Need to use at least one color plane.");

                InterleavePixelColors = interleavePixelColors;

                OutputAsFloatArray = outputAsFloatArray;
                if (!OutputAsFloatArray)
                {
                    OffsetImage = Defaults.Offset;
                    ScaleImage = Defaults.Scale;
                }
                else
                {
                    OffsetImage = offsetImage;
                    ScaleImage = scaleImage;
                }
                Contracts.CheckParam(FloatUtils.IsFinite(OffsetImage), nameof(offsetImage));
                Contracts.CheckParam(FloatUtils.IsFiniteNonZero(ScaleImage), nameof(scaleImage));
            }

            internal ColumnOptions(string name, string inputColumnName, ModelLoadContext ctx)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.AssertNonEmpty(inputColumnName);
                Contracts.AssertValue(ctx);

                Name = name;
                InputColumnName = inputColumnName;

                // *** Binary format ***
                // byte: colors
                // byte: order
                // byte: convert
                // Float: offset
                // Float: scale
                // byte: separateChannels
                ColorsToExtract = (ImagePixelExtractingEstimator.ColorBits)ctx.Reader.ReadByte();
                Contracts.CheckDecode(ColorsToExtract != 0);
                Contracts.CheckDecode((ColorsToExtract & ImagePixelExtractingEstimator.ColorBits.All) == ColorsToExtract);
                if (ctx.Header.ModelVerWritten <= ImagePixelExtractingTransformer.BeforeOrderVersion)
                    OrderOfExtraction = ColorsOrder.ARGB;
                else
                {
                    OrderOfExtraction = (ImagePixelExtractingEstimator.ColorsOrder)ctx.Reader.ReadByte();
                    Contracts.CheckDecode(OrderOfExtraction != 0);
                }

                // Count the planes.
                int planes = (int)ColorsToExtract;
                planes = (planes & 0x05) + ((planes >> 1) & 0x05);
                planes = (planes & 0x03) + ((planes >> 2) & 0x03);
                Planes = (byte)planes;
                Contracts.Assert(0 < Planes & Planes <= 4);

                OutputAsFloatArray = ctx.Reader.ReadBoolByte();
                OffsetImage = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFinite(OffsetImage));
                ScaleImage = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFiniteNonZero(ScaleImage));
                Contracts.CheckDecode(OutputAsFloatArray || OffsetImage == 0 && ScaleImage == 1);
                InterleavePixelColors = ctx.Reader.ReadBoolByte();
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);
#if DEBUG
                // This code is used in deserialization - assert that it matches what we computed above.
                int planes = (int)ColorsToExtract;
                planes = (planes & 0x05) + ((planes >> 1) & 0x05);
                planes = (planes & 0x03) + ((planes >> 2) & 0x03);
                Contracts.Assert(planes == Planes);
#endif

                // *** Binary format ***
                // byte: colors
                // byte: order
                // byte: convert
                // Float: offset
                // Float: scale
                // byte: separateChannels
                Contracts.Assert(ColorsToExtract != 0);
                Contracts.Assert((ColorsToExtract & ImagePixelExtractingEstimator.ColorBits.All) == ColorsToExtract);
                ctx.Writer.Write((byte)ColorsToExtract);
                ctx.Writer.Write((byte)OrderOfExtraction);
                ctx.Writer.WriteBoolByte(OutputAsFloatArray);
                Contracts.Assert(FloatUtils.IsFinite(OffsetImage));
                ctx.Writer.Write(OffsetImage);
                Contracts.Assert(FloatUtils.IsFiniteNonZero(ScaleImage));
                Contracts.Assert(OutputAsFloatArray || OffsetImage == 0 && ScaleImage == 1);
                ctx.Writer.Write(ScaleImage);
                ctx.Writer.WriteBoolByte(InterleavePixelColors);
            }
        }

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>. Null means <paramref name="inputColumnName"/> is replaced.</param>
        /// <param name="inputColumnName">Name of the input column.</param>
        /// <param name="colorsToExtract">What colors to extract.</param>
        /// <param name="orderOfExtraction">In which order to extract colors from pixel.</param>
        /// <param name="interleavePixelColors">Whether to interleave the pixels, meaning keep them in the <paramref name="orderOfExtraction"/> order, or leave them in the plannar form:
        /// all the values for one color for all pixels, then all the values for another color and so on.</param>
        /// <param name="offsetImage">Offset each pixel's color value by this amount. Applied to color value before <paramref name="scaleImage"/>.</param>
        /// <param name="scaleImage">Scale each pixel's color value by this amount. Applied to color value after <paramref name="offsetImage"/>.</param>
        /// <param name="outputAsFloatArray">Output array as float array. If false, output as byte array.</param>
        [BestFriend]
        internal ImagePixelExtractingEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            ColorBits colorsToExtract = Defaults.Colors,
            ColorsOrder orderOfExtraction = Defaults.Order,
            bool interleavePixelColors = Defaults.Interleave,
            float offsetImage = Defaults.Offset,
            float scaleImage = Defaults.Scale,
            bool outputAsFloatArray = Defaults.Convert)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImagePixelExtractingEstimator)),
                  new ImagePixelExtractingTransformer(env, outputColumnName, inputColumnName, colorsToExtract, orderOfExtraction, interleavePixelColors, offsetImage, scaleImage, outputAsFloatArray))
        {
        }

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columns">Describes the parameters of pixel extraction for each column pair.</param>
        internal ImagePixelExtractingEstimator(IHostEnvironment env, params ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImagePixelExtractingEstimator)), new ImagePixelExtractingTransformer(env, columns))
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.InputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName);
                if (!(col.ItemType is ImageType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, new ImageType().ToString(), col.GetTypeString());

                var itemType = colInfo.OutputAsFloatArray ? NumberDataViewType.Single : NumberDataViewType.Byte;
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, itemType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
