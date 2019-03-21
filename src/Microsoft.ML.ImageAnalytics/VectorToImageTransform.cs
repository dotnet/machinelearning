// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.Image;

[assembly: LoadableClass(VectorToImageConvertingTransformer.Summary, typeof(IDataTransform), typeof(VectorToImageConvertingTransformer), typeof(VectorToImageConvertingTransformer.Options), typeof(SignatureDataTransform),
    ImagePixelExtractingTransformer.UserName, "VectorToImageTransform", "VectorToImage")]

[assembly: LoadableClass(VectorToImageConvertingTransformer.Summary, typeof(IDataTransform), typeof(VectorToImageConvertingTransformer), null, typeof(SignatureLoadDataTransform),
    VectorToImageConvertingTransformer.UserName, VectorToImageConvertingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(VectorToImageConvertingTransformer), null, typeof(SignatureLoadModel),
    VectorToImageConvertingTransformer.UserName, VectorToImageConvertingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(VectorToImageConvertingTransformer), null, typeof(SignatureLoadRowMapper),
    VectorToImageConvertingTransformer.UserName, VectorToImageConvertingTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms.Image
{
    /// <summary>
    /// <see cref="ITransformer"/> produced by fitting the <see cref="IDataView"/> to an <see cref="VectorToImageConvertingEstimator" /> .
    /// </summary>
    /// <remarks>
    /// <seealso cref="ImageEstimatorsCatalog.ConvertToImage(TransformsCatalog, VectorToImageConvertingEstimator.ColumnOptions[])" />
    /// <seealso cref="ImageEstimatorsCatalog.ConvertToImage(TransformsCatalog, int, int, string, string, ImagePixelExtractingEstimator.ColorBits, ImagePixelExtractingEstimator.ColorsOrder, bool, float, float, int, int, int, int)"/>
    /// <seealso cref="ImageEstimatorsCatalog"/>
    /// </remarks>
    public sealed class VectorToImageConvertingTransformer : OneToOneTransformerBase
    {
        internal class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool? ContainsAlpha;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool? ContainsRed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool? ContainsGreen;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool? ContainsBlue;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Order of channels")]
            public ImagePixelExtractingEstimator.ColorsOrder? Order;

            // REVIEW: Consider turning this into an enum that allows for pixel, line, or planar interleaving.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in specified order")]
            public bool? Interleave;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Width of the image", ShortName = "width")]
            public int? ImageWidth;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Height of the image", ShortName = "height")]
            public int? ImageHeight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Offset (pre-scale)")]
            public Single? Offset;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale factor")]
            public Single? Scale;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Default value for alpha channel. Will be used if ContainsAlpha set to false")]
            public int? DefaultAlpha;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Default value for red channel. Will be used if ContainsRed set to false")]
            public int? DefaultRed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Default value for green channel. Will be used if ContainsGreen set to false")]
            public int? DefaultGreen;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Default value for blue channel. Will be used if ContainsGreen set to false")]
            public int? DefaultBlue;

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
                if (ContainsAlpha != null || ContainsRed != null || ContainsGreen != null || ContainsBlue != null || ImageWidth != null ||
                    ImageHeight != null || Offset != null || Scale != null || Interleave != null || Order != null || DefaultAlpha != null ||
                    DefaultBlue != null || DefaultGreen != null || DefaultRed != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        internal class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool ContainsAlpha = (ImagePixelExtractingEstimator.Defaults.Colors & ImagePixelExtractingEstimator.ColorBits.Alpha) > 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool ContainsRed = (ImagePixelExtractingEstimator.Defaults.Colors & ImagePixelExtractingEstimator.ColorBits.Red) > 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool ContainsGreen = (ImagePixelExtractingEstimator.Defaults.Colors & ImagePixelExtractingEstimator.ColorBits.Green) > 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool ContainsBlue = (ImagePixelExtractingEstimator.Defaults.Colors & ImagePixelExtractingEstimator.ColorBits.Blue) > 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Order of colors.")]
            public ImagePixelExtractingEstimator.ColorsOrder Order = ImagePixelExtractingEstimator.Defaults.Order;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in specified order")]
            public bool Interleave = ImagePixelExtractingEstimator.Defaults.Interleave;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Width of the image", ShortName = "width")]
            public int ImageWidth;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Height of the image", ShortName = "height")]
            public int ImageHeight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Offset (pre-scale)")]
            public Single Offset = VectorToImageConvertingEstimator.Defaults.Offset;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale factor")]
            public Single Scale = VectorToImageConvertingEstimator.Defaults.Scale;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Default value for alpha channel. Will be used if ContainsAlpha set to false")]
            public int DefaultAlpha = VectorToImageConvertingEstimator.Defaults.DefaultAlpha;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Default value for red channel. Will be used if ContainsRed set to false")]
            public int DefaultRed = VectorToImageConvertingEstimator.Defaults.DefaultRed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Default value for green channel. Will be used if ContainsGreen set to false")]
            public int DefaultGreen = VectorToImageConvertingEstimator.Defaults.DefaultGreen;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Default value for blue channel. Will be used if ContainsBlue set to false")]
            public int DefaultBlue = VectorToImageConvertingEstimator.Defaults.DefaultBlue;
        }

        internal const string Summary = "Converts vector array into image type.";
        internal const string UserName = "Vector To Image Transform";
        internal const string LoaderSignature = "VectorToImageConverter";
        internal const uint BeforeOrderVersion = 0x00010002;
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "VECTOIMG",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verWrittenCur: 0x00010003, // order for pixel colors, default colors, no size(float)
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(VectorToImageConvertingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "VectorToImageConverter";

        private readonly VectorToImageConvertingEstimator.ColumnOptions[] _columns;

        /// <summary>
        /// The columns passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<VectorToImageConvertingEstimator.ColumnOptions> Columns => _columns.AsReadOnly();

        internal VectorToImageConvertingTransformer(IHostEnvironment env, params VectorToImageConvertingEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(columns);

            _columns = columns.ToArray();
        }

        /// <param name="env">The host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="imageHeight">The height of the output images.</param>
        /// <param name="imageWidth">The width of the output images.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="colorsPresent">Specifies which <see cref="ImagePixelExtractingEstimator.ColorBits"/> are in present the input pixel vectors. The order of colors is specified in <paramref name="orderOfColors"/>.</param>
        /// <param name="orderOfColors">The order in which colors are presented in the input vector.</param>
        /// <param name="interleavedColors">Whether the pixels are interleaved, meaning whether they are in <paramref name="orderOfColors"/> order, or separated in the planar form, where the colors are specified one by one
        /// for all the pixels of the image. </param>
        /// <param name="scaleImage">Scale each pixel's color value by this amount.</param>
        /// <param name="offsetImage">Offset each pixel's color value by this amount.</param>
        /// <param name="defaultAlpha">Default value for alpha color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Alpha"/>.</param>
        /// <param name="defaultRed">Default value for red color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Red"/>.</param>
        /// <param name="defaultGreen">Default value for grenn color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Green"/>.</param>
        /// <param name="defaultBlue">Default value for blue color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Blue"/>.</param>
        internal VectorToImageConvertingTransformer(IHostEnvironment env, string outputColumnName,
            int imageHeight, int imageWidth,
            string inputColumnName = null,
            ImagePixelExtractingEstimator.ColorBits colorsPresent = ImagePixelExtractingEstimator.Defaults.Colors,
            ImagePixelExtractingEstimator.ColorsOrder orderOfColors = ImagePixelExtractingEstimator.Defaults.Order,
            bool interleavedColors = ImagePixelExtractingEstimator.Defaults.Interleave,
            float scaleImage = VectorToImageConvertingEstimator.Defaults.Scale,
            float offsetImage = VectorToImageConvertingEstimator.Defaults.Offset,
            int defaultAlpha = VectorToImageConvertingEstimator.Defaults.DefaultAlpha,
            int defaultRed = VectorToImageConvertingEstimator.Defaults.DefaultRed,
            int defaultGreen = VectorToImageConvertingEstimator.Defaults.DefaultGreen,
            int defaultBlue = VectorToImageConvertingEstimator.Defaults.DefaultBlue)
            : this(env, new VectorToImageConvertingEstimator.ColumnOptions(outputColumnName, imageHeight, imageWidth, inputColumnName, colorsPresent, orderOfColors, interleavedColors, scaleImage, offsetImage, defaultAlpha, defaultRed, defaultGreen, defaultBlue))
        {
        }

        // Constructor corresponding to SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Columns, nameof(args.Columns));

            var columns = new VectorToImageConvertingEstimator.ColumnOptions[args.Columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var item = args.Columns[i];
                columns[i] = new VectorToImageConvertingEstimator.ColumnOptions(item, args);
            }

            var transformer = new VectorToImageConvertingTransformer(env, columns);
            return new RowToRowMapperTransform(env, input, transformer.MakeRowMapper(input.Schema), transformer.MakeRowMapper);
        }

        private VectorToImageConvertingTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <base>
            // foreach added column
            //   ColumnOptions

            _columns = new VectorToImageConvertingEstimator.ColumnOptions[ColumnPairs.Length];
            for (int i = 0; i < _columns.Length; i++)
                _columns[i] = new VectorToImageConvertingEstimator.ColumnOptions(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, ctx);
        }

        private static VectorToImageConvertingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            if (ctx.Header.ModelVerWritten <= VectorToImageConvertingTransformer.BeforeOrderVersion)
                ctx.Reader.ReadFloat();
            return h.Apply("Loading Model",
            ch =>
            {
                return new VectorToImageConvertingTransformer(h, ctx);
            });
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
            // foreach added column
            //   ColInfoEx
            base.SaveColumns(ctx);

            for (int i = 0; i < _columns.Length; i++)
                _columns[i].Save(ctx);
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(VectorToImageConvertingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            var inputColName = _columns[col].InputColumnName;
            var vectorType = inputSchema[srcCol].Type as VectorType;
            if (vectorType == null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColName, "image", inputSchema[srcCol].Type.ToString());

            if (vectorType.GetValueCount() != _columns[col].ImageHeight * _columns[col].ImageWidth * _columns[col].Planes)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColName, new VectorType(vectorType.ItemType, _columns[col].ImageHeight, _columns[col].ImageWidth, _columns[col].Planes).ToString(), vectorType.ToString());
        }

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly VectorToImageConvertingTransformer _parent;
            private readonly ImageType[] _types;

            public Mapper(VectorToImageConvertingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = ConstructTypes(_parent._columns);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent._columns.Select((x, idx) => new DataViewSchema.DetachedColumn(x.Name, _types[idx], null)).ToArray();

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                Host.Assert(0 <= iinfo && iinfo < _parent._columns.Length);

                var type = _types[iinfo];
                var ex = _parent._columns[iinfo];
                bool needScale = ex.OffsetImage != 0 || ex.ScaleImage != 1;
                disposer = null;
                var sourceType = InputSchema[ColMapNewToOld[iinfo]].Type;
                var sourceItemType = sourceType.GetItemType();
                if (sourceItemType == NumberDataViewType.Single || sourceItemType == NumberDataViewType.Double)
                    return GetterFromType<float>(NumberDataViewType.Single, input, iinfo, ex, needScale);
                else
                    if (sourceItemType == NumberDataViewType.Byte)
                    return GetterFromType<byte>(NumberDataViewType.Byte, input, iinfo, ex, false);
                else
                    throw Contracts.Except("We only support float, double or byte arrays");
            }

            private ValueGetter<Bitmap> GetterFromType<TValue>(PrimitiveDataViewType srcType, DataViewRow input, int iinfo,
                VectorToImageConvertingEstimator.ColumnOptions ex, bool needScale) where TValue : IConvertible
            {
                Contracts.Assert(typeof(TValue) == srcType.RawType);
                var getSrc = RowCursorUtils.GetVecGetterAs<TValue>(srcType, input, ColMapNewToOld[iinfo]);
                var src = default(VBuffer<TValue>);
                int width = ex.ImageWidth;
                int height = ex.ImageHeight;
                float offset = ex.OffsetImage;
                float scale = ex.ScaleImage;

                return
                    (ref Bitmap dst) =>
                    {
                        getSrc(ref src);
                        if (src.GetValues().Length == 0)
                        {
                            dst = null;
                            return;
                        }
                        VBuffer<TValue> dense = default;
                        src.CopyToDense(ref dense);
                        var values = dense.GetValues();
                        dst = new Bitmap(width, height);
                        dst.SetResolution(width, height);
                        int cpix = height * width;
                        int position = 0;
                        ImagePixelExtractingEstimator.GetOrder(ex.Order, ex.Colors, out int a, out int r, out int b, out int g);

                        for (int y = 0; y < height; ++y)
                            for (int x = 0; x < width; x++)
                            {
                                float red = ex.DefaultRed;
                                float green = ex.DefaultGreen;
                                float blue = ex.DefaultBlue;
                                float alpha = ex.DefaultAlpha;
                                if (ex.InterleavedColors)
                                {
                                    if (ex.Alpha)
                                        alpha = Convert.ToSingle(values[position + a]);
                                    if (ex.Red)
                                        red = Convert.ToSingle(values[position + r]);
                                    if (ex.Green)
                                        green = Convert.ToSingle(values[position + g]);
                                    if (ex.Blue)
                                        blue = Convert.ToSingle(values[position + b]);
                                    position += ex.Planes;
                                }
                                else
                                {
                                    position = y * width + x;
                                    if (ex.Alpha) alpha = Convert.ToSingle(values[position + cpix * a]);
                                    if (ex.Red) red = Convert.ToSingle(values[position + cpix * r]);
                                    if (ex.Green) green = Convert.ToSingle(values[position + cpix * g]);
                                    if (ex.Blue) blue = Convert.ToSingle(values[position + cpix * b]);
                                }
                                Color pixel;
                                if (!needScale)
                                    pixel = Color.FromArgb((int)alpha, (int)red, (int)green, (int)blue);
                                else
                                {
                                    pixel = Color.FromArgb(
                                        ex.Alpha ? (int)Math.Round(alpha * scale - offset) : 0,
                                        (int)Math.Round(red * scale - offset),
                                        (int)Math.Round(green * scale - offset),
                                        (int)Math.Round(blue * scale - offset));
                                }
                                dst.SetPixel(x, y, pixel);
                            }
                    };
            }

            private static ImageType[] ConstructTypes(VectorToImageConvertingEstimator.ColumnOptions[] columns)
            {
                return columns.Select(c => new ImageType(c.ImageHeight, c.ImageWidth)).ToArray();
            }
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> that converts vectors containing pixel representations of images in to<see cref="ImageType"/> representation.
    /// </summary>
    /// <remarks>
    /// Calling <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> in this estimator, produces an <see cref="VectorToImageConvertingTransformer"/>.
    /// <seealso cref="ImageEstimatorsCatalog.ConvertToImage(TransformsCatalog, ColumnOptions[])" />
    /// <seealso cref="ImageEstimatorsCatalog.ConvertToImage(TransformsCatalog, int, int, string, string, ImagePixelExtractingEstimator.ColorBits, ImagePixelExtractingEstimator.ColorsOrder, bool, float, float, int, int, int, int)"/>
    /// <seealso cref="ImageEstimatorsCatalog"/>
    /// </remarks>
    public sealed class VectorToImageConvertingEstimator : TrivialEstimator<VectorToImageConvertingTransformer>
    {
        internal static class Defaults
        {
            public const float Scale = 1f;
            public const float Offset = 0f;
            public const int DefaultAlpha = 255;
            public const int DefaultRed = 0;
            public const int DefaultGreen = 0;
            public const int DefaultBlue = 0;
        }
        /// <summary>
        /// Describes how the transformer handles one vector to image conversion column pair.
        /// </summary>
        [BestFriend]
        internal sealed class ColumnOptions
        {
            /// <summary>Name of the column resulting from the transformation of <see cref="InputColumnName"/>.</summary>
            public readonly string Name;

            /// <summary> Name of column to transform.</summary>
            public readonly string InputColumnName;

            public readonly ImagePixelExtractingEstimator.ColorBits Colors;
            public readonly ImagePixelExtractingEstimator.ColorsOrder Order;
            public readonly bool InterleavedColors;
            public readonly byte Planes;

            public readonly int ImageWidth;
            public readonly int ImageHeight;
            public readonly float OffsetImage;
            public readonly float ScaleImage;

            public readonly int DefaultAlpha;
            public readonly int DefaultRed;
            public readonly int DefaultGreen;
            public readonly int DefaultBlue;

            public bool Alpha => (Colors & ImagePixelExtractingEstimator.ColorBits.Alpha) != 0;
            public bool Red => (Colors & ImagePixelExtractingEstimator.ColorBits.Red) != 0;
            public bool Green => (Colors & ImagePixelExtractingEstimator.ColorBits.Green) != 0;
            public bool Blue => (Colors & ImagePixelExtractingEstimator.ColorBits.Blue) != 0;

            internal ColumnOptions(VectorToImageConvertingTransformer.Column item, VectorToImageConvertingTransformer.Options args)
            {
                Contracts.CheckValue(item, nameof(item));
                Contracts.CheckValue(args, nameof(args));

                Name = item.Name;
                InputColumnName = item.Source ?? item.Name;

                if (item.ContainsAlpha ?? args.ContainsAlpha)
                { Colors |= ImagePixelExtractingEstimator.ColorBits.Alpha; Planes++; }
                if (item.ContainsRed ?? args.ContainsRed)
                { Colors |= ImagePixelExtractingEstimator.ColorBits.Red; Planes++; }
                if (item.ContainsGreen ?? args.ContainsGreen)
                { Colors |= ImagePixelExtractingEstimator.ColorBits.Green; Planes++; }
                if (item.ContainsBlue ?? args.ContainsBlue)
                { Colors |= ImagePixelExtractingEstimator.ColorBits.Blue; Planes++; }
                Contracts.CheckUserArg(Planes > 0, nameof(item.ContainsRed), "Need to use at least one color plane");

                Order = item.Order ?? args.Order;
                InterleavedColors = item.Interleave ?? args.Interleave;

                ImageWidth = item.ImageWidth ?? args.ImageWidth;
                ImageHeight = item.ImageHeight ?? args.ImageHeight;
                OffsetImage = item.Offset ?? args.Offset;
                ScaleImage = item.Scale ?? args.Scale;
                Contracts.CheckUserArg(FloatUtils.IsFinite(OffsetImage), nameof(item.Offset));
                Contracts.CheckUserArg(FloatUtils.IsFiniteNonZero(ScaleImage), nameof(item.Scale));
            }

            internal ColumnOptions(string outputColumnName, string inputColumnName, ModelLoadContext ctx)
            {
                Contracts.AssertNonEmpty(outputColumnName);
                Contracts.AssertNonEmpty(inputColumnName);
                Contracts.AssertValue(ctx);

                Name = outputColumnName;
                InputColumnName = inputColumnName;

                // *** Binary format ***
                // byte: colors
                // byte: order
                // int: widht
                // int: height
                // Float: offset
                // Float: scale
                // byte: interleave
                // int: defaultAlpha
                // int: defaultRed
                // int: defaultGreen
                // int: defaultBlue
                Colors = (ImagePixelExtractingEstimator.ColorBits)ctx.Reader.ReadByte();
                Contracts.CheckDecode(Colors != 0);
                Contracts.CheckDecode((Colors & ImagePixelExtractingEstimator.ColorBits.All) == Colors);

                // Count the planes.
                int planes = (int)Colors;
                planes = (planes & 0x05) + ((planes >> 1) & 0x05);
                planes = (planes & 0x03) + ((planes >> 2) & 0x03);
                Planes = (byte)planes;
                Contracts.Assert(0 < Planes & Planes <= 4);

                if (ctx.Header.ModelVerWritten <= VectorToImageConvertingTransformer.BeforeOrderVersion)
                    Order = ImagePixelExtractingEstimator.ColorsOrder.ARGB;
                else
                {
                    Order = (ImagePixelExtractingEstimator.ColorsOrder)ctx.Reader.ReadByte();
                    Contracts.CheckDecode(Order != 0);
                }

                ImageWidth = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(ImageWidth > 0);
                ImageHeight = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(ImageHeight > 0);
                OffsetImage = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFinite(OffsetImage));
                ScaleImage = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFiniteNonZero(ScaleImage));
                InterleavedColors = ctx.Reader.ReadBoolByte();

                if (ctx.Header.ModelVerWritten <= VectorToImageConvertingTransformer.BeforeOrderVersion)
                {
                    DefaultAlpha = 0;
                    DefaultRed = 0;
                    DefaultGreen = 0;
                    DefaultBlue = 0;
                }
                else
                {
                    DefaultAlpha = ctx.Reader.ReadInt32();
                    DefaultRed = ctx.Reader.ReadInt32();
                    DefaultGreen = ctx.Reader.ReadInt32();
                    DefaultBlue = ctx.Reader.ReadInt32();
                }
            }

            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="imageHeight">The height of the output images.</param>
            /// <param name="imageWidth">The width of the output images.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="colorsPresent">Specifies which <see cref="ImagePixelExtractingEstimator.ColorBits"/> are present in the input pixel vectors. The order of colors is specified in <paramref name="orderOfColors"/>.</param>
            /// <param name="orderOfColors">The order in which colors are presented in the input vector.</param>
            /// <param name="interleavedColors">Whether the pixels are interleaved, meaning whether they are in <paramref name="orderOfColors"/> order, or separated in the planar form, where the colors are specified one by one
            /// alpha, red, green, blue for all the pixels of the image. </param>
            /// <param name="scaleImage">The values are scaled by this value before being converted to pixels. Applied to vector value before <paramref name="offsetImage"/></param>
            /// <param name="offsetImage">The offset is subtracted before converting the values to pixels. Applied to vector value after <paramref name="scaleImage"/>.</param>
            /// <param name="defaultAlpha">Default value for alpha color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Alpha"/>.</param>
            /// <param name="defaultRed">Default value for red color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Red"/>.</param>
            /// <param name="defaultGreen">Default value for grenn color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Green"/>.</param>
            /// <param name="defaultBlue">Default value for blue color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Blue"/>.</param>
            public ColumnOptions(string name,
                int imageHeight, int imageWidth,
                string inputColumnName = null,
                ImagePixelExtractingEstimator.ColorBits colorsPresent = ImagePixelExtractingEstimator.Defaults.Colors,
                ImagePixelExtractingEstimator.ColorsOrder orderOfColors = ImagePixelExtractingEstimator.Defaults.Order,
                bool interleavedColors = ImagePixelExtractingEstimator.Defaults.Interleave,
                float scaleImage = VectorToImageConvertingEstimator.Defaults.Scale,
                float offsetImage = VectorToImageConvertingEstimator.Defaults.Offset,
                int defaultAlpha = VectorToImageConvertingEstimator.Defaults.DefaultAlpha,
                int defaultRed = VectorToImageConvertingEstimator.Defaults.DefaultRed,
                int defaultGreen = VectorToImageConvertingEstimator.Defaults.DefaultGreen,
                int defaultBlue = VectorToImageConvertingEstimator.Defaults.DefaultBlue)
            {
                Contracts.CheckNonWhiteSpace(name, nameof(name));

                Name = name;
                InputColumnName = inputColumnName ?? name;
                Colors = colorsPresent;
                if ((byte)(Colors & ImagePixelExtractingEstimator.ColorBits.Alpha) > 0)
                    Planes++;
                if ((byte)(Colors & ImagePixelExtractingEstimator.ColorBits.Red) > 0)
                    Planes++;
                if ((byte)(Colors & ImagePixelExtractingEstimator.ColorBits.Green) > 0)
                    Planes++;
                if ((byte)(Colors & ImagePixelExtractingEstimator.ColorBits.Blue) > 0)
                    Planes++;
                Contracts.CheckParam(Planes > 0, nameof(colorsPresent), "Need to use at least one color plane");

                Order = orderOfColors;
                InterleavedColors = interleavedColors;

                Contracts.CheckParam(imageWidth > 0, nameof(imageWidth), "Image width must be greater than zero");
                Contracts.CheckParam(imageHeight > 0, nameof(imageHeight), "Image height must be greater than zero");
                Contracts.CheckParam(FloatUtils.IsFinite(offsetImage), nameof(offsetImage));
                Contracts.CheckParam(FloatUtils.IsFiniteNonZero(scaleImage), nameof(scaleImage));
                ImageWidth = imageWidth;
                ImageHeight = imageHeight;
                OffsetImage = offsetImage;
                ScaleImage = scaleImage;
                DefaultAlpha = defaultAlpha;
                DefaultRed = defaultRed;
                DefaultGreen = defaultGreen;
                DefaultBlue = defaultBlue;
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

#if DEBUG
                // This code is used in deserialization - assert that it matches what we computed above.
                int planes = (int)Colors;
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
                // byte: interleave
                // int: defaultAlpha
                // int: defaultRed
                // int: defaultGreen
                // int: defaultBlue
                Contracts.Assert(Colors != 0);
                Contracts.Assert((Colors & ImagePixelExtractingEstimator.ColorBits.All) == Colors);
                ctx.Writer.Write((byte)Colors);
                ctx.Writer.Write((byte)Order);
                ctx.Writer.Write(ImageWidth);
                ctx.Writer.Write(ImageHeight);
                Contracts.Assert(FloatUtils.IsFinite(OffsetImage));
                ctx.Writer.Write(OffsetImage);
                Contracts.Assert(FloatUtils.IsFiniteNonZero(ScaleImage));
                ctx.Writer.Write(ScaleImage);
                ctx.Writer.WriteBoolByte(InterleavedColors);
                ctx.Writer.Write(DefaultAlpha);
                ctx.Writer.Write(DefaultRed);
                ctx.Writer.Write(DefaultGreen);
                ctx.Writer.Write(DefaultBlue);
            }
        }

        ///<summary>
        /// Convert pixels values into an image.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageHeight">The height of the output images.</param>
        /// <param name="imageWidth">The width of the output images.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>. Null means <paramref name="inputColumnName"/> is replaced.</param>
        /// <param name="inputColumnName">Name of the input column.</param>
        /// <param name="colorsPresent">Specifies which <see cref="ImagePixelExtractingEstimator.ColorBits"/> are in present the input pixel vectors. The order of colors is specified in <paramref name="orderOfColors"/>.</param>
        /// <param name="orderOfColors">The order in which colors are presented in the input vector.</param>
        /// <param name="interleavedColors">Whether the pixels are interleaved, meaning whether they are in <paramref name="orderOfColors"/> order, or separated in the planar form, where the colors are specified one by one
        /// alpha, red, green, blue for all the pixels of the image. </param>
        /// <param name="scaleImage">The values are scaled by this value before being converted to pixels. Applied to vector value before <paramref name="offsetImage"/>.</param>
        /// <param name="offsetImage">The offset is subtracted before converting the values to pixels. Applied to vector value after <paramref name="scaleImage"/>.</param>
        /// <param name="defaultAlpha">Default value for alpha color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Alpha"/>.</param>
        /// <param name="defaultRed">Default value for red color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Red"/>.</param>
        /// <param name="defaultGreen">Default value for grenn color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Green"/>.</param>
        /// <param name="defaultBlue">Default value for blue color, would be overriden if <paramref name="colorsPresent"/> contains <see cref="ImagePixelExtractingEstimator.ColorBits.Blue"/>.</param>
        [BestFriend]
        internal VectorToImageConvertingEstimator(IHostEnvironment env,
            int imageHeight,
            int imageWidth,
            string outputColumnName,
            string inputColumnName = null,
            ImagePixelExtractingEstimator.ColorBits colorsPresent = ImagePixelExtractingEstimator.Defaults.Colors,
            ImagePixelExtractingEstimator.ColorsOrder orderOfColors = ImagePixelExtractingEstimator.Defaults.Order,
            bool interleavedColors = ImagePixelExtractingEstimator.Defaults.Interleave,
            float scaleImage = VectorToImageConvertingEstimator.Defaults.Scale,
            float offsetImage = VectorToImageConvertingEstimator.Defaults.Offset,
            int defaultAlpha = VectorToImageConvertingEstimator.Defaults.DefaultAlpha,
            int defaultRed = VectorToImageConvertingEstimator.Defaults.DefaultRed,
            int defaultGreen = VectorToImageConvertingEstimator.Defaults.DefaultGreen,
            int defaultBlue = VectorToImageConvertingEstimator.Defaults.DefaultBlue)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(VectorToImageConvertingEstimator)),
                  new VectorToImageConvertingTransformer(env, outputColumnName, imageHeight, imageWidth, inputColumnName, colorsPresent, orderOfColors, interleavedColors, scaleImage, offsetImage, defaultAlpha, defaultRed, defaultGreen, defaultBlue))
        {
        }

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columnOptions">The <see cref="ColumnOptions"/> describing how the transform handles each vector to image conversion column pair.</param>
        internal VectorToImageConvertingEstimator(IHostEnvironment env, params ColumnOptions[] columnOptions)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(VectorToImageConvertingEstimator)), new VectorToImageConvertingTransformer(env, columnOptions))
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
                if (col.Kind != SchemaShape.Column.VectorKind.Vector || (col.ItemType != NumberDataViewType.Single && col.ItemType != NumberDataViewType.Double && col.ItemType != NumberDataViewType.Byte))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.InputColumnName, "known-size vector of type float, double or byte", col.GetTypeString());

                var itemType = new ImageType(colInfo.ImageHeight, colInfo.ImageWidth);
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Scalar, itemType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}

