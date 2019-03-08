// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(ImagePixelExtractingTransformer.Summary, typeof(IDataTransform), typeof(ImagePixelExtractingTransformer), typeof(ImagePixelExtractingTransformer.Options), typeof(SignatureDataTransform),
    ImagePixelExtractingTransformer.UserName, "ImagePixelExtractorTransform", "ImagePixelExtractor")]

[assembly: LoadableClass(ImagePixelExtractingTransformer.Summary, typeof(IDataTransform), typeof(ImagePixelExtractingTransformer), null, typeof(SignatureLoadDataTransform),
    ImagePixelExtractingTransformer.UserName, ImagePixelExtractingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImagePixelExtractingTransformer), null, typeof(SignatureLoadModel),
    ImagePixelExtractingTransformer.UserName, ImagePixelExtractingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImagePixelExtractingTransformer), null, typeof(SignatureLoadRowMapper),
    ImagePixelExtractingTransformer.UserName, ImagePixelExtractingTransformer.LoaderSignature)]

namespace Microsoft.ML.ImageAnalytics
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
        internal class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public ImagePixelExtractingEstimator.ColumnOptions[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool? UseAlpha;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool? UseRed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool? UseGreen;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool? UseBlue;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Order of colors.")]
            public ImagePixelExtractingEstimator.ColorsOrder? Order;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in specified order")]
            public bool? Interleave;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to convert to floating point", ShortName = "conv")]
            public bool? AsFloat;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Offset (pre-scale)")]
            public float? Offset;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale factor")]
            public float? Scale;
        }

        /// <summary>
        /// Immutable class holding the settings for the <see cref="ImagePixelExtractingTransformer"/> for one input output column pair.
        /// </summary>
        internal sealed class ColumnInfos
        {
            public readonly string Name;
            public readonly string InputColumnName;
            public readonly ImagePixelExtractingEstimator.ColorBits Colors;
            public readonly ImagePixelExtractingEstimator.ColorsOrder Order;
            public readonly float Offset;
            public readonly float Scale;
            public readonly bool Interleave;
            public readonly bool AsFloat;

            internal readonly byte Planes;

            internal ColumnInfos(ImagePixelExtractingEstimator.ColumnOptions item, Options options)
            {
                Contracts.CheckValue(item, nameof(item));
                Contracts.CheckValue(options, nameof(options));

                Name = item.Name;
                InputColumnName = item.Source ?? item.Name;
                if (options.UseAlpha ?? item.UseAlpha) { Colors |= ImagePixelExtractingEstimator.ColorBits.Alpha; Planes++; }
                if (options.UseRed ?? item.UseRed) { Colors |= ImagePixelExtractingEstimator.ColorBits.Red; Planes++; }
                if (options.UseGreen ?? item.UseGreen) { Colors |= ImagePixelExtractingEstimator.ColorBits.Green; Planes++; }
                if (options.UseBlue ?? item.UseBlue) { Colors |= ImagePixelExtractingEstimator.ColorBits.Blue; Planes++; }
                Contracts.CheckUserArg(Planes > 0, nameof(item.UseRed), "Need to use at least one color plane");

                Order = options.Order ?? item.Order;
                Interleave = options.Interleave ?? item.Interleave;

                AsFloat = options.AsFloat ?? item.AsFloat;
                if (!AsFloat)
                {
                    Offset = ImagePixelExtractingEstimator.Defaults.Offset;
                    Scale = ImagePixelExtractingEstimator.Defaults.Scale;
                }
                else
                {
                    Offset = options.Offset ?? item.Offset;
                    Scale = options.Scale ?? item.Scale ;
                    Contracts.CheckUserArg(FloatUtils.IsFinite(Offset), nameof(item.Offset));
                    Contracts.CheckUserArg(FloatUtils.IsFiniteNonZero(Scale), nameof(item.Scale));
                }
            }

            public ColumnInfos(ImagePixelExtractingEstimator.ColumnOptions column)
            {
                Contracts.CheckNonWhiteSpace(column.Name, nameof(column.Name));

                Name = column.Name;
                InputColumnName = column.Source ?? column.Name;
                Colors = column.Colors;
                Order = column.Order;
                if ((Colors & ImagePixelExtractingEstimator.ColorBits.Alpha) == ImagePixelExtractingEstimator.ColorBits.Alpha) Planes++;
                if ((Colors & ImagePixelExtractingEstimator.ColorBits.Red) == ImagePixelExtractingEstimator.ColorBits.Red) Planes++;
                if ((Colors & ImagePixelExtractingEstimator.ColorBits.Green) == ImagePixelExtractingEstimator.ColorBits.Green) Planes++;
                if ((Colors & ImagePixelExtractingEstimator.ColorBits.Blue) == ImagePixelExtractingEstimator.ColorBits.Blue) Planes++;
                Contracts.CheckParam(Planes > 0, nameof(Colors), "Need to use at least one color plane.");

                Interleave = column.Interleave;

                AsFloat = column.AsFloat;
                if (!AsFloat)
                {
                    Offset = ImagePixelExtractingEstimator.Defaults.Offset;
                    Scale = ImagePixelExtractingEstimator.Defaults.Scale;
                }
                else
                {
                    Offset = column.Offset;
                    Scale = column.Scale;
                }
                Contracts.CheckParam(FloatUtils.IsFinite(Offset), nameof(column.Offset));
                Contracts.CheckParam(FloatUtils.IsFiniteNonZero(Scale), nameof(column.Scale));
            }

            internal ColumnInfos(string name, string inputColumnName, ModelLoadContext ctx)
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
                Colors = (ImagePixelExtractingEstimator.ColorBits)ctx.Reader.ReadByte();
                Contracts.CheckDecode(Colors != 0);
                Contracts.CheckDecode((Colors & ImagePixelExtractingEstimator.ColorBits.All) == Colors);
                if (ctx.Header.ModelVerWritten <= ImagePixelExtractingTransformer.BeforeOrderVersion)
                    Order = ImagePixelExtractingEstimator.ColorsOrder.ARGB;
                else
                {
                    Order = (ImagePixelExtractingEstimator.ColorsOrder)ctx.Reader.ReadByte();
                    Contracts.CheckDecode(Order != 0);
                }

                // Count the planes.
                int planes = (int)Colors;
                planes = (planes & 0x05) + ((planes >> 1) & 0x05);
                planes = (planes & 0x03) + ((planes >> 2) & 0x03);
                Planes = (byte)planes;
                Contracts.Assert(0 < Planes & Planes <= 4);

                AsFloat = ctx.Reader.ReadBoolByte();
                Offset = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFinite(Offset));
                Scale = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFiniteNonZero(Scale));
                Contracts.CheckDecode(AsFloat || Offset == 0 && Scale == 1);
                Interleave = ctx.Reader.ReadBoolByte();
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
                // byte: separateChannels
                Contracts.Assert(Colors != 0);
                Contracts.Assert((Colors & ImagePixelExtractingEstimator.ColorBits.All) == Colors);
                ctx.Writer.Write((byte)Colors);
                ctx.Writer.Write((byte)Order);
                ctx.Writer.WriteBoolByte(AsFloat);
                Contracts.Assert(FloatUtils.IsFinite(Offset));
                ctx.Writer.Write(Offset);
                Contracts.Assert(FloatUtils.IsFiniteNonZero(Scale));
                Contracts.Assert(AsFloat || Offset == 0 && Scale == 1);
                ctx.Writer.Write(Scale);
                ctx.Writer.WriteBoolByte(Interleave);
            }
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

        private readonly ColumnInfos[] _columns;

        /// <summary>
        /// The columns passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<ColumnInfos> Columns => _columns.AsReadOnly();

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columns">Describes the parameters of pixel extraction for each column pair.</param>
        internal ImagePixelExtractingTransformer(IHostEnvironment env, params ImagePixelExtractingEstimator.ColumnOptions[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.Select(column => new ColumnInfos(column)).ToArray();
        }

        private ImagePixelExtractingTransformer(IHostEnvironment env, Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(options.Columns))
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(options.Columns, nameof(options.Columns));
            _columns = options.Columns.Select(column => new ColumnInfos(column, options)).ToArray();
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(ImagePixelExtractingEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.Source)).ToArray();
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            env.CheckValue(input, nameof(input));
            var transformer = new ImagePixelExtractingTransformer(env, options);
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

            _columns = new ColumnInfos[ColumnPairs.Length];
            for (int i = 0; i < _columns.Length; i++)
                _columns[i] = new ColumnInfos(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, ctx);
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

                if (_parent._columns[iinfo].AsFloat)
                    return GetGetterCore<float>(input, iinfo, out disposer);
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

                int planes = ex.Interleave ? dims[2] : dims[0];
                int height = ex.Interleave ? dims[0] : dims[1];
                int width = ex.Interleave ? dims[1] : dims[2];

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

                        float offset = ex.Offset;
                        float scale = ex.Scale;
                        Contracts.Assert(scale != 0);

                        // REVIEW: split the getter into 2 specialized getters, one for float case and one for byte case.
                        Span<float> vf = typeof(TValue) == typeof(float) ? MemoryMarshal.Cast<TValue, float>(editor.Values) : default;
                        Span<byte> vb = typeof(TValue) == typeof(byte) ? MemoryMarshal.Cast<TValue, byte>(editor.Values) : default;
                        Contracts.Assert(!vf.IsEmpty || !vb.IsEmpty);
                        bool needScale = offset != 0 || scale != 1;
                        Contracts.Assert(!needScale || !vf.IsEmpty);

                        ImagePixelExtractingEstimator.GetOrder(ex.Order, ex.Colors, out int a, out int r, out int b, out int g);

                        int h = height;
                        int w = width;

                        if (ex.Interleave)
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

                    if (column.Interleave)
                        types[i] = new VectorType(column.AsFloat ? NumberDataViewType.Single : NumberDataViewType.Byte, height, width, column.Planes);
                    else
                        types[i] = new VectorType(column.AsFloat ? NumberDataViewType.Single : NumberDataViewType.Byte, column.Planes, height, width);
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
            public const bool AsFloat = true;
            public const float Scale = 1f;
            public const float Offset = 0f;
        }

        /// <summary>
        /// Describes how the transformer handles one image pixel extraction column pair.
        /// </summary>
        public sealed class ColumnOptions : OneToOneColumn
        {
            /// <summary> What colors to extract.</summary>
            [Argument(ArgumentType.AtMostOnce, Hide = true)]
            public ColorBits Colors = Defaults.Colors;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Order of channels")]
            public ColorsOrder Order = Defaults.Order;

            // REVIEW: Consider turning this into an enum that allows for pixel, line, or planar interleaving.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in specified order")]
            public bool Interleave = Defaults.Interleave;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to convert to floating point", ShortName = "conv")]
            public bool AsFloat = Defaults.AsFloat;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Offset (pre-scale)")]
            public float Offset = Defaults.Offset;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale factor")]
            public float Scale = Defaults.Scale;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            [BestFriend]
            internal bool UseAlpha;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            [BestFriend]
            internal bool UseRed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            [BestFriend]
            internal bool UseGreen;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            [BestFriend]
            internal bool UseBlue;

            internal static ColumnOptions Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new ColumnOptions();
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

                var itemType = colInfo.AsFloat ? NumberDataViewType.Single : NumberDataViewType.Byte;
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, itemType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
