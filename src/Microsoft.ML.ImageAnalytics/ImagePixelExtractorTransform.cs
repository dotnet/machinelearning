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
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;

[assembly: LoadableClass(ImagePixelExtractorTransformer.Summary, typeof(IDataTransform), typeof(ImagePixelExtractorTransformer), typeof(ImagePixelExtractorTransformer.Arguments), typeof(SignatureDataTransform),
    ImagePixelExtractorTransformer.UserName, "ImagePixelExtractorTransform", "ImagePixelExtractor")]

[assembly: LoadableClass(ImagePixelExtractorTransformer.Summary, typeof(IDataTransform), typeof(ImagePixelExtractorTransformer), null, typeof(SignatureLoadDataTransform),
    ImagePixelExtractorTransformer.UserName, ImagePixelExtractorTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImagePixelExtractorTransformer), null, typeof(SignatureLoadModel),
    ImagePixelExtractorTransformer.UserName, ImagePixelExtractorTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImagePixelExtractorTransformer), null, typeof(SignatureLoadRowMapper),
    ImagePixelExtractorTransformer.UserName, ImagePixelExtractorTransformer.LoaderSignature)]

namespace Microsoft.ML.ImageAnalytics
{
    /// <summary>
    /// Transformer which takes one or many columns of <see cref="ImageType"/> and convert them into vector representation.
    /// </summary>
    public sealed class ImagePixelExtractorTransformer : OneToOneTransformerBase
    {
        public class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool? UseAlpha;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool? UseRed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool? UseGreen;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool? UseBlue;

            // REVIEW: Consider turning this into an enum that allows for pixel, line, or planar interleaving.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in ARGB order", ShortName = "interleave")]
            public bool? InterleaveArgb;

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
                    Offset != null || Scale != null || InterleaveArgb != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        public class Arguments : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool UseAlpha = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool UseRed = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool UseGreen = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool UseBlue = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in ARGB order", ShortName = "interleave")]
            public bool InterleaveArgb = Defaults.Interleave;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to convert to floating point", ShortName = "conv")]
            public bool Convert = Defaults.Convert;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Offset (pre-scale)")]
            public Single? Offset;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale factor")]
            public Single? Scale;
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

        internal static class Defaults
        {
            public const ColorBits Colors = ColorBits.Rgb;
            public const bool Interleave = false;
            public const bool Convert = true;
            public const float Scale = 1f;
            public const float Offset = 0f;
        }

        /// <summary>
        /// Describes how the transformer handles one image pixel extraction column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            public readonly string Name;
            public readonly string InputColumnName;

            public readonly ColorBits Colors;
            public readonly byte Planes;
            public readonly float Offset;
            public readonly float Scale;
            public readonly bool Interleave;
            public readonly bool AsFloat;

            public bool Alpha => (Colors & ColorBits.Alpha) != 0;
            public bool Red => (Colors & ColorBits.Red) != 0;
            public bool Green => (Colors & ColorBits.Green) != 0;
            public bool Blue => (Colors & ColorBits.Blue) != 0;

            internal ColumnInfo(Column item, Arguments args)
            {
                Contracts.CheckValue(item, nameof(item));
                Contracts.CheckValue(args, nameof(args));

                Name = item.Name;
                InputColumnName = item.Source ?? item.Name;

                if (item.UseAlpha ?? args.UseAlpha) { Colors |= ColorBits.Alpha; Planes++; }
                if (item.UseRed ?? args.UseRed) { Colors |= ColorBits.Red; Planes++; }
                if (item.UseGreen ?? args.UseGreen) { Colors |= ColorBits.Green; Planes++; }
                if (item.UseBlue ?? args.UseBlue) { Colors |= ColorBits.Blue; Planes++; }
                Contracts.CheckUserArg(Planes > 0, nameof(item.UseRed), "Need to use at least one color plane");

                Interleave = item.InterleaveArgb ?? args.InterleaveArgb;

                AsFloat = item.Convert ?? args.Convert;
                if (!AsFloat)
                {
                    Offset = Defaults.Offset;
                    Scale = Defaults.Scale;
                }
                else
                {
                    Offset = item.Offset ?? args.Offset ?? Defaults.Offset;
                    Scale = item.Scale ?? args.Scale ?? Defaults.Scale;
                    Contracts.CheckUserArg(FloatUtils.IsFinite(Offset), nameof(item.Offset));
                    Contracts.CheckUserArg(FloatUtils.IsFiniteNonZero(Scale), nameof(item.Scale));
                }
            }

            /// <summary>
            /// Describes how the transformer handles one input-output column pair.
            /// </summary>
            /// <param name="name">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
            /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="name"/> will be used as source.</param>
            /// <param name="colors">What colors to extract.</param>
            /// <param name="interleave"></param>
            /// <param name="scale">Scale color pixel value by this amount.</param>
            /// <param name="offset">Offset color pixel value by this amount.</param>
            /// <param name="asFloat">Output array as float array. If false, output as byte array.</param>

            public ColumnInfo(string name, string inputColumnName = null,
                ColorBits colors = Defaults.Colors,
                bool interleave = Defaults.Interleave,
                float scale = Defaults.Scale,
                float offset = Defaults.Offset,
                bool asFloat = Defaults.Convert)
            {
                Contracts.CheckNonWhiteSpace(name, nameof(name));

                Name = name;
                InputColumnName = inputColumnName ?? name;
                Colors = colors;

                if ((Colors & ColorBits.Alpha) == ColorBits.Alpha) Planes++;
                if ((Colors & ColorBits.Red) == ColorBits.Red) Planes++;
                if ((Colors & ColorBits.Green) == ColorBits.Green) Planes++;
                if ((Colors & ColorBits.Blue) == ColorBits.Blue) Planes++;
                Contracts.CheckParam(Planes > 0, nameof(colors), "Need to use at least one color plane");

                Interleave = interleave;

                AsFloat = asFloat;
                if (!AsFloat)
                {
                    Offset = Defaults.Offset;
                    Scale = Defaults.Scale;
                }
                else
                {
                    Offset = offset;
                    Scale = scale;
                }
                Contracts.CheckParam(FloatUtils.IsFinite(Offset), nameof(offset));
                Contracts.CheckParam(FloatUtils.IsFiniteNonZero(Scale), nameof(scale));
            }

            internal ColumnInfo(string name, string inputColumnName, ModelLoadContext ctx)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.AssertNonEmpty(inputColumnName);
                Contracts.AssertValue(ctx);

                Name = name;
                InputColumnName = inputColumnName;

                // *** Binary format ***
                // byte: colors
                // byte: convert
                // Float: offset
                // Float: scale
                // byte: separateChannels
                Colors = (ColorBits)ctx.Reader.ReadByte();
                Contracts.CheckDecode(Colors != 0);
                Contracts.CheckDecode((Colors & ColorBits.All) == Colors);

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

            public void Save(ModelSaveContext ctx)
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
                // byte: convert
                // Float: offset
                // Float: scale
                // byte: separateChannels
                Contracts.Assert(Colors != 0);
                Contracts.Assert((Colors & ColorBits.All) == Colors);
                ctx.Writer.Write((byte)Colors);
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

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGPXEXT",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImagePixelExtractorTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "ImagePixelExtractor";

        private readonly ColumnInfo[] _columns;

        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="colors">What colors to extract.</param>
        /// <param name="interleave"></param>
        /// <param name="scale">Scale color pixel value by this amount.</param>
        /// <param name="offset">Offset color pixel value by this amount.</param>
        /// <param name="asFloat">Output array as float array. If false, output as byte array.</param>
        public ImagePixelExtractorTransformer(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            ColorBits colors = ColorBits.Rgb,
            bool interleave = Defaults.Interleave,
            float scale = Defaults.Scale,
            float offset = Defaults.Offset,
            bool asFloat = Defaults.Convert)
            : this(env, new ColumnInfo(outputColumnName, inputColumnName, colors, interleave, scale, offset, asFloat))
        {
        }

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columns">Describes the parameters of pixel extraction for each column pair.</param>
        public ImagePixelExtractorTransformer(IHostEnvironment env, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Column, nameof(args.Column));

            var columns = new ColumnInfo[args.Column.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var item = args.Column[i];
                columns[i] = new ColumnInfo(item, args);
            }

            var transformer = new ImagePixelExtractorTransformer(env, columns);
            return new RowToRowMapperTransform(env, input, transformer.MakeRowMapper(input.Schema), transformer.MakeRowMapper);
        }

        // Factory method for SignatureLoadModel.
        private static ImagePixelExtractorTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new ImagePixelExtractorTransformer(host, ctx);
        }

        private ImagePixelExtractorTransformer(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>

            // for each added column
            //   ColumnInfo

            _columns = new ColumnInfo[ColumnPairs.Length];
            for (int i = 0; i < _columns.Length; i++)
                _columns[i] = new ColumnInfo(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, Schema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>

            // for each added column
            //   ColumnInfo

            base.SaveColumns(ctx);

            foreach (ColumnInfo info in _columns)
                info.Save(ctx);
        }

        private protected override IRowMapper MakeRowMapper(Schema schema) => new Mapper(this, schema);

        protected override void CheckInputColumn(Schema inputSchema, int col, int srcCol)
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
            private readonly ImagePixelExtractorTransformer _parent;
            private readonly VectorType[] _types;

            public Mapper(ImagePixelExtractorTransformer parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = ConstructTypes();
            }

            protected override Schema.DetachedColumn[] GetOutputColumnsCore()
                => _parent._columns.Select((x, idx) => new Schema.DetachedColumn(x.Name, _types[idx], null)).ToArray();

            protected override Delegate MakeGetter(Row input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent._columns.Length);

                if (_parent._columns[iinfo].AsFloat)
                    return GetGetterCore<Single>(input, iinfo, out disposer);
                return GetGetterCore<byte>(input, iinfo, out disposer);
            }

            //REVIEW Rewrite it to where TValue : IConvertible
            private ValueGetter<VBuffer<TValue>> GetGetterCore<TValue>(Row input, int iinfo, out Action disposer)
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

                var getSrc = input.GetGetter<Bitmap>(ColMapNewToOld[iinfo]);
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

                        bool a = ex.Alpha;
                        bool r = ex.Red;
                        bool g = ex.Green;
                        bool b = ex.Blue;

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
                                        if (a) { vb[idst++] = pb.A; }
                                        if (r) { vb[idst++] = pb.R; }
                                        if (g) { vb[idst++] = pb.G; }
                                        if (b) { vb[idst++] = pb.B; }
                                    }
                                    else if (!needScale)
                                    {
                                        if (a) { vf[idst++] = pb.A; }
                                        if (r) { vf[idst++] = pb.R; }
                                        if (g) { vf[idst++] = pb.G; }
                                        if (b) { vf[idst++] = pb.B; }
                                    }
                                    else
                                    {
                                        if (a) { vf[idst++] = (pb.A - offset) * scale; }
                                        if (r) { vf[idst++] = (pb.R - offset) * scale; }
                                        if (g) { vf[idst++] = (pb.G - offset) * scale; }
                                        if (b) { vf[idst++] = (pb.B - offset) * scale; }
                                    }
                                }
                            Contracts.Assert(idst == size);
                        }
                        else
                        {
                            int idstMin = 0;
                            for (int y = 0; y < h; ++y)
                            {
                                int idstBase = idstMin + y * w;

                                // Note that the bytes are in order BGR[A]. We arrange the layers in order ARGB.
                                if (!vb.IsEmpty)
                                {
                                    for (int x = 0; x < w; x++, idstBase++)
                                    {
                                        var pb = src.GetPixel(x, y);
                                        int idst = idstBase;
                                        if (a) { vb[idst] = pb.A; idst += cpix; }
                                        if (r) { vb[idst] = pb.R; idst += cpix; }
                                        if (g) { vb[idst] = pb.G; idst += cpix; }
                                        if (b) { vb[idst] = pb.B; idst += cpix; }
                                    }
                                }
                                else if (!needScale)
                                {
                                    for (int x = 0; x < w; x++, idstBase++)
                                    {
                                        var pb = src.GetPixel(x, y);
                                        int idst = idstBase;
                                        if (a) { vf[idst] = pb.A; idst += cpix; }
                                        if (r) { vf[idst] = pb.R; idst += cpix; }
                                        if (g) { vf[idst] = pb.G; idst += cpix; }
                                        if (b) { vf[idst] = pb.B; idst += cpix; }
                                    }
                                }
                                else
                                {
                                    for (int x = 0; x < w; x++, idstBase++)
                                    {
                                        var pb = src.GetPixel(x, y);
                                        int idst = idstBase;
                                        if (a) { vf[idst] = (pb.A - offset) * scale; idst += cpix; }
                                        if (r) { vf[idst] = (pb.R - offset) * scale; idst += cpix; }
                                        if (g) { vf[idst] = (pb.G - offset) * scale; idst += cpix; }
                                        if (b) { vf[idst] = (pb.B - offset) * scale; idst += cpix; }
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
                        types[i] = new VectorType(column.AsFloat ? NumberType.Float : NumberType.U1, height, width, column.Planes);
                    else
                        types[i] = new VectorType(column.AsFloat ? NumberType.Float : NumberType.U1, column.Planes, height, width);
                }
                return types;
            }
        }
    }

    /// <summary>
    /// Estimator which extract pixel values out of image and produce array of values.
    /// </summary>
    public sealed class ImagePixelExtractingEstimator : TrivialEstimator<ImagePixelExtractorTransformer>
    {
        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>. Null means <paramref name="inputColumnName"/> is replaced.</param>
        /// <param name="inputColumnName">Name of the input column.</param>
        /// <param name="colors">What colors to extract.</param>
        /// <param name="interleave"></param>
        /// <param name="scale">Scale color pixel value by this amount.</param>
        /// <param name="offset">Offset color pixel value by this amount.</param>
        /// <param name="asFloat">Output array as float array. If false, output as byte array.</param>
        public ImagePixelExtractingEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            ImagePixelExtractorTransformer.ColorBits colors = ImagePixelExtractorTransformer.Defaults.Colors,
            bool interleave = ImagePixelExtractorTransformer.Defaults.Interleave, float scale = ImagePixelExtractorTransformer.Defaults.Scale,
            float offset = ImagePixelExtractorTransformer.Defaults.Offset, bool asFloat = ImagePixelExtractorTransformer.Defaults.Convert)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImagePixelExtractingEstimator)), new ImagePixelExtractorTransformer(env, outputColumnName, inputColumnName, colors, interleave))
        {
        }

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columns">Describes the parameters of pixel extraction for each column pair.</param>
        public ImagePixelExtractingEstimator(IHostEnvironment env, params ImagePixelExtractorTransformer.ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImagePixelExtractingEstimator)), new ImagePixelExtractorTransformer(env, columns))
        {
        }

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

                var itemType = colInfo.AsFloat ? NumberType.R4 : NumberType.U1;
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Vector, itemType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
