// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;

[assembly: LoadableClass(ImagePixelExtractorTransform.Summary, typeof(IDataTransform), typeof(ImagePixelExtractorTransform), typeof(ImagePixelExtractorTransform.Arguments), typeof(SignatureDataTransform),
    ImagePixelExtractorTransform.UserName, "ImagePixelExtractorTransform", "ImagePixelExtractor")]

[assembly: LoadableClass(ImagePixelExtractorTransform.Summary, typeof(IDataTransform), typeof(ImagePixelExtractorTransform), null, typeof(SignatureLoadDataTransform),
    ImagePixelExtractorTransform.UserName, ImagePixelExtractorTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(ImagePixelExtractorTransform), null, typeof(SignatureLoadModel),
    ImagePixelExtractorTransform.UserName, ImagePixelExtractorTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImagePixelExtractorTransform), null, typeof(SignatureLoadRowMapper),
    ImagePixelExtractorTransform.UserName, ImagePixelExtractorTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    /// <summary>
    /// Transform which takes one or many columns of <see cref="ImageType"/> and convert them into vector representation.
    /// </summary>
    public sealed class ImagePixelExtractorTransform : OneToOneTransformerBase
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
            public bool InterleaveArgb = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to convert to floating point", ShortName = "conv")]
            public bool Convert = true;

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

        public sealed class ColumnInfo
        {
            public readonly string Input;
            public readonly string Output;

            public readonly ColorBits Colors;
            public readonly byte Planes;

            public readonly bool Convert;
            public readonly float Offset;
            public readonly float Scale;
            public readonly bool Interleave;

            public bool Alpha => (Colors & ColorBits.Alpha) != 0;
            public bool Red => (Colors & ColorBits.Red) != 0;
            public bool Green => (Colors & ColorBits.Green) != 0;
            public bool Blue => (Colors & ColorBits.Blue) != 0;

            internal ColumnInfo(Column item, Arguments args)
            {
                Contracts.CheckValue(item, nameof(item));
                Contracts.CheckValue(args, nameof(args));

                Input = item.Source ?? item.Name;
                Output = item.Name;

                if (item.UseAlpha ?? args.UseAlpha) { Colors |= ColorBits.Alpha; Planes++; }
                if (item.UseRed ?? args.UseRed) { Colors |= ColorBits.Red; Planes++; }
                if (item.UseGreen ?? args.UseGreen) { Colors |= ColorBits.Green; Planes++; }
                if (item.UseBlue ?? args.UseBlue) { Colors |= ColorBits.Blue; Planes++; }
                Contracts.CheckUserArg(Planes > 0, nameof(item.UseRed), "Need to use at least one color plane");

                Interleave = item.InterleaveArgb ?? args.InterleaveArgb;

                Convert = item.Convert ?? args.Convert;
                if (!Convert)
                {
                    Offset = 0;
                    Scale = 1;
                }
                else
                {
                    Offset = item.Offset ?? args.Offset ?? 0;
                    Scale = item.Scale ?? args.Scale ?? 1;
                    Contracts.CheckUserArg(FloatUtils.IsFinite(Offset), nameof(item.Offset));
                    Contracts.CheckUserArg(FloatUtils.IsFiniteNonZero(Scale), nameof(item.Scale));
                }
            }

            public ColumnInfo(string input, string output, ColorBits colors = ColorBits.Rgb, bool interleave = false)
                : this(input, output, colors, interleave, false, 1f, 0f)
            {
            }

            public ColumnInfo(string input, string output, ColorBits colors = ColorBits.Rgb, bool interleave = false, float scale = 1f, float offset = 0f)
                : this(input, output, colors, interleave, true, scale, offset)
            {
            }

            private ColumnInfo(string input, string output, ColorBits colors, bool interleave, bool convert, float scale, float offset)
            {
                Contracts.CheckNonEmpty(input, nameof(input));
                Contracts.CheckNonEmpty(output, nameof(output));

                Input = input;
                Output = output;
                Colors = colors;

                if ((Colors & ColorBits.Alpha) == ColorBits.Alpha) Planes++;
                if ((Colors & ColorBits.Red) == ColorBits.Red) Planes++;
                if ((Colors & ColorBits.Green) == ColorBits.Green) Planes++;
                if ((Colors & ColorBits.Blue) == ColorBits.Blue) Planes++;
                Contracts.CheckParam(Planes > 0, nameof(colors), "Need to use at least one color plane");

                Interleave = interleave;

                Convert = convert;
                if (!Convert)
                {
                    Offset = 0;
                    Scale = 1;
                }
                else
                {
                    Offset = offset;
                    Scale = scale;
                    Contracts.CheckParam(FloatUtils.IsFinite(Offset), nameof(offset));
                    Contracts.CheckParam(FloatUtils.IsFiniteNonZero(Scale), nameof(scale));
                }
            }

            internal ColumnInfo(string input, string output, ModelLoadContext ctx)
            {
                Contracts.AssertNonEmpty(input);
                Contracts.AssertNonEmpty(output);
                Contracts.AssertValue(ctx);

                Input = input;
                Output = output;

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

                Convert = ctx.Reader.ReadBoolByte();
                Offset = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFinite(Offset));
                Scale = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFiniteNonZero(Scale));
                Contracts.CheckDecode(Convert || Offset == 0 && Scale == 1);
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
                ctx.Writer.WriteBoolByte(Convert);
                Contracts.Assert(FloatUtils.IsFinite(Offset));
                ctx.Writer.Write(Offset);
                Contracts.Assert(FloatUtils.IsFiniteNonZero(Scale));
                Contracts.Assert(Convert || Offset == 0 && Scale == 1);
                ctx.Writer.Write(Scale);
                ctx.Writer.WriteBoolByte(Interleave);
            }
        }

        internal const string Summary = "Extract color plane(s) from an image. Options include scaling, offset and conversion to floating point.";
        internal const string UserName = "Image Pixel Extractor Transform";
        public const string LoaderSignature = "ImagePixelExtractor";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGPXEXT",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImagePixelExtractorTransform).Assembly.FullName);
        }

        private const string RegistrationName = "ImagePixelExtractor";

        private readonly ColumnInfo[] _columns;

        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();

        public ImagePixelExtractorTransform(IHostEnvironment env, string inputColumn, string outputColumn,
            ColorBits colors = ColorBits.Rgb, bool interleave = false)
            : this(env, new ColumnInfo(inputColumn, outputColumn, colors, interleave))
        {
        }

        public ImagePixelExtractorTransform(IHostEnvironment env, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            _columns = columns.ToArray();
        }

        private static (string input, string output)[] GetColumnPairs(ColumnInfo[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Input, x.Output)).ToArray();
        }

        // SignatureDataTransform.
        public static IDataTransform Create(IHostEnvironment env, Arguments args, IDataView input)
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

            var transformer = new ImagePixelExtractorTransform(env, columns);
            return new RowToRowMapperTransform(env, input, transformer.MakeRowMapper(input.Schema));
        }

        // Factory method for SignatureLoadModel.
        private static ImagePixelExtractorTransform Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(RegistrationName);
            host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new ImagePixelExtractorTransform(host, ctx);
        }

        private ImagePixelExtractorTransform(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>

            // for each added column
            //   ColumnInfo

            _columns = new ColumnInfo[ColumnPairs.Length];
            for (int i = 0; i < _columns.Length; i++)
                _columns[i] = new ColumnInfo(ColumnPairs[i].input, ColumnPairs[i].output, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
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

        protected override IRowMapper MakeRowMapper(ISchema schema)
            => new Mapper(this, Schema.Create(schema));

        protected override void CheckInputColumn(ISchema inputSchema, int col, int srcCol)
        {
            var inputColName = _columns[col].Input;
            var imageType = inputSchema.GetColumnType(srcCol) as ImageType;
            if (imageType == null)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColName, "image", inputSchema.GetColumnType(srcCol).ToString());
            if (imageType.Height <= 0 || imageType.Width <= 0)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColName, "known-size image", "unknown-size image");
            if ((long)imageType.Height * imageType.Width > int.MaxValue / 4)
                throw Host.Except("Image dimensions are too large");
        }

        private sealed class Mapper : MapperBase
        {
            private readonly ImagePixelExtractorTransform _parent;
            private readonly VectorType[] _types;

            public Mapper(ImagePixelExtractorTransform parent, Schema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
                _types = ConstructTypes();
            }

            public override Schema.Column[] GetOutputColumns()
                => _parent._columns.Select((x, idx) => new Schema.Column(x.Output, _types[idx], null)).ToArray();

            protected override Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent._columns.Length);

                if (_parent._columns[iinfo].Convert)
                    return GetGetterCore<Single>(input, iinfo, out disposer);
                return GetGetterCore<byte>(input, iinfo, out disposer);
            }

            //REVIEW Rewrite it to where TValue : IConvertible
            private ValueGetter<VBuffer<TValue>> GetGetterCore<TValue>(IRow input, int iinfo, out Action disposer)
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
                            dst = new VBuffer<TValue>(size, 0, dst.Values, dst.Indices);
                            return;
                        }

                        Host.Check(src.PixelFormat == System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                        Host.Check(src.Height == height && src.Width == width);

                        var values = dst.Values;
                        if (Utils.Size(values) < size)
                            values = new TValue[size];

                        float offset = ex.Offset;
                        float scale = ex.Scale;
                        Contracts.Assert(scale != 0);

                        var vf = values as float[];
                        var vb = values as byte[];
                        Contracts.Assert(vf != null || vb != null);
                        bool needScale = offset != 0 || scale != 1;
                        Contracts.Assert(!needScale || vf != null);

                        bool a = ex.Alpha;
                        bool r = ex.Red;
                        bool g = ex.Green;
                        bool b = ex.Blue;

                        int h = height;
                        int w = width;

                        if (ex.Interleave)
                        {
                            int idst = 0;
                            for (int x = 0; x < w; x++)
                                for (int y = 0; y < h; ++y)
                                {
                                    var pb = src.GetPixel(x, y);
                                    if (vb != null)
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
                            if (ex.Alpha)
                            {
                                // The image only has rgb but we need to supply alpha as well, so fake it up,
                                // assuming that it is 0xFF.
                                if (vf != null)
                                {
                                    Single v = (0xFF - offset) * scale;
                                    for (int i = 0; i < cpix; i++)
                                        vf[i] = v;
                                }
                                else
                                {
                                    for (int i = 0; i < cpix; i++)
                                        vb[i] = 0xFF;
                                }
                                idstMin = cpix;

                                // We've preprocessed alpha, avoid it in the
                                // scan operation below.
                                a = false;
                            }

                            for (int y = 0; y < h; ++y)
                            {
                                int idstBase = idstMin + y * w;

                                // Note that the bytes are in order BGR[A]. We arrange the layers in order ARGB.
                                if (vb != null)
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

                        dst = new VBuffer<TValue>(size, values, dst.Indices);
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
                        types[i] = new VectorType(column.Convert ? NumberType.Float : NumberType.U1, height, width, column.Planes);
                    else
                        types[i] = new VectorType(column.Convert ? NumberType.Float : NumberType.U1, column.Planes, height, width);
                }
                return types;
            }
        }
    }

    public sealed class ImagePixelExtractingEstimator : TrivialEstimator<ImagePixelExtractorTransform>
    {
        public ImagePixelExtractingEstimator(IHostEnvironment env, string inputColumn, string outputColumn,
                ImagePixelExtractorTransform.ColorBits colors = ImagePixelExtractorTransform.ColorBits.Rgb, bool interleave = false)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImagePixelExtractingEstimator)), new ImagePixelExtractorTransform(env, inputColumn, outputColumn, colors, interleave))
        {
        }

        public ImagePixelExtractingEstimator(IHostEnvironment env, params ImagePixelExtractorTransform.ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImagePixelExtractingEstimator)), new ImagePixelExtractorTransform(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(colInfo.Input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (!(col.ItemType is ImageType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, new ImageType().ToString(), col.GetTypeString());

                var itemType = colInfo.Convert ? NumberType.R4 : NumberType.U1;
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.Vector, itemType, false);
            }

            return new SchemaShape(result.Values);
        }

        private interface IColInput
        {
            Custom<Bitmap> Input { get; }

            ImagePixelExtractorTransform.ColumnInfo MakeColumnInfo(string input, string output);
        }

        internal sealed class OutPipelineColumn<T> : Vector<T>, IColInput
        {
            public Custom<Bitmap> Input { get; }
            private static readonly ImagePixelExtractorTransform.Arguments _defaultArgs = new ImagePixelExtractorTransform.Arguments();
            private readonly ImagePixelExtractorTransform.Column _colParam;

            public OutPipelineColumn(Custom<Bitmap> input, ImagePixelExtractorTransform.Column col)
                : base(Reconciler.Inst, input)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(typeof(T) == typeof(float) || typeof(T) == typeof(byte));
                Input = input;
                _colParam = col;
            }

            public ImagePixelExtractorTransform.ColumnInfo MakeColumnInfo(string input, string output)
            {
                // In principle, the analyzer should only call the the reconciler once for these columns.
                Contracts.Assert(_colParam.Source == null);
                Contracts.Assert(_colParam.Name == null);

                _colParam.Name = output;
                _colParam.Source = input;
                return new ImagePixelExtractorTransform.ColumnInfo(_colParam, _defaultArgs);
            }
        }

        /// <summary>
        /// Reconciler to an <see cref="ImagePixelExtractingEstimator"/> for the <see cref="PipelineColumn"/>.
        /// </summary>
        /// <remarks>Because we want to use the same reconciler for </remarks>
        /// <see cref="ImageStaticPipe.ExtractPixels(Custom{Bitmap}, bool, bool, bool, bool, bool, float, float)"/>
        /// <see cref="ImageStaticPipe.ExtractPixelsAsBytes(Custom{Bitmap}, bool, bool, bool, bool, bool)"/>
        private sealed class Reconciler : EstimatorReconciler
        {
            /// <summary>
            /// Because there are no global settings that cannot be overridden, we can always just use the same reconciler.
            /// </summary>
            public static Reconciler Inst = new Reconciler();

            private Reconciler() { }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                var cols = new ImagePixelExtractorTransform.ColumnInfo[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                {
                    var outCol = (IColInput)toOutput[i];
                    cols[i] = outCol.MakeColumnInfo(inputNames[outCol.Input], outputNames[toOutput[i]]);
                }
                return new ImagePixelExtractingEstimator(env, cols);
            }
        }
    }
}
