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

[assembly: LoadableClass(ImagePixelExtractorTransform.Summary, typeof(IDataTransform), typeof(ImagePixelExtractorTransform), typeof(ImagePixelExtractorTransform.Arguments), typeof(SignatureDataTransform),
    ImagePixelExtractorTransform.UserName, "ImagePixelExtractorTransform", "ImagePixelExtractor")]

[assembly: LoadableClass(ImagePixelExtractorTransform.Summary, typeof(IDataTransform), typeof(ImagePixelExtractorTransform), null, typeof(SignatureLoadDataTransform),
    ImagePixelExtractorTransform.UserName, ImagePixelExtractorTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(ImagePixelExtractorTransform), null, typeof(SignatureLoadModel),
    ImagePixelExtractorTransform.UserName, ImagePixelExtractorTransform.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImagePixelExtractorTransform.Mapper), null, typeof(SignatureLoadRowMapper),
    ImagePixelExtractorTransform.UserName, ImagePixelExtractorTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    /// <summary>
    /// Transform which takes one or many columns of <see cref="ImageType"/> and convert them into vector representation.
    /// </summary>
    public sealed class ImagePixelExtractorTransform : ITransformer, ICanSaveModel
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

            public bool Alpha { get { return (Colors & ColorBits.Alpha) != 0; } }
            public bool Red { get { return (Colors & ColorBits.Red) != 0; } }
            public bool Green { get { return (Colors & ColorBits.Green) != 0; } }
            public bool Blue { get { return (Colors & ColorBits.Blue) != 0; } }

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
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "ImagePixelExtractor";

        private readonly IHost _host;
        private readonly ColumnInfo[] _columns;

        public IReadOnlyCollection<ColumnInfo> Columns => _columns.AsReadOnly();

        public ImagePixelExtractorTransform(IHostEnvironment env, string inputColumn, string outputColumn,
            ColorBits colors = ColorBits.Rgb, bool interleave = false)
            : this(env, new ColumnInfo(inputColumn, outputColumn, colors, interleave))
        {
        }

        public ImagePixelExtractorTransform(IHostEnvironment env, params ColumnInfo[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(columns, nameof(columns));

            _columns = columns.ToArray();
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

        public ImagePixelExtractorTransform(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: id of input column name

            // for each added column
            //   ColumnInfo

            int n = ctx.Reader.ReadInt32();

            var names = new (string input, string output)[n];
            for (int i = 0; i < n; i++)
            {
                var output = ctx.LoadNonEmptyString();
                var input = ctx.LoadNonEmptyString();
                names[i] = (input, output);
            }

            _columns = new ColumnInfo[n];
            for (int i = 0; i < _columns.Length; i++)
                _columns[i] = new ColumnInfo(names[i].input, names[i].output, ctx);
        }

        // Factory method for SignatureLoadDataTransform.
        public static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            var transformer = new ImagePixelExtractorTransform(env, ctx);
            return new RowToRowMapperTransform(env, input, transformer.MakeRowMapper(input.Schema));
        }

        public void Save(ModelSaveContext ctx) => SaveContents(_host, ctx, _columns);

        private static void SaveContents(IHostEnvironment env, ModelSaveContext ctx, ColumnInfo[] columns)
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

            // for each added column
            //   ColumnInfo

            ctx.Writer.Write(columns.Length);
            for (int i = 0; i < columns.Length; i++)
            {
                ctx.SaveNonEmptyString(columns[i].Output);
                ctx.SaveNonEmptyString(columns[i].Input);
            }

            for (int i = 0; i < columns.Length; i++)
                columns[i].Save(ctx);
        }

        private IRowMapper MakeRowMapper(ISchema schema)
            => new Mapper(_host, _columns, schema);

        private static void CheckInput(IExceptionContext ctx, ISchema inputSchema, string input, out int srcCol)
        {
            Contracts.AssertValueOrNull(ctx);
            Contracts.AssertValue(inputSchema);
            Contracts.AssertNonEmpty(input);

            if (!inputSchema.TryGetColumnIndex(input, out srcCol))
                throw ctx.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
            var imageType = inputSchema.GetColumnType(srcCol) as ImageType;
            if (imageType == null)
                throw ctx.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "image", inputSchema.GetColumnType(srcCol).ToString());
            if (imageType.Height <= 0 || imageType.Width <= 0)
                throw ctx.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "known-size image", "unknown-size image");
            if ((long)imageType.Height * imageType.Width > int.MaxValue / 4)
                throw ctx.Except("Image dimensions are too large");
        }

        public ISchema GetOutputSchema(ISchema inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            // Check that all the input columns are present and are images of known size.
            foreach (var column in _columns)
                CheckInput(_host, inputSchema, column.Input, out int col);

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
            private readonly ColumnInfo[] _columns;
            private readonly VectorType[] _types;
            private readonly ISchema _inputSchema;
            private readonly Dictionary<int, int> _colMapNewToOld;

            public Mapper(IHostEnvironment env, ColumnInfo[] columns, ISchema inputSchema)
            {
                Contracts.AssertValue(env);
                _host = env.Register(nameof(Mapper));
                _host.AssertValue(columns);
                _host.AssertValue(inputSchema);

                _colMapNewToOld = new Dictionary<int, int>();
                for (int i = 0; i < columns.Length; i++)
                {
                    CheckInput(_host, inputSchema, columns[i].Input, out int srcCol);
                    _colMapNewToOld.Add(i, srcCol);
                }

                _columns = columns;
                _inputSchema = inputSchema;
                _types = ConstructTypes();
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

            public Func<int, bool> GetDependencies(Func<int, bool> activeOutput)
            {
                var active = new bool[_inputSchema.ColumnCount];
                foreach (var pair in _colMapNewToOld)
                    if (activeOutput(pair.Key))
                        active[pair.Value] = true;
                return col => active[col];
            }

            public RowMapperColumnInfo[] GetOutputColumns()
                => _columns.Select((x, idx) => new RowMapperColumnInfo(x.Output, _types[idx], null)).ToArray();

            public void Save(ModelSaveContext ctx) => SaveContents(_host, ctx, _columns);

            // Factory method for SignatureLoadRowMapper.
            public static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, ISchema inputSchema)
            {
                Contracts.CheckValue(env, nameof(env));
                env.CheckValue(ctx, nameof(ctx));
                env.CheckValue(inputSchema, nameof(inputSchema));
                var transformer = new ImagePixelExtractorTransform(env, ctx);
                return transformer.MakeRowMapper(inputSchema);
            }

            private Delegate MakeGetter(IRow input, int iinfo, out Action disposer)
            {
                _host.AssertValue(input);
                _host.Assert(0 <= iinfo && iinfo < _columns.Length);

                if (_columns[iinfo].Convert)
                    return GetGetterCore<Single>(input, iinfo, out disposer);
                return GetGetterCore<byte>(input, iinfo, out disposer);
            }

            //REVIEW Rewrite it to where TValue : IConvertible
            private ValueGetter<VBuffer<TValue>> GetGetterCore<TValue>(IRow input, int iinfo, out Action disposer)
            {
                var type = _types[iinfo];
                _host.Assert(type.DimCount == 3);

                var ex = _columns[iinfo];

                int planes = ex.Interleave ? type.GetDim(2) : type.GetDim(0);
                int height = ex.Interleave ? type.GetDim(0) : type.GetDim(1);
                int width = ex.Interleave ? type.GetDim(1) : type.GetDim(2);

                int size = type.ValueCount;
                _host.Assert(size > 0);
                _host.Assert(size == planes * height * width);
                int cpix = height * width;

                var getSrc = input.GetGetter<Bitmap>(_colMapNewToOld[iinfo]);
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

                        _host.Check(src.PixelFormat == System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                        _host.Check(src.Height == height && src.Width == width);

                        var values = dst.Values;
                        if (Utils.Size(values) < size)
                            values = new TValue[size];

                        Single offset = ex.Offset;
                        Single scale = ex.Scale;
                        _host.Assert(scale != 0);

                        var vf = values as Single[];
                        var vb = values as byte[];
                        _host.Assert(vf != null || vb != null);
                        bool needScale = offset != 0 || scale != 1;
                        _host.Assert(!needScale || vf != null);

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
                                    var pb = src.GetPixel(y, x);
                                    if (vb != null)
                                    {
                                        if (a) { vb[idst++] = (byte)0; }
                                        if (r) { vb[idst++] = pb.R; }
                                        if (g) { vb[idst++] = pb.G; }
                                        if (b) { vb[idst++] = pb.B; }
                                    }
                                    else if (!needScale)
                                    {
                                        if (a) { vf[idst++] = 0.0f; }
                                        if (r) { vf[idst++] = pb.R; }
                                        if (g) { vf[idst++] = pb.G; }
                                        if (b) { vf[idst++] = pb.B; }
                                    }
                                    else
                                    {
                                        if (a) { vf[idst++] = 0.0f; }
                                        if (r) { vf[idst++] = (pb.R - offset) * scale; }
                                        if (g) { vf[idst++] = (pb.B - offset) * scale; }
                                        if (b) { vf[idst++] = (pb.G - offset) * scale; }
                                    }
                                }
                            _host.Assert(idst == size);
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
                var types = new VectorType[_columns.Length];
                for (int i = 0; i < _columns.Length; i++)
                {
                    var column = _columns[i];
                    Contracts.Assert(column.Planes > 0);

                    var type = _inputSchema.GetColumnType(_colMapNewToOld[i]) as ImageType;
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

    public sealed class ImagePixelExtractorEstimator: TrivialEstimator<ImagePixelExtractorTransform>
    {
        public ImagePixelExtractorEstimator(IHostEnvironment env, string inputColumn, string outputColumn,
                ImagePixelExtractorTransform.ColorBits colors = ImagePixelExtractorTransform.ColorBits.Rgb, bool interleave = false)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImagePixelExtractorEstimator)), new ImagePixelExtractorTransform(env, inputColumn, outputColumn, colors, interleave))
        {
        }

        public ImagePixelExtractorEstimator(IHostEnvironment env, params ImagePixelExtractorTransform.ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImagePixelExtractorEstimator)), new ImagePixelExtractorTransform(env, columns))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var colInfo in Transformer.Columns)
            {
                var col = inputSchema.FindColumn(colInfo.Input);

                if (col == null)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input);
                if (!(col.ItemType is ImageType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", colInfo.Input, new ImageType().ToString(), col.GetTypeString());

                var itemType = colInfo.Convert ? NumberType.R4 : NumberType.U1;
                result[colInfo.Output] = new SchemaShape.Column(colInfo.Output, SchemaShape.Column.VectorKind.Vector, itemType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
