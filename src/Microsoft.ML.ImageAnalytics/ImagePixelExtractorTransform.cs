// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Drawing;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.ImageAnalytics;

[assembly: LoadableClass(ImagePixelExtractorTransform.Summary, typeof(ImagePixelExtractorTransform), typeof(ImagePixelExtractorTransform.Arguments), typeof(SignatureDataTransform),
    ImagePixelExtractorTransform.UserName, "ImagePixelExtractorTransform", "ImagePixelExtractor")]

[assembly: LoadableClass(ImagePixelExtractorTransform.Summary, typeof(ImagePixelExtractorTransform), null, typeof(SignatureLoadDataTransform),
    ImagePixelExtractorTransform.UserName, ImagePixelExtractorTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    // REVIEW: Rewrite as LambdaTransform to simplify.
    public sealed class ImagePixelExtractorTransform : OneToOneTransformBase
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
        private enum ColorBits : byte
        {
            Alpha = 0x01,
            Red = 0x02,
            Green = 0x04,
            Blue = 0x08,

            All = Alpha | Red | Green | Blue
        }

        private sealed class ColInfoEx
        {
            public readonly ColorBits Colors;
            public readonly byte Planes;

            public readonly bool Convert;
            public readonly Single Offset;
            public readonly Single Scale;
            public readonly bool Interleave;

            public bool Alpha { get { return (Colors & ColorBits.Alpha) != 0; } }
            public bool Red { get { return (Colors & ColorBits.Red) != 0; } }
            public bool Green { get { return (Colors & ColorBits.Green) != 0; } }
            public bool Blue { get { return (Colors & ColorBits.Blue) != 0; } }

            public ColInfoEx(Column item, Arguments args)
            {
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

            public ColInfoEx(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);

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
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "ImagePixelExtractor";

        private readonly ColInfoEx[] _exes;
        private readonly VectorType[] _types;

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public ImagePixelExtractorTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column, input,
                t => t is ImageType ? null : "Expected Image type")
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                var item = args.Column[i];
                _exes[i] = new ColInfoEx(item, args);
            }

            _types = ConstructTypes(true);
        }

        private ImagePixelExtractorTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, t => t is ImageType ? null : "Expected Image type")
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // foreach added column
            //   ColInfoEx
            Host.AssertNonEmpty(Infos);
            _exes = new ColInfoEx[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
                _exes[i] = new ColInfoEx(ctx);

            _types = ConstructTypes(false);
        }

        public static ImagePixelExtractorTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model",
                ch =>
                {
                    // *** Binary format ***
                    // int: sizeof(Float)
                    // <remainder handled in ctors>
                    int cbFloat = ctx.Reader.ReadInt32();
                    ch.CheckDecode(cbFloat == sizeof(Single));
                    return new ImagePixelExtractorTransform(h, ctx, input);
                });
        }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // <base>
            // foreach added column
            //   ColInfoEx
            ctx.Writer.Write(sizeof(Single));
            SaveBase(ctx);

            Host.Assert(_exes.Length == Infos.Length);
            for (int i = 0; i < _exes.Length; i++)
                _exes[i].Save(ctx);
        }

        private VectorType[] ConstructTypes(bool user)
        {
            var types = new VectorType[Infos.Length];
            for (int i = 0; i < Infos.Length; i++)
            {
                var info = Infos[i];
                var ex = _exes[i];
                Host.Assert(ex.Planes > 0);

                var type = Source.Schema.GetColumnType(info.Source) as ImageType;
                Host.Assert(type != null);
                if (type.Height <= 0 || type.Width <= 0)
                {
                    // REVIEW: Could support this case by making the destination column be variable sized.
                    // However, there's no mechanism to communicate the dimensions through with the pixel data.
                    string name = Source.Schema.GetColumnName(info.Source);
                    throw user ?
                        Host.ExceptUserArg(nameof(Arguments.Column), "Column '{0}' does not have known size", name) :
                        Host.Except("Column '{0}' does not have known size", name);
                }
                int height = type.Height;
                int width = type.Width;
                Host.Assert(height > 0);
                Host.Assert(width > 0);
                Host.Assert((long)height * width <= int.MaxValue / 4);

                if (ex.Interleave)
                    types[i] = new VectorType(ex.Convert ? NumberType.Float : NumberType.U1, height, width, ex.Planes);
                else
                    types[i] = new VectorType(ex.Convert ? NumberType.Float : NumberType.U1, ex.Planes, height, width);
            }
            Metadata.Seal();
            return types;
        }

        protected override ColumnType GetColumnTypeCore(int iinfo)
        {
            Host.Assert(0 <= iinfo & iinfo < Infos.Length);
            return _types[iinfo];
        }

        protected override Delegate GetGetterCore(IChannel ch, IRow input, int iinfo, out Action disposer)
        {
            Host.AssertValueOrNull(ch);
            Host.AssertValue(input);
            Host.Assert(0 <= iinfo && iinfo < Infos.Length);

            if (_exes[iinfo].Convert)
                return GetGetterCore<Single>(input, iinfo, out disposer);
            return GetGetterCore<byte>(input, iinfo, out disposer);
        }

        private ValueGetter<VBuffer<TValue>> GetGetterCore<TValue>(IRow input, int iinfo, out Action disposer)
        {
            var type = _types[iinfo];
            Host.Assert(type.DimCount == 3);

            var ex = _exes[iinfo];

            int planes = ex.Interleave ? type.GetDim(2) : type.GetDim(0);
            int height = ex.Interleave ? type.GetDim(0) : type.GetDim(1);
            int width = ex.Interleave ? type.GetDim(1) : type.GetDim(2);

            int size = type.ValueCount;
            Host.Assert(size > 0);
            Host.Assert(size == planes * height * width);
            int cpix = height * width;

            var getSrc = GetSrcGetter<Bitmap>(input, iinfo);
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

                    Single offset = ex.Offset;
                    Single scale = ex.Scale;
                    Host.Assert(scale != 0);

                    var vf = values as Single[];
                    var vb = values as byte[];
                    Host.Assert(vf != null || vb != null);
                    bool needScale = offset != 0 || scale != 1;
                    Host.Assert(!needScale || vf != null);

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
                        {
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
                        }

                        Host.Assert(idst == size);
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
    }
}
