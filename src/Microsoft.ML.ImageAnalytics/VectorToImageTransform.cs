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
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

[assembly: LoadableClass(VectorToImageTransform.Summary, typeof(VectorToImageTransform), typeof(VectorToImageTransform.Arguments),
    typeof(SignatureDataTransform), VectorToImageTransform.UserName, "VectorToImageTransform", "VectorToImage")]

[assembly: LoadableClass(VectorToImageTransform.Summary, typeof(VectorToImageTransform), null, typeof(SignatureLoadDataTransform),
    VectorToImageTransform.UserName, VectorToImageTransform.LoaderSignature)]

namespace Microsoft.ML.Runtime.ImageAnalytics
{
    // REVIEW: Rewrite as LambdaTransform to simplify.

    /// <summary>
    /// Transform which takes one or many columns with vectors in them and transform them to <see cref="ImageType"/> representation.
    /// </summary>
    public sealed class VectorToImageTransform : OneToOneTransformBase
    {
        public class Column : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool? ContainsAlpha;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool? ContainsRed;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool? ContainsGreen;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool? ContainsBlue;

            // REVIEW: Consider turning this into an enum that allows for pixel, line, or planar interleaving.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in ARGB order", ShortName = "interleave")]
            public bool? InterleaveArgb;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Width of the image", ShortName = "width")]
            public int? ImageWidth;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Height of the image", ShortName = "height")]
            public int? ImageHeight;

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
                if (ContainsAlpha != null || ContainsRed != null || ContainsGreen != null || ContainsBlue != null || ImageWidth != null ||
                    ImageHeight != null || Offset != null || Scale != null || InterleaveArgb != null)
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
            public bool ContainsAlpha = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool ContainsRed = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool ContainsGreen = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool ContainsBlue = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in ARGB order", ShortName = "interleave")]
            public bool InterleaveArgb = false;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Width of the image", ShortName = "width")]
            public int ImageWidth;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Height of the image", ShortName = "height")]
            public int ImageHeight;

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

            public readonly int Width;
            public readonly int Height;
            public readonly Single Offset;
            public readonly Single Scale;
            public readonly bool Interleave;

            public bool Alpha { get { return (Colors & ColorBits.Alpha) != 0; } }
            public bool Red { get { return (Colors & ColorBits.Red) != 0; } }
            public bool Green { get { return (Colors & ColorBits.Green) != 0; } }
            public bool Blue { get { return (Colors & ColorBits.Blue) != 0; } }

            public ColInfoEx(Column item, Arguments args)
            {
                if (item.ContainsAlpha ?? args.ContainsAlpha) { Colors |= ColorBits.Alpha; Planes++; }
                if (item.ContainsRed ?? args.ContainsRed) { Colors |= ColorBits.Red; Planes++; }
                if (item.ContainsGreen ?? args.ContainsGreen) { Colors |= ColorBits.Green; Planes++; }
                if (item.ContainsBlue ?? args.ContainsBlue) { Colors |= ColorBits.Blue; Planes++; }
                Contracts.CheckUserArg(Planes > 0, nameof(item.ContainsRed), "Need to use at least one color plane");

                Interleave = item.InterleaveArgb ?? args.InterleaveArgb;

                Width = item.ImageWidth ?? args.ImageWidth;
                Height = item.ImageHeight ?? args.ImageHeight;
                Offset = item.Offset ?? args.Offset ?? 0;
                Scale = item.Scale ?? args.Scale ?? 1;
                Contracts.CheckUserArg(FloatUtils.IsFinite(Offset), nameof(item.Offset));
                Contracts.CheckUserArg(FloatUtils.IsFiniteNonZero(Scale), nameof(item.Scale));
            }

            public ColInfoEx(ModelLoadContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // byte: colors
                // int: widht
                // int: height
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

                Width = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Width > 0);
                Height = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(Height > 0);
                Offset = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFinite(Offset));
                Scale = ctx.Reader.ReadFloat();
                Contracts.CheckDecode(FloatUtils.IsFiniteNonZero(Scale));
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
                ctx.Writer.Write(Width);
                ctx.Writer.Write(Height);
                Contracts.Assert(FloatUtils.IsFinite(Offset));
                ctx.Writer.Write(Offset);
                Contracts.Assert(FloatUtils.IsFiniteNonZero(Scale));
                ctx.Writer.Write(Scale);
                ctx.Writer.WriteBoolByte(Interleave);
            }
        }

        public const string Summary = "Converts vector array into image type.";
        public const string UserName = "Vector To Image Transform";
        public const string LoaderSignature = "VectorToImageConverter";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "VECTOIMG",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private const string RegistrationName = "VectorToImageConverter";

        private readonly ColInfoEx[] _exes;
        private readonly ImageType[] _types;

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public VectorToImageTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, Contracts.CheckRef(args, nameof(args)).Column, input,
                t => t is VectorType ? null : "Expected VectorType type")
        {
            Host.AssertNonEmpty(Infos);
            Host.Assert(Infos.Length == Utils.Size(args.Column));

            _exes = new ColInfoEx[Infos.Length];
            _types = new ImageType[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                var item = args.Column[i];
                _exes[i] = new ColInfoEx(item, args);
                _types[i] = new ImageType(_exes[i].Height, _exes[i].Width);
            }
            Metadata.Seal();
        }

        private VectorToImageTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, ctx, input, t => t is VectorType ? null : "Expected VectorType type")
        {
            Host.AssertValue(ctx);

            // *** Binary format ***
            // <prefix handled in static Create method>
            // <base>
            // foreach added column
            //   ColInfoEx
            Host.AssertNonEmpty(Infos);
            _exes = new ColInfoEx[Infos.Length];
            _types = new ImageType[Infos.Length];
            for (int i = 0; i < _exes.Length; i++)
            {
                _exes[i] = new ColInfoEx(ctx);
                _types[i] = new ImageType(_exes[i].Height, _exes[i].Width);
            }
            Metadata.Seal();
        }

        public static VectorToImageTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
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
                    return new VectorToImageTransform(h, ctx, input);
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

            var type = _types[iinfo];
            var ex = _exes[iinfo];
            bool needScale = ex.Offset != 0 || ex.Scale != 1;
            disposer = null;
            var sourceType = Schema.GetColumnType(Infos[iinfo].Source);
            if (sourceType.ItemType == NumberType.R4 || sourceType.ItemType == NumberType.R8)
                return GetterFromType<float>(input, iinfo, ex, needScale);
            else
                if (sourceType.ItemType == NumberType.U1)
                return GetterFromType<byte>(input, iinfo, ex, false);
            else
                throw Contracts.Except("We only support float or byte arrays");

        }

        private ValueGetter<Bitmap> GetterFromType<TValue>(IRow input, int iinfo, ColInfoEx ex, bool needScale) where TValue : IConvertible
        {
            var getSrc = GetSrcGetter<VBuffer<TValue>>(input, iinfo);
            var src = default(VBuffer<TValue>);
            int width = ex.Width;
            int height = ex.Height;
            float offset = ex.Offset;
            float scale = ex.Scale;

            return
                (ref Bitmap dst) =>
                {
                    getSrc(ref src);
                    if (src.Count == 0)
                    {
                        dst = null;
                        return;
                    }
                    VBuffer<TValue> dense = default;
                    src.CopyToDense(ref dense);
                    var values = dense.Values;
                    dst = new Bitmap(width, height);
                    dst.SetResolution(width, height);
                    int cpix = height * width;
                    int planes = dense.Count / cpix;
                    int position = 0;

                    for (int x = 0; x < width; x++)
                        for (int y = 0; y < height; ++y)
                        {
                            float R = 0;
                            float G = 0;
                            float B = 0;
                            float A = 0;
                            if (ex.Interleave)
                            {
                                if (ex.Alpha) position++;
                                if (ex.Red) R = Convert.ToSingle(values[position++]);
                                if (ex.Green) G = Convert.ToSingle(values[position++]);
                                if (ex.Blue) B = Convert.ToSingle(values[position++]);
                            }
                            else
                            {
                                position = y * width + x;
                                if (ex.Alpha) { A = Convert.ToSingle(values[position]); position += cpix; }
                                if (ex.Red) { R = Convert.ToSingle(values[position]); position += cpix; }
                                if (ex.Green) { G = Convert.ToSingle(values[position]); position += cpix; }
                                if (ex.Blue) { B = Convert.ToSingle(values[position]); position += cpix; }
                            }
                            Color pixel;
                            if (!needScale)
                                pixel = Color.FromArgb((int)A, (int)R, (int)G, (int)B);
                            else
                            {
                                pixel = Color.FromArgb(
                                    (int)((A - offset) * scale),
                                    (int)((R - offset) * scale),
                                    (int)((G - offset) * scale),
                                    (int)((B - offset) * scale));
                            }
                            dst.SetPixel(x, y, pixel);
                        }
                };
        }
    }
}

