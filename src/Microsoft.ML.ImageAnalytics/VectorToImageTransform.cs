// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model;

[assembly: LoadableClass(VectorToImageConvertingTransformer.Summary, typeof(IDataTransform), typeof(VectorToImageConvertingTransformer), typeof(VectorToImageConvertingTransformer.Options), typeof(SignatureDataTransform),
    ImagePixelExtractingTransformer.UserName, "VectorToImageTransform", "VectorToImage")]

[assembly: LoadableClass(VectorToImageConvertingTransformer.Summary, typeof(IDataTransform), typeof(VectorToImageConvertingTransformer), null, typeof(SignatureLoadDataTransform),
    VectorToImageConvertingTransformer.UserName, VectorToImageConvertingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(VectorToImageConvertingTransformer), null, typeof(SignatureLoadModel),
    VectorToImageConvertingTransformer.UserName, VectorToImageConvertingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(VectorToImageConvertingTransformer), null, typeof(SignatureLoadRowMapper),
    VectorToImageConvertingTransformer.UserName, VectorToImageConvertingTransformer.LoaderSignature)]

namespace Microsoft.ML.ImageAnalytics
{
    /// <summary>
    /// <see cref="ITransformer"/> produced by fitting the <see cref="IDataView"/> to an <see cref="VectorToImageConvertingEstimator" /> .
    /// </summary>
    /// <remarks>
    /// <seealso cref="ImageEstimatorsCatalog.ConvertToImage(TransformsCatalog, VectorToImageConvertingEstimator.ColumnInfo[])" />
    /// <seealso cref="ImageEstimatorsCatalog.ConvertToImage(TransformsCatalog, int, int, string, string, ImagePixelExtractingEstimator.ColorBits, bool, float, float)" />
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
                    ImageHeight != null || Offset != null || Scale != null || InterleaveArgb != null)
                {
                    return false;
                }
                return TryUnparseCore(sb);
            }
        }

        internal class Options: TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use alpha channel", ShortName = "alpha")]
            public bool ContainsAlpha = (Defaults.Colors & ImagePixelExtractingEstimator.ColorBits.Alpha) > 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use red channel", ShortName = "red")]
            public bool ContainsRed = (Defaults.Colors & ImagePixelExtractingEstimator.ColorBits.Red) > 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use green channel", ShortName = "green")]
            public bool ContainsGreen = (Defaults.Colors & ImagePixelExtractingEstimator.ColorBits.Green) > 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to use blue channel", ShortName = "blue")]
            public bool ContainsBlue = (Defaults.Colors & ImagePixelExtractingEstimator.ColorBits.Blue) > 0;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether to separate each channel or interleave in ARGB order", ShortName = "interleave")]
            public bool InterleaveArgb = Defaults.InterleaveArgb;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Width of the image", ShortName = "width")]
            public int ImageWidth;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Height of the image", ShortName = "height")]
            public int ImageHeight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Offset (pre-scale)")]
            public Single Offset = Defaults.Offset;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Scale factor")]
            public Single Scale = Defaults.Scale;
        }

        internal static class Defaults
        {
            public const ImagePixelExtractingEstimator.ColorBits Colors = ImagePixelExtractingEstimator.ColorBits.Rgb;
            public const bool InterleaveArgb = false;
            public const Single Offset = 0;
            public const Single Scale = 1;
        }

        internal const string Summary = "Converts vector array into image type.";
        internal const string UserName = "Vector To Image Transform";
        internal const string LoaderSignature = "VectorToImageConverter";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "VECTOIMG",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verWrittenCur: 0x00010003, // don't serialize sizeof(Single)
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010003,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(VectorToImageConvertingTransformer).Assembly.FullName);
        }

        private const string RegistrationName = "VectorToImageConverter";

        private readonly VectorToImageConvertingEstimator.ColumnInfo[] _columns;

        /// <summary>
        /// The columns passed to this <see cref="ITransformer"/>.
        /// </summary>
        public IReadOnlyCollection<VectorToImageConvertingEstimator.ColumnInfo> Columns => _columns.AsReadOnly();

        internal VectorToImageConvertingTransformer(IHostEnvironment env, params VectorToImageConvertingEstimator.ColumnInfo[] columns)
            :base(Contracts.CheckRef(env, nameof(env)).Register(RegistrationName), GetColumnPairs(columns))
        {
            Host.AssertNonEmpty(columns);

            _columns = columns.ToArray();
        }

        internal VectorToImageConvertingTransformer(IHostEnvironment env, string outputColumnName, string inputColumnName, int imageHeight, int imageWidth, ImagePixelExtractingEstimator.ColorBits colors, bool interleave, float scale, float offset)
            : this(env, new VectorToImageConvertingEstimator.ColumnInfo(outputColumnName, inputColumnName, imageHeight, imageWidth, colors, interleave, scale, offset))
        {
        }

        // Constructor corresponding to SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options args, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckValue(input, nameof(input));

            env.CheckValue(args.Columns, nameof(args.Columns));

            var columns = new VectorToImageConvertingEstimator.ColumnInfo[args.Columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                var item = args.Columns[i];
                columns[i] = new VectorToImageConvertingEstimator.ColumnInfo(item, args);
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
            //   ColumnInfo

            _columns = new VectorToImageConvertingEstimator.ColumnInfo[ColumnPairs.Length];
            for (int i = 0; i < _columns.Length; i++)
                _columns[i] = new VectorToImageConvertingEstimator.ColumnInfo(ColumnPairs[i].outputColumnName, ColumnPairs[i].inputColumnName, ctx);
        }

        private static VectorToImageConvertingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

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

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(VectorToImageConvertingEstimator.ColumnInfo[] columns)
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

            if (vectorType.GetValueCount() != _columns[col].Height * _columns[col].Width * _columns[col].Planes)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColName, new VectorType(vectorType.ItemType, _columns[col].Height, _columns[col].Width, _columns[col].Planes).ToString(), vectorType.ToString());
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
                bool needScale = ex.Offset != 0 || ex.Scale != 1;
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
                VectorToImageConvertingEstimator.ColumnInfo ex, bool needScale) where TValue : IConvertible
            {
                Contracts.Assert(typeof(TValue) == srcType.RawType);
                var getSrc = RowCursorUtils.GetVecGetterAs<TValue>(srcType, input, ColMapNewToOld[iinfo]);
                var src = default(VBuffer<TValue>);
                int width = ex.Width;
                int height = ex.Height;
                float offset = ex.Offset;
                float scale = ex.Scale;

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

                        for (int y = 0; y < height; ++y)
                            for (int x = 0; x < width; x++)
                            {
                                float red = 0;
                                float green = 0;
                                float blue = 0;
                                float alpha = 0;
                                if (ex.Interleave)
                                {
                                    if (ex.Alpha)
                                        alpha = Convert.ToSingle(values[position++]);
                                    if (ex.Red)
                                        red = Convert.ToSingle(values[position++]);
                                    if (ex.Green)
                                        green = Convert.ToSingle(values[position++]);
                                    if (ex.Blue)
                                        blue = Convert.ToSingle(values[position++]);
                                }
                                else
                                {
                                    position = y * width + x;
                                    if (ex.Alpha)
                                    { alpha = Convert.ToSingle(values[position]); position += cpix; }
                                    if (ex.Red)
                                    { red = Convert.ToSingle(values[position]); position += cpix; }
                                    if (ex.Green)
                                    { green = Convert.ToSingle(values[position]); position += cpix; }
                                    if (ex.Blue)
                                    { blue = Convert.ToSingle(values[position]); position += cpix; }
                                }
                                Color pixel;
                                if (!needScale)
                                    pixel = Color.FromArgb((int)alpha, (int)red, (int)green, (int)blue);
                                else
                                {
                                    pixel = Color.FromArgb(
                                        ex.Alpha ? (int)Math.Round((alpha - offset) * scale) : 0,
                                        (int)Math.Round((red - offset) * scale),
                                        (int)Math.Round((green - offset) * scale),
                                        (int)Math.Round((blue - offset) * scale));
                                }
                                dst.SetPixel(x, y, pixel);
                            }
                    };
            }

            private static ImageType[] ConstructTypes(VectorToImageConvertingEstimator.ColumnInfo[] columns)
            {
                return columns.Select(c => new ImageType(c.Height, c.Width)).ToArray();
            }
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> that converts vectors containing pixel representations of images in to<see cref="ImageType"/> representation.
    /// </summary>
    /// <remarks>
    /// Calling <see cref="IEstimator{TTransformer}.Fit(IDataView)"/> in this estimator, produces an <see cref="VectorToImageConvertingTransformer"/>.
    /// <seealso cref="ImageEstimatorsCatalog.ConvertToImage(TransformsCatalog, ColumnInfo[])" />
    /// <seealso cref="ImageEstimatorsCatalog.ConvertToImage(TransformsCatalog, int, int, string, string, ImagePixelExtractingEstimator.ColorBits, bool, float, float)" />
    /// <seealso cref="ImageEstimatorsCatalog"/>
    /// </remarks>
    public sealed class VectorToImageConvertingEstimator : TrivialEstimator<VectorToImageConvertingTransformer>
    {
        /// <summary>
        /// Describes how the transformer handles one image pixel extraction column pair.
        /// </summary>
        public sealed class ColumnInfo
        {
            /// <summary>Name of the column resulting from the transformation of <see cref="InputColumnName"/>.</summary>
            public readonly string Name;

            /// <summary> Name of column to transform.</summary>
            public readonly string InputColumnName;

            public readonly ImagePixelExtractingEstimator.ColorBits Colors;
            public readonly byte Planes;

            public readonly int Width;
            public readonly int Height;
            public readonly Single Offset;
            public readonly Single Scale;
            public readonly bool Interleave;

            public bool Alpha => (Colors & ImagePixelExtractingEstimator.ColorBits.Alpha) != 0;
            public bool Red => (Colors & ImagePixelExtractingEstimator.ColorBits.Red) != 0;
            public bool Green => (Colors & ImagePixelExtractingEstimator.ColorBits.Green) != 0;
            public bool Blue => (Colors & ImagePixelExtractingEstimator.ColorBits.Blue) != 0;

            internal ColumnInfo(VectorToImageConvertingTransformer.Column item, VectorToImageConvertingTransformer.Options args)
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

                Interleave = item.InterleaveArgb ?? args.InterleaveArgb;

                Width = item.ImageWidth ?? args.ImageWidth;
                Height = item.ImageHeight ?? args.ImageHeight;
                Offset = item.Offset ?? args.Offset;
                Scale = item.Scale ?? args.Scale;
                Contracts.CheckUserArg(FloatUtils.IsFinite(Offset), nameof(item.Offset));
                Contracts.CheckUserArg(FloatUtils.IsFiniteNonZero(Scale), nameof(item.Scale));
            }

            internal ColumnInfo(string outputColumnName, string inputColumnName, ModelLoadContext ctx)
            {
                Contracts.AssertNonEmpty(outputColumnName);
                Contracts.AssertNonEmpty(inputColumnName);
                Contracts.AssertValue(ctx);

                Name = outputColumnName;
                InputColumnName = inputColumnName;

                // *** Binary format ***
                // byte: colors
                // int: widht
                // int: height
                // Float: offset
                // Float: scale
                // byte: separateChannels
                Colors = (ImagePixelExtractingEstimator.ColorBits)ctx.Reader.ReadByte();
                Contracts.CheckDecode(Colors != 0);
                Contracts.CheckDecode((Colors & ImagePixelExtractingEstimator.ColorBits.All) == Colors);

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

            public ColumnInfo(string outputColumnName, string inputColumnName,
                int imageHeight, int imageWidth, ImagePixelExtractingEstimator.ColorBits colors, bool interleave, float scale, float offset)
            {
                Contracts.CheckNonWhiteSpace(outputColumnName, nameof(InputColumnName));

                Name = outputColumnName;
                InputColumnName = inputColumnName ?? outputColumnName;
                Colors = colors;
                if ((byte)(Colors & ImagePixelExtractingEstimator.ColorBits.Alpha) > 0)
                    Planes++;
                if ((byte)(Colors & ImagePixelExtractingEstimator.ColorBits.Red) > 0)
                    Planes++;
                if ((byte)(Colors & ImagePixelExtractingEstimator.ColorBits.Green) > 0)
                    Planes++;
                if ((byte)(Colors & ImagePixelExtractingEstimator.ColorBits.Blue) > 0)
                    Planes++;
                Contracts.CheckParam(Planes > 0, nameof(colors), "Need to use at least one color plane");

                Interleave = interleave;

                Contracts.CheckParam(imageWidth > 0, nameof(imageWidth), "Image width must be greater than zero");
                Contracts.CheckParam(imageHeight > 0, nameof(imageHeight), "Image height must be greater than zero");
                Contracts.CheckParam(FloatUtils.IsFinite(offset), nameof(offset));
                Contracts.CheckParam(FloatUtils.IsFiniteNonZero(scale), nameof(scale));
                Width = imageWidth;
                Height = imageHeight;
                Offset = offset;
                Scale = scale;
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
                // byte: convert
                // Float: offset
                // Float: scale
                // byte: separateChannels
                Contracts.Assert(Colors != 0);
                Contracts.Assert((Colors & ImagePixelExtractingEstimator.ColorBits.All) == Colors);
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

        ///<summary>
        /// Convert pixels values into an image.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageHeight">The height of the image</param>
        /// <param name="imageWidth">The width of the image</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>. Null means <paramref name="inputColumnName"/> is replaced.</param>
        /// <param name="inputColumnName">Name of the input column.</param>
        /// <param name="colors">What colors to extract.</param>
        /// <param name="interleave">Whether to interleave the pixels, meaning keep them in the `RGB RGB` order, or leave them in the plannar form: of all red pixels,
        /// than all green, than all blue.</param>
        /// <param name="scale">Scale color pixel value by this amount.</param>
        /// <param name="offset">Offset color pixel value by this amount.</param>
        [BestFriend]
        internal VectorToImageConvertingEstimator(IHostEnvironment env,
            int imageHeight,
            int imageWidth,
            string outputColumnName,
            string inputColumnName = null,
            ImagePixelExtractingEstimator.ColorBits colors = ImagePixelExtractingEstimator.ColorBits.Rgb,
            bool interleave = false,
            float scale = 1,
            float offset = 0)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(VectorToImageConvertingEstimator)), new VectorToImageConvertingTransformer(env, outputColumnName, inputColumnName, imageHeight, imageWidth, colors, interleave, scale, offset))
        {
        }

        ///<summary>
        /// Extract pixels values from image and produce array of values.
        ///</summary>
        /// <param name="env">The host environment.</param>
        /// <param name="columns">Describes the parameters of pixel extraction for each column pair.</param>
        internal VectorToImageConvertingEstimator(IHostEnvironment env, params ColumnInfo[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(VectorToImageConvertingEstimator)), new VectorToImageConvertingTransformer(env, columns))
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

                var itemType = new ImageType(colInfo.Height, colInfo.Width);
                result[colInfo.Name] = new SchemaShape.Column(colInfo.Name, SchemaShape.Column.VectorKind.Scalar, itemType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}

