// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.SEAL;
using Microsoft.ML.Transforms;
using Microsoft.Research.SEAL;

[assembly: LoadableClass(SealTransformer.Summary, typeof(IDataTransform), typeof(SealTransformer),
    typeof(SealTransformer.Options), typeof(SignatureDataTransform), SealTransformer.UserName, SealTransformer.ShortName)]

[assembly: LoadableClass(SealTransformer.Summary, typeof(IDataTransform), typeof(SealTransformer),
    null, typeof(SignatureLoadDataTransform), SealTransformer.UserName, SealTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(SealTransformer), null, typeof(SignatureLoadModel),
    SealTransformer.UserName, SealTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SealTransformer), null, typeof(SignatureLoadRowMapper),
    SealTransformer.UserName, SealTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(SealTransformer))]

namespace Microsoft.ML.SEAL
{
    /// <summary>
    /// This is a transformer that will encrypt or decrypt a column passing through it.
    /// </summary>
    public class SealTransformer : RowToRowTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "Encrypt or Decrypt Mode", SortOrder = 0)]
            public bool Encrypt;

            [Argument(ArgumentType.Required, HelpText = "Scale for ", SortOrder = 1)]
            public double Scale;

            [Argument(ArgumentType.Required, HelpText = "SEAL Encryption Parameters", SortOrder = 2)]
            public EncryptionParameters SealEncryptionParameters;

            [Argument(ArgumentType.Required, HelpText = "SEAL Public Key File Name", SortOrder = 3)]
            public string SealKeyFileName;

            [Argument(ArgumentType.Required, HelpText = "Name of the input column.", SortOrder = 4)]
            public string InputColumn;

            [Argument(ArgumentType.Required, HelpText = "Name of the output column.", SortOrder = 5)]
            public string OutputColumn;
        }

        internal const string Summary = "Transforms the data using SEAL.";
        internal const string UserName = "SealTransform";
        internal const string ShortName = "STransform";
        internal const string LoaderSignature = "SealTransform";

        internal readonly bool Encrypt;
        internal readonly double Scale;
        internal readonly EncryptionParameters SealEncryptionParameters;
        internal readonly string SealKeyFileName;
        internal readonly CKKSEncoder SealCkksEncoder;
        internal readonly Encryptor SealEncryptor;
        internal readonly Decryptor SealDecryptor;
        internal readonly string InputColumnName;
        internal readonly string OutputColumnName;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SEALTRAN",
                verWrittenCur: 0x00010000,
                verReadableCur: 0x00010000,
                verWeCanReadBack: 0x00010000,
                loaderSignature: LoaderSignature,
            loaderAssemblyName: typeof(SealTransformer).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            return new SealTransformer(env, options).MakeDataTransform(input);
        }

        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadModel.
        private static SealTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            ctx.CheckVersionInfo(GetVersionInfo());
            var encrypt = ctx.LoadString();
            var scale = Convert.ToDouble(ctx.LoadString());
            var parameters = new EncryptionParameters();
            parameters.Load(ctx.Reader.BaseStream);
            var sealKeyFilePath = ctx.LoadString();
            var inputColumnName = ctx.LoadString();
            var outputColumnName = ctx.LoadString();
            return new SealTransformer(env, new Options()
            {
                Encrypt = encrypt == "Encrypt",
                Scale = scale,
                SealEncryptionParameters = parameters,
                SealKeyFileName = sealKeyFilePath,
                InputColumn = inputColumnName,
                OutputColumn = outputColumnName
            });
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private SealTransformer(IHostEnvironment env, Options options) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(SealTransformer)))
        {
            Host.CheckValue(options, nameof(options));
            Host.CheckNonWhiteSpace(options.InputColumn, nameof(options.InputColumn));
            Host.CheckNonWhiteSpace(options.OutputColumn, nameof(options.OutputColumn));

            Encrypt = options.Encrypt;
            Scale = options.Scale;
            SealEncryptionParameters = options.SealEncryptionParameters;
            var context = new SEALContext(SealEncryptionParameters);
            SealCkksEncoder = new CKKSEncoder(context);
            SealKeyFileName = options.SealKeyFileName;

            using (FileStream fs = File.OpenRead(SealKeyFileName))
            {
                if (Encrypt)
                {
                    Microsoft.Research.SEAL.PublicKey pk = new Microsoft.Research.SEAL.PublicKey();
                    pk.Load(context, fs);
                    SealEncryptor = new Encryptor(context, pk);
                }
                else
                {
                    Microsoft.Research.SEAL.SecretKey sk = new Microsoft.Research.SEAL.SecretKey();
                    sk.Load(context, fs);
                    SealDecryptor = new Decryptor(context, sk);
                }
            }

            InputColumnName = options.InputColumn;
            OutputColumnName = options.OutputColumn;
        }

        internal SealTransformer(IHostEnvironment env, bool encrypted, double scale, EncryptionParameters sealEncryptionParameters, string sealKeyFilePath, string outputColumnName, string inputColumnName = null)
            : this(env, new Options()
            {
                Encrypt = encrypted,
                Scale = scale,
                SealEncryptionParameters = sealEncryptionParameters,
                SealKeyFileName = sealKeyFilePath,
                InputColumn = inputColumnName ?? outputColumnName,
                OutputColumn = outputColumnName
            })
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            if (Encrypt) ctx.SaveString("Encrypt");
            else ctx.SaveString("Decrypt");
            ctx.SaveString(Convert.ToString(Scale));
            SealEncryptionParameters.Save(ctx.Writer.BaseStream);
            ctx.SaveString(SealKeyFileName);
            ctx.SaveString(InputColumnName);
            ctx.SaveString(OutputColumnName);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private readonly SealTransformer _parent;
            private readonly int _featureColIndex;

            public Mapper(SealTransformer parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _featureColIndex = inputSchema.GetColumnOrNull(_parent.InputColumnName)?.Index ?? -1;
                var errorMsg = string.Format("The data to encrypt contains no '{0}' column", _parent.InputColumnName);
                parent.Host.Check(_featureColIndex >= 0, errorMsg);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return new[]
                {
                    new DataViewSchema.DetachedColumn(_parent.OutputColumnName, new CiphertextDataViewType())
                };
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);
                Host.Assert(input.IsColumnActive(input.Schema[_featureColIndex]));

                if(_parent.Encrypt)
                {
                    var getFeature = input.GetGetter<VBuffer<double>>(input.Schema[_featureColIndex]);
                    ValueGetter<VBuffer<Ciphertext>> encrypt = (ref VBuffer<Ciphertext> dst) =>
                    {
                        VBuffer<double> features = default;
                        getFeature(ref features);
                        var denseFeatures = new VBuffer<double>();
                        features.CopyToDense(ref denseFeatures);
                        var ciphers = new List<Ciphertext>();

                        for (ulong i = 0; i < (ulong)denseFeatures.Length; i += _parent.SealCkksEncoder.SlotCount)
                        {
                            var listVals = new List<double>();
                            for (ulong j = 0; (j < _parent.SealCkksEncoder.SlotCount) && ((i + j) < (ulong)denseFeatures.Length); ++j)
                                listVals.Add(denseFeatures.GetItemOrDefault((int)(i + j)));
                            var plain = new Plaintext();
                            _parent.SealCkksEncoder.Encode(listVals, _parent.Scale, plain);
                            var encrypted = new Ciphertext();
                            _parent.SealEncryptor.Encrypt(plain, encrypted);
                            ciphers.Add(encrypted);
                        }

                        dst = new VBuffer<Ciphertext>(ciphers.Count, ciphers.ToArray());
                    };
                    return encrypt;
                }
                else
                {
                    var getFeature = input.GetGetter<VBuffer<Ciphertext>>(input.Schema[_featureColIndex]);
                    ValueGetter<VBuffer<double>> decrypt = (ref VBuffer<double> dst) =>
                    {
                        VBuffer<Ciphertext> ciphers = default;
                        getFeature(ref ciphers);
                        var decrypted = new List<double>();

                        foreach (var cipher in ciphers.DenseValues())
                        {
                            var plain = new Plaintext();
                            _parent.SealDecryptor.Decrypt(cipher, plain);
                            var listVals = new List<double>();
                            _parent.SealCkksEncoder.Decode(plain, listVals);
                            foreach (var f in listVals) decrypted.Add(f);
                        }
                    };
                    return decrypt;
                }
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput) => col => col == _featureColIndex;

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    /// <summary>
    /// This is a estimator that will encrypt or decrypt a column passing through it.
    /// </summary>
    public sealed class SealEstimator : IEstimator<SealTransformer>
    {
        private readonly IHostEnvironment _env;
        private readonly bool _encrypt;
        private readonly double _scale;
        private readonly EncryptionParameters _sealEncryptionParameters;
        private readonly string _sealKeyFilePath;
        private readonly string _inputColumnName;
        private readonly string _outputColumnName;

        /// <summary>
        /// Initializes a new instance of <see cref="SealEstimator"/>.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="encrypt">Whether the estimator should encrypt (true) or decrypt (false) the data.</param>
        /// <param name="scale">How much to scale the values.</param>
        /// <param name="polyModDegree">The polynomial modulus degree.</param>
        /// <param name="sealKeyFilePath">The path to the SEAL key file.</param>
        /// <param name="coeffModuli">The coefficient moduli needed to create the SEAL context.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.
        /// This column's data type will be the same as that of the input column.</param>
        /// <param name="inputColumnName">Name of the column to copy the data from.
        /// This estimator operates over any data type.</param>
        public SealEstimator(IHostEnvironment env, bool encrypt, double scale, ulong polyModDegree, string sealKeyFilePath, IEnumerable<SmallModulus> coeffModuli, string outputColumnName, string inputColumnName = null)
        {
            _env = env;
            _encrypt = encrypt;
            _scale = scale;
            _sealEncryptionParameters = new EncryptionParameters(SchemeType.CKKS);
            _sealEncryptionParameters.PolyModulusDegree = polyModDegree;
            _sealEncryptionParameters.CoeffModulus = coeffModuli;
            _sealKeyFilePath = sealKeyFilePath;
            _inputColumnName = inputColumnName ?? outputColumnName;
            _outputColumnName = outputColumnName;
        }

        SchemaShape IEstimator<SealTransformer>.GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.GetEnumerator();
            var columnsList = new List<SchemaShape.Column>();
            while (columns.MoveNext()) columnsList.Add(columns.Current);
            if (_encrypt) columnsList.Add(new SchemaShape.Column(_inputColumnName, SchemaShape.Column.VectorKind.Scalar, new CiphertextDataViewType(), false));
            else columnsList.Add(new SchemaShape.Column(_inputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Double, false));
            SchemaShape outputShape = new SchemaShape(columnsList);
            return outputShape;
        }

        /// <summary>
        /// Trains and returns a <see cref="SealTransformer"/>.
        /// </summary>
        public SealTransformer Fit(IDataView input)
        {
            return new SealTransformer(_env, _encrypt, _scale, _sealEncryptionParameters, _sealKeyFilePath, _outputColumnName, _inputColumnName);
        }
    }
}
