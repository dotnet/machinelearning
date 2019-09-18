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

[assembly: LoadableClass(SealEncryptionTransformer.Summary, typeof(IDataTransform), typeof(SealEncryptionTransformer),
    typeof(SealEncryptionTransformer.Options), typeof(SignatureDataTransform), SealEncryptionTransformer.UserName, SealEncryptionTransformer.ShortName)]

[assembly: LoadableClass(SealEncryptionTransformer.Summary, typeof(IDataTransform), typeof(SealEncryptionTransformer),
    null, typeof(SignatureLoadDataTransform), SealEncryptionTransformer.UserName, SealEncryptionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(SealEncryptionTransformer), null, typeof(SignatureLoadModel),
    SealEncryptionTransformer.UserName, SealEncryptionTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SealEncryptionTransformer), null, typeof(SignatureLoadRowMapper),
    SealEncryptionTransformer.UserName, SealEncryptionTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(SealEncryptionTransformer))]

namespace Microsoft.ML.SEAL
{
    public sealed class SealEncryptionTransformer : RowToRowTransformerBase
    {
        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "SEAL Encryption Parameters", SortOrder = 0)]
            public EncryptionParameters SealEncryptionParameters;

            [Argument(ArgumentType.Required, HelpText = "SEAL Public Key File Name", SortOrder = 1)]
            public string SealPublicKeyFileName;

            [Argument(ArgumentType.Required, HelpText = "Name of the input column.", SortOrder = 1)]
            public string InputColumn;

            [Argument(ArgumentType.Required, HelpText = "Name of the output column.", SortOrder = 2)]
            public string OutputColumn;
        }

        internal const string Summary = "Transforms the data using SEAL encryption.";
        internal const string UserName = "SealEncryptionTransform";
        internal const string ShortName = "SETransform";
        internal const string LoaderSignature = "SealTransform";

        internal readonly EncryptionParameters SealEncryptionParameters;
        internal readonly string SealPublicKeyFileName;
        internal readonly CKKSEncoder SealCkksEncoder;
        internal readonly Encryptor SealEncryptor;
        internal readonly string InputColumnName;
        internal readonly string OutputColumnName;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SEALENCR",
                verWrittenCur: 0x00010000,
                verReadableCur: 0x00010000,
                verWeCanReadBack: 0x00010000,
                loaderSignature: LoaderSignature,
            loaderAssemblyName: typeof(SealEncryptionTransformer).Assembly.FullName);
        }

        // Factory method for SignatureDataTransform
        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            return new SealEncryptionTransformer(env, options).MakeDataTransform(input);
        }

        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadModel.
        private static SealEncryptionTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            ctx.CheckVersionInfo(GetVersionInfo());
            var parameters = EncryptionParameters.Load(ctx.Reader.BaseStream);
            var sealPublicKeyFilePath = ctx.LoadString();
            var inputColumnName = ctx.LoadString();
            var outputColumnName = ctx.LoadString();
            return new SealEncryptionTransformer(env, new Options()
            {
                SealEncryptionParameters = parameters,
                SealPublicKeyFileName = sealPublicKeyFilePath,
                InputColumn = inputColumnName,
                OutputColumn = outputColumnName
            });
        }

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private SealEncryptionTransformer(IHostEnvironment env, Options options) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(SealEncryptionTransformer)))
        {
            Host.CheckValue(options, nameof(options));
            Host.CheckNonWhiteSpace(options.InputColumn, nameof(options.InputColumn));
            Host.CheckNonWhiteSpace(options.OutputColumn, nameof(options.OutputColumn));

            SealEncryptionParameters = options.SealEncryptionParameters;
            var context = SEALContext.Create(SealEncryptionParameters);
            SealCkksEncoder = new CKKSEncoder(context);
            SealPublicKeyFileName = options.SealPublicKeyFileName;

            using (FileStream fs = File.OpenRead(SealPublicKeyFileName))
            {
                Microsoft.Research.SEAL.PublicKey pk = new Microsoft.Research.SEAL.PublicKey();
                pk.Load(context, fs);
                SealEncryptor = new Encryptor(context, pk);
            }

            InputColumnName = options.InputColumn;
            OutputColumnName = options.OutputColumn;
        }

        internal SealEncryptionTransformer(IHostEnvironment env, EncryptionParameters sealEncryptionParameters, string sealPublicKeyFilePath, string outputColumnName, string inputColumnName = null)
            : this(env, new Options()
            {
                SealEncryptionParameters = sealEncryptionParameters,
                SealPublicKeyFileName = sealPublicKeyFilePath,
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
            EncryptionParameters.Save(SealEncryptionParameters, ctx.Writer.BaseStream);
            ctx.SaveString(SealPublicKeyFileName);
            ctx.SaveString(InputColumnName);
            ctx.SaveString(OutputColumnName);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : MapperBase
        {
            private readonly SealEncryptionTransformer _parent;
            private readonly int _featureColIndex;

            public Mapper(SealEncryptionTransformer parent, DataViewSchema inputSchema) :
                base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                _parent = parent;
                _featureColIndex = inputSchema.GetColumnOrNull(_parent.InputColumnName)?.Index ?? -1;
                var errorMsg = string.Format("The data to encrypt contains no '{0}' column", _parent.InputColumnName);
                parent.Host.Check(_featureColIndex > 0, errorMsg);
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return new[]
                {
                    new DataViewSchema.DetachedColumn(_parent.OutputColumnName, CiphertextDataViewType.Instance)
                };
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                disposer = null;
                Host.AssertValue(input);
                Host.Assert(input.IsColumnActive(input.Schema[_featureColIndex]));
                var getFeature = input.GetGetter<VBuffer<float>>(input.Schema[_featureColIndex]);
                ValueGetter<VBuffer<Ciphertext>> encrypt = (ref VBuffer<Ciphertext> dst) =>
                {
                    VBuffer<float> features = default;
                    getFeature(ref features);
                    var denseFeatures = new VBuffer<float>();
                    features.CopyToDense(ref denseFeatures);
                    var ciphers = new List<Ciphertext>();

                    for (ulong i = 0; i < (ulong)denseFeatures.Length; i += _parent.SealCkksEncoder.SlotCount)
                    {
                        var listVals = new List<double>();
                        for (ulong j = 0; (j < _parent.SealCkksEncoder.SlotCount) && ((i + j) < (ulong)denseFeatures.Length); ++j)
                            listVals.Add(denseFeatures.GetItemOrDefault((int)(i + j)));
                        var plain = new Plaintext();
                        var scale = Math.Pow(2.0, 60);
                        _parent.SealCkksEncoder.Encode(listVals, scale, plain);
                        var encrypted = new Ciphertext();
                        _parent.SealEncryptor.Encrypt(plain, encrypted);
                        ciphers.Add(encrypted);
                    }

                    dst = new VBuffer<Ciphertext>(ciphers.Count, ciphers.ToArray());
                };

                return encrypt;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput) => col => col == _featureColIndex;

            private protected override void SaveModel(ModelSaveContext ctx) => _parent.SaveModel(ctx);
        }
    }

    public sealed class SealEncryptionEstimator : IEstimator<SealEncryptionTransformer>
    {
        private readonly IHostEnvironment _env;
        private readonly EncryptionParameters _sealEncryptionParameters;
        private readonly string _sealPublicKeyFilePath;
        private readonly string _inputColumnName;
        private readonly string _outputColumnName;

        public SealEncryptionEstimator(IHostEnvironment env, ulong polyModDegree, string sealPublicKeyFilePath, string outputColumnName, string inputColumnName = null)
        {
            _env = env;
            _sealEncryptionParameters = new EncryptionParameters(SchemeType.CKKS);
            _sealEncryptionParameters.PolyModulusDegree = polyModDegree;
            _sealEncryptionParameters.CoeffModulus = DefaultParams.CoeffModulus128(polyModulusDegree: polyModDegree);
            _sealPublicKeyFilePath = sealPublicKeyFilePath;
            _inputColumnName = inputColumnName ?? outputColumnName;
            _outputColumnName = outputColumnName;
        }

        SchemaShape IEstimator<SealEncryptionTransformer>.GetOutputSchema(SchemaShape inputSchema)
        {
            var columns = inputSchema.GetEnumerator();
            var columnsList = new List<SchemaShape.Column>();
            while (columns.MoveNext()) columnsList.Add(columns.Current);
            var encColumnSchema = new SchemaShape.Column(_inputColumnName, SchemaShape.Column.VectorKind.Scalar, CiphertextDataViewType.Instance, false);
            columnsList.Add(encColumnSchema);
            SchemaShape outputShape = new SchemaShape(columnsList);
            return outputShape;
        }

        SealEncryptionTransformer IEstimator<SealEncryptionTransformer>.Fit(IDataView input)
        {
            return new SealEncryptionTransformer(_env, _sealEncryptionParameters, _sealPublicKeyFilePath, _outputColumnName, _inputColumnName);
        }
    }
}
