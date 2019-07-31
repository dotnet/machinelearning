using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Torch;
using Microsoft.ML.Transforms;
using TorchSharp.Tensor;

[assembly: LoadableClass(TorchScorerTransformer.Summary, typeof(IDataTransform), typeof(TorchScorerTransformer), typeof(TorchScorerEstimator.Options), typeof(SignatureDataTransform),
    TorchScorerTransformer.UserName)]

[assembly: LoadableClass(TorchScorerTransformer.Summary, typeof(IDataTransform), typeof(TorchScorerTransformer), null, typeof(SignatureLoadDataTransform),
    TorchScorerTransformer.UserName, TorchScorerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(TorchScorerTransformer), null, typeof(SignatureLoadModel),
    TorchScorerTransformer.UserName, TorchScorerTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TorchScorerTransformer), null, typeof(SignatureLoadRowMapper),
    TorchScorerTransformer.UserName, TorchScorerTransformer.LoaderSignature)]

namespace Microsoft.ML.Transforms
{
    public sealed class TorchScorerTransformer : RowToRowTransformerBase
    {
        private readonly string _savedModelPath;

        internal readonly string OutputColumnName;
        internal readonly string[] InputColumnNames;
        internal readonly long[][] InputShapes;
        internal readonly TorchModuleWrapper Module;

        internal const string Summary = "Transforms the data using a Torch model.";
        internal const string UserName = "TorchTransform";
        internal const string LoaderSignature = "TorchTransform";
        private const string _modelFileRepo = "TorchJITModule";

        private class OutputCache
        {
            public long Position;
            public Dictionary<string, TorchTensor> Outputs;
            public OutputCache()
            {
                Position = -1;
                Outputs = new Dictionary<string, TorchTensor>();
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TORCHSCO", // Torch scoring
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TorchScorerTransformer).Assembly.FullName);
        }

        internal TorchScorerTransformer(IHostEnvironment env, TorchModuleWrapper module, string outputColumnName, string[] inputColumnNames,
            long[][] inputShape, string savedModelPath)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchScorerTransformer)))
        {
            Host.CheckValue(module, nameof(module));
            Host.CheckNonWhiteSpace(outputColumnName, nameof(outputColumnName));
            Host.CheckNonWhiteSpace(savedModelPath, nameof(savedModelPath));
            Host.CheckValue(inputColumnNames, nameof(inputColumnNames));
            Host.Check(!inputColumnNames.Any(x => x == null), "Input column names cannot not be null.");
            Host.CheckValue(inputShape, nameof(inputShape));

            OutputColumnName = outputColumnName;
            InputColumnNames = inputColumnNames;
            InputShapes = inputShape;
            Module = module;
            _savedModelPath = savedModelPath;
        }

        // Factory method for SignatureLoadModel.
        internal TorchScorerTransformer(IHostEnvironment env, ModelLoadContext ctx)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchScorerTransformer)))
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // string output column name
            // int: number of input columns
            // for each input column
            //   string: input column name
            //   int: length of the inputshape array for this input column
            //   for each element in the inputshape
            //      long: element
            // stream: torch JIT module

            OutputColumnName = ctx.LoadNonEmptyString();

            var numInputs = ctx.Reader.ReadInt32();
            Host.CheckDecode(numInputs > 0);
            InputColumnNames = new string[numInputs];
            InputShapes = new long[numInputs][];
            for (int i = 0; i < InputColumnNames.Length; i++)
            {
                InputColumnNames[i] = ctx.LoadNonEmptyString();
                var inputShapeLength = ctx.Reader.ReadInt32();
                Host.CheckDecode(inputShapeLength > 0);
                InputShapes[i] = new long[inputShapeLength];
                for (int j = 0; j < InputShapes[i].Length; j++)
                {
                    InputShapes[i][j] = ctx.Reader.ReadInt64();
                    Host.CheckDecode(InputShapes[i][j] > 0);
                }
            }

            // Creates a temporary directory with a file containing the model. We can the load the model using the Torch::JIT::Module::Load() method that takes a path.
            // REVIEW: ideally we can load a module directly from a stream without the trick of a temp directory. This needs to be added to TorchSharp.
            var tempDirPath = Path.GetFullPath(Path.Combine(Path.GetTempPath(), nameof(TorchScorerTransformer) + "_" + Guid.NewGuid()));
            TorchUtils.CreateFolder(Host, tempDirPath);
            try
            {
                string fullFilePath = null;
                var load = ctx.TryLoadBinaryStream(_modelFileRepo, reader =>
                {
                    long fileLength = reader.ReadInt64();
                    fullFilePath = Path.Combine(tempDirPath, _modelFileRepo + ".bin");
                    using (var fs = new FileStream(fullFilePath, FileMode.Create, FileAccess.Write))
                    {
                        long actualRead = reader.BaseStream.CopyRange(fs, fileLength);
                        Host.Assert(actualRead == fileLength);
                    }
                });
                Host.CheckDecode(load);

                _savedModelPath = fullFilePath;
                Module = TorchUtils.LoadTorchModel(Host, _savedModelPath).Module;
            }
            catch (Exception)
            {
                Directory.Delete(tempDirPath, true);
                throw;
            }
            Host.CheckDecode(_savedModelPath != null);
            Host.CheckDecode(Module != null);
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // string output column name
            // int: number of input columns
            // for each input column
            //   string: input column name
            //   int: length of the inputshape array for this input column
            //   for each element in the inputshape
            //      long: element
            // stream: torch JIT module

            Host.AssertNonWhiteSpace(OutputColumnName);
            ctx.SaveNonEmptyString(OutputColumnName);

            Host.AssertNonEmpty(InputColumnNames);
            Host.AssertNonEmpty(InputShapes);
            Host.Assert(InputColumnNames.Length == InputShapes.Length);
            ctx.Writer.Write(InputColumnNames.Length);
            for (int i = 0; i < InputColumnNames.Length; i++)
            {
                ctx.SaveNonEmptyString(InputColumnNames[i]);
                Host.AssertNonEmpty(InputShapes[i]);
                ctx.Writer.Write(InputShapes[i].Length);
                foreach (var dim in InputShapes[i])
                    ctx.Writer.Write(dim);
            }

            // REVIEW: The below requires the model file not to have been moved.
            // A better alternative would be to use the Torch::JIT::Module::Save() method that uses a stream.
            Host.AssertNonWhiteSpace(_savedModelPath);
            ctx.SaveBinaryStream(_modelFileRepo, writer =>
            {
                using (var fs = new FileStream(_savedModelPath, FileMode.Open))
                {
                    long fileLength = fs.Length;
                    writer.Write(fileLength);
                    long actualWritten = fs.CopyRange(writer.BaseStream, fileLength);
                    Host.Assert(actualWritten == fileLength);
                }
            });
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, TorchScorerEstimator.Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));
            env.CheckValue(options.InputColumnNames, nameof(options.InputColumnNames));
            env.CheckValue(options.OutputColumnName, nameof(options.OutputColumnName));

            return new TorchScorerEstimator(env, options).Fit(input).MakeDataTransform(input);
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => new TorchScorerTransformer(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => new TorchScorerTransformer(env, ctx).MakeRowMapper(inputSchema);

        private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema)
            => new Mapper(this, inputSchema);

        ~TorchScorerTransformer()
        {
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                Module.Dispose();
            }
        }

        internal static ITensorValueGetter[] GetTensorValueGetters(DataViewRow input,
            int[] inputColIndices,
            bool[] isInputVector,
            Type[] inputTypes,
            long[][] inputShapes)
        {
            var srcTensorGetters = new ITensorValueGetter[inputColIndices.Length];
            for (int i = 0; i < inputColIndices.Length; i++)
            {
                int colIndex = inputColIndices[i];
                srcTensorGetters[i] = CreateTensorValueGetter(input, inputTypes[i], isInputVector[i], colIndex, inputShapes[i]);
            }
            return srcTensorGetters;
        }

        internal static ITensorValueGetter CreateTensorValueGetter<T>(DataViewRow input, bool isVector, int colIndex, long[] shape)
        {
            if (isVector)
                return new TensorValueGetterVec<T>(input, colIndex, shape);
            return new TensorValueGetter<T>(input, colIndex, shape);
        }

        internal static ITensorValueGetter CreateTensorValueGetter(DataViewRow input, Type type, bool isVector, int colIndex, long[] shape)
        {
            Contracts.AssertValue(type);
            return Utils.MarshalInvoke(CreateTensorValueGetter<int>, type, input, isVector, colIndex, shape);
        }

        internal interface ITensorValueGetter
        {
            void BufferTrainingData();
            TorchTensor GetTensor();

            TorchTensor GetBufferedBatchTensor();
        }

        internal class TensorValueGetter<T> : ITensorValueGetter
        {
            private readonly ValueGetter<T> _srcgetter;
            private readonly long[] _shape;
            private readonly T[] _bufferedData;
            private int _position;

            public TensorValueGetter(DataViewRow input, int colIndex, long[] shape)
            {
                _srcgetter = input.GetGetter<T>(input.Schema[colIndex]);
                _shape = shape;
                _position = 0;
                _bufferedData = new T[shape.Aggregate(1, (x, y) => x * (int)y)];
            }

            public TorchTensor GetTensor()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                return scalar.ToTorchTensor();
            }

            public void BufferTrainingData()
            {
                var scalar = default(T);
                _srcgetter(ref scalar);
                _bufferedData[_position++] = scalar;
            }

            public TorchTensor GetBufferedBatchTensor()
            {
                var tensor = _bufferedData.ToTorchTensor(_shape);
                _position = 0;
                return tensor;
            }
        }

        internal class TensorValueGetterVec<T> : ITensorValueGetter
        {
            private readonly ValueGetter<VBuffer<T>> _srcgetter;
            private VBuffer<T> _vBuffer;
            private long[] _shape;
            private T[] _denseData;
            private readonly T[] _bufferedData;
            private int _position;

            public TensorValueGetterVec(DataViewRow input, int colIndex, long[] shape)
            {
                _srcgetter = input.GetGetter<VBuffer<T>>(input.Schema[colIndex]);
                _shape = shape;
                _vBuffer = default;
                _denseData = default;
                _bufferedData = new T[shape.Aggregate(1, (x, y) => x * (int)y)];
            }

            public TorchTensor GetTensor()
            {
                _srcgetter(ref _vBuffer);

                // _denseData.Length can be greater than _vBuffer.Length sometime after
                // Utils.EnsureSize is exectued. Use _vBuffer.Length to access the elements in _denseData.
                // This is done to reduce memory allocation every time tensor is created.
                Utils.EnsureSize(ref _denseData, _vBuffer.Length, keepOld: false);
                _vBuffer.CopyTo(_denseData);

                return _denseData.ToTorchTensor(_shape);
            }

            public void BufferTrainingData()
            {
                _srcgetter(ref _vBuffer);
                _vBuffer.CopyTo(_bufferedData, _position);
                _position += _vBuffer.Length;
            }

            public TorchTensor GetBufferedBatchTensor()
            {
                var tensor = _bufferedData.ToTorchTensor(_shape);
                _position = 0;
                return tensor;
            }
        }

        private sealed class Mapper : MapperBase
        {
            private readonly TorchScorerTransformer _parent;
            private readonly int[] _inputColIndices;
            private readonly bool[] _isInputVector;
            private readonly Type[] _rawType;

            public Mapper(TorchScorerTransformer parent, DataViewSchema inputSchema) :
                   base(Contracts.CheckRef(parent, nameof(parent)).Host.Register(nameof(Mapper)), inputSchema, parent)
            {
                Host.CheckValue(parent, nameof(parent));
                _parent = parent;
                _inputColIndices = new int[_parent.InputColumnNames.Length];
                _isInputVector = new bool[_parent.InputColumnNames.Length];
                _rawType = new Type[_parent.InputColumnNames.Length];

                for (int i = 0; i < _parent.InputColumnNames.Length; i++)
                {
                    // Check presence of input columns.
                    if (!inputSchema.TryGetColumnIndex(_parent.InputColumnNames[i], out _inputColIndices[i]))
                        throw Host.ExceptSchemaMismatch(nameof(inputSchema), "source", _parent.InputColumnNames[i]);

                    // Check input column types
                    var type = inputSchema[_inputColIndices[i]].Type;
                    _rawType[i] = type.RawType;

                    if (type is VectorDataViewType vectorType)
                    {
                        _rawType[i] = vectorType.ItemType.RawType;
                        _isInputVector[i] = true;
                        var colTypeDims = vectorType.Dimensions.Select(dim => (long)dim).ToArray();
                        var colShapeLength = colTypeDims.Length;

                        if (colShapeLength != _parent.InputShapes[i].Length)
                            throw Host.Except($"Input shape mismatch: Input Column '{_parent.InputColumnNames[i]}' vector shape length {colShapeLength} does not match expected {_parent.InputShapes[i].Length}.");

                        if (_parent.InputShapes[i].Where(x => x == -1).Count() > 1)
                            throw Host.Except($"Shape with mode than 1 undefined dimension: Input Column '{_parent.InputColumnNames[i]}'.");

                        for (int j = 0; j < colShapeLength; j++)
                        {
                            if (colTypeDims[j] != _parent.InputShapes[i][j] && _parent.InputShapes[i][j] != -1)
                                throw Host.Except($"Input shape mismatch: Input Column '{_parent.InputColumnNames[i]}' dimension {j} of size {colTypeDims[j]} does not match expected size {_parent.InputShapes[j]}.");
                            if (_parent.InputShapes[i][j] == -1)
                                _parent.InputShapes[i][j] = colTypeDims[j];
                        }
                    }
                    else
                    {
                        _parent.InputShapes[i] = new long[] { 1 };
                    }
                }
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
                return new DataViewSchema.DetachedColumn[]
                {
                    // REVIEW: we need to deal with how we get the output size for the vector.
                    new DataViewSchema.DetachedColumn(_parent.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single), null)
                };
            }

            private void UpdateCacheIfNeeded(long position, ITensorValueGetter[] inputGetters, OutputCache outputCache)
            {
                if (outputCache.Position != position)
                {
                    var inputTensors = new TorchTensor[_inputColIndices.Length];

                    for (int colIndex = 0; colIndex < _inputColIndices.Length; colIndex++)
                    {
                        inputTensors[colIndex] = inputGetters[colIndex].GetTensor();
                    }

                    TorchTensor result = _parent.Module.Forward(inputTensors);

                    outputCache.Outputs[_parent.OutputColumnName] = result;
                    outputCache.Position = position;

                    foreach (var tensor in inputTensors)
                        tensor.Dispose();
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Host.AssertValue(input);
                var outputCache = new OutputCache();
                disposer = () => outputCache.Outputs.Values.ToList().ForEach(x => x.Dispose());
                var srcTensorGetters = GetTensorValueGetters(input, _inputColIndices, _isInputVector, _rawType, _parent.InputShapes);
                return Utils.MarshalInvoke(MakeGetter<int>, typeof(float), input, iinfo, srcTensorGetters, outputCache);
            }

            private Delegate MakeGetter<T>(DataViewRow input, int iinfo, ITensorValueGetter[] srcTensorGetters,  OutputCache outputCache)
            {
                Host.AssertValue(input);
                ValueGetter<VBuffer<T>> valuegetter = (ref VBuffer<T> dst) =>
                {
                    UpdateCacheIfNeeded(input.Position, srcTensorGetters, outputCache);

                    var outputTensor = outputCache.Outputs[_parent.OutputColumnName];
                    var resultSize = outputTensor.NumberOfElements;
                    var editor = VBufferEditor.Create(ref dst, (int)resultSize);
                    var data = outputTensor.Data<T>();

                    data.CopyTo(editor.Values);
                    dst = editor.Commit();
                };
                return valuegetter;
            }

            private protected override Func<int, bool> GetDependenciesCore(Func<int, bool> activeOutput)
            {
                // Range goes from 0 to 1 because there is only one output column.
                return col => Enumerable.Range(0, 1).Any(i => activeOutput(i)) && _inputColIndices.Any(i => i == col);
            }

            private protected override void SaveModel(ModelSaveContext ctx)
                => _parent.SaveModel(ctx);
        }
    }

    /// <summary>
    /// Estimator for the <see cref="TorchScorerTransformer"/>.
    /// </summary>
    public sealed class TorchScorerEstimator : TrivialEstimator<TorchScorerTransformer>
    {
        /// <summary>
        /// The options for the <see cref="TorchScorerTransformer"/>.
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// The name of the output column of the transformation.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the output column", Name = "Name", ShortName = "name", SortOrder = 2)]
            public string OutputColumnName;

            /// <summary>
            /// The names of the columns containing the inputs for the model. If <see langword="null"/>, this defaults to <see cref="OutputColumnName"/>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The names of the input columns", Name = "Sources", ShortName = "src", SortOrder = 1)]
            public string[] InputColumnNames;

            /// <summary>
            /// The shape of the model input.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The shape of the input tensor", ShortName = "shape", SortOrder = 0)]
            public long[][] InputShapes;

            /// <summary>
            /// Location of the Torch model.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "Torch model used by the transform.", SortOrder = 0)]
            internal string ModelLocation = ".";
        }

        private readonly string _outputColumnName;
        private readonly string[] _inputColumnNames;
        private readonly long[][] _inputShapes;

        internal TorchScorerEstimator(IHostEnvironment env, Options options, TorchModuleWrapper torchModule)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchScorerEstimator)),
                  new TorchScorerTransformer(env, torchModule, options.OutputColumnName, options.InputColumnNames, options.InputShapes, options.ModelLocation))
        {
            Host.CheckNonEmpty(options.OutputColumnName, nameof(options.OutputColumnName));
            Host.CheckValue(options.InputShapes, nameof(options.InputShapes));
            Host.CheckParam(!options.InputShapes.Any(x => x.Any(y => y < -1)), nameof(options.InputShapes), "Negative shape dimensions not supported.");
            _outputColumnName = options.OutputColumnName;
            _inputColumnNames = options.InputColumnNames ?? new[] { options.OutputColumnName };
            _inputShapes = options.InputShapes;
        }

        internal TorchScorerEstimator(IHostEnvironment env, Options options)
            : this(env, options, TorchUtils.LoadTorchModel(env, options.ModelLocation).Module)
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var resultDic = inputSchema.ToDictionary(x => x.Name);

            for (var i = 0; i < _inputColumnNames.Length; i++)
            {
                var input = _inputColumnNames[i];
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (col.Kind != SchemaShape.Column.VectorKind.Vector || col.ItemType != NumberDataViewType.Single)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector of Single", col.GetTypeString());
            }

            resultDic[_outputColumnName] = new SchemaShape.Column(
                _outputColumnName,
                SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single,
                false);

            return new SchemaShape(resultDic.Values);
        }
    }
}
