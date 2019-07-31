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
using TorchSharp.Tensor;
using static Microsoft.ML.Transforms.TorchScorerTransformer;

namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Estimator for the <see cref="TorchScorerTransformer"/> when a model needs to be trained or fine-tuned.
    /// </summary>
    public sealed class TorchTrainerEstimator : IEstimator<TorchScorerTransformer>
    {
        /// <summary>
        /// The options for the <see cref="TorchTrainerEstimator"/>.
        /// </summary>
        internal sealed class Options : TransformInputBase
        {
            /// <summary>
            /// The name of the output column of the transformation.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the output column", Name = "Name", ShortName = "name", SortOrder = 3)]
            public string OutputColumnName;

            /// <summary>
            /// The names of the columns containing the inputs for the model. If <see langword="null"/>, this defaults to <see cref="OutputColumnName"/>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The names of the input columns", Name = "Sources", ShortName = "src", SortOrder = 1)]
            public string[] InputColumnNames;

            /// <summary>
            /// The names of the columns containing the taget column for the model. If <see langword="null"/>, this defaults to Label.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The name of the input columns", Name = "Label", ShortName = "label", SortOrder = 2)]
            public string InputLabelColumnName;

            /// <summary>
            /// The shape of the model input.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The shape of the input tensor", ShortName = "shape", SortOrder = 0)]
            public long[][] InputShapes;

            /// <summary>
            /// The optimizer used for training. Default is Adam.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The optimizer used for training", ShortName = "optimizer", SortOrder = 3)]
            public TorchSharp.NN.Optimizer Optimizer;

            /// <summary>
            /// The loss function used for training.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce | ArgumentType.Required, HelpText = "The loss function used for training", ShortName = "loss", SortOrder = 4)]
            public TorchSharp.NN.LossFunction.Loss Loss;

            /// <summary>
            /// The number of epochs for trainig. Default is 10.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The number of epochs", ShortName = "epochs", SortOrder = 5)]
            public int Epochs;

            /// <summary>
            /// The number of epochs for trainig. Default is 32.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "The batch size", ShortName = "bs", SortOrder = 6)]
            public int BatchSize;
        }

        private readonly IHost _host;
        private readonly string _outputColumnName;
        private readonly string[] _inputColumnNames;
        private readonly long[][] _inputShapes;
        private readonly string _labelColumnName;
        private readonly TorchModuleWrapper _module;
        private readonly TorchSharp.NN.Optimizer _optimizer;
        private readonly TorchSharp.NN.LossFunction.Loss _loss;
        private readonly int _epochs;
        private readonly int _batchSize;

        internal TorchTrainerEstimator(IHostEnvironment env, Options options, TorchModel torchModel)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(TorchTrainerEstimator));
            _host.CheckNonEmpty(options.OutputColumnName, nameof(options.OutputColumnName));
            _host.CheckValue(options.InputShapes, nameof(options.InputShapes));
            _host.CheckParam(!options.InputShapes.Any(x => x.Any(y => y < -1)), nameof(options.InputShapes), "Negative shape dimensions not supported.");
            if (!(torchModel.Module is TorchNNModuleWrapper nnModule))
            {
                throw _host.Except("Training of JIT modules not supported yet");
            }
            _outputColumnName = options.OutputColumnName;
            _inputColumnNames = options.InputColumnNames ?? new[] { options.OutputColumnName };
            _labelColumnName = options.InputLabelColumnName;
            _inputShapes = options.InputShapes;
            _module = torchModel.Module;
            _optimizer = options.Optimizer;
            _loss = options.Loss;
            _epochs = options.Epochs;
            _batchSize = options.BatchSize;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            var resultDic = inputSchema.ToDictionary(x => x.Name);

            for (var i = 0; i < _inputColumnNames.Length; i++)
            {
                var input = _inputColumnNames[i];
                if (!inputSchema.TryFindColumn(input, out var col))
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input);
                if (col.Kind != SchemaShape.Column.VectorKind.Vector || col.ItemType != NumberDataViewType.Single)
                    throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", input, "vector of Single", col.GetTypeString());
            }

            resultDic[_outputColumnName] = new SchemaShape.Column(
                _outputColumnName,
                SchemaShape.Column.VectorKind.VariableVector,
                NumberDataViewType.Single,
                false);

            return new SchemaShape(resultDic.Values);
        }

        public TorchScorerTransformer Fit(IDataView input)
        {
            var inputsForTraining = new string[_inputColumnNames.Length + 1];
            var inputColIndices = new int[inputsForTraining.Length];
            var isInputVector = new bool[inputsForTraining.Length];
            var inputTypes = new Type[inputsForTraining.Length];
            var inputShapes = new long[_inputColumnNames.Length + 1][];

            for (int i = 0; i < _inputColumnNames.Length; i++)
            {
                inputShapes[i] = _inputShapes[i];
            }
            inputShapes[_inputColumnNames.Length] = new long[] { -1 };

            for (int i = 0; i < _inputColumnNames.Length; i++)
            {
                inputsForTraining[i] = _inputColumnNames[i];
            }

            var inputSchema = input.Schema;
            for (int i = 0; i < inputsForTraining.Length - 1; i++)
            {
                (inputColIndices[i], isInputVector[i], inputTypes[i]) =
                    GetTrainingInputInfo(inputSchema, inputsForTraining[i], ref inputShapes[i], _batchSize);
            }

            var index = inputsForTraining.Length - 1;
            inputsForTraining[index] = _labelColumnName;
            (inputColIndices[index], isInputVector[index], inputTypes[index]) =
                    GetTrainingInputInfo(inputSchema, _labelColumnName, ref inputShapes[index], _batchSize);

            var cols = input.Schema.Where(c => inputColIndices.Contains(c.Index));

            var nnModule = _module as TorchNNModuleWrapper;
            nnModule.Module.Train();

            for (int epoch = 0; epoch < _epochs; epoch++)
            {
                using (var cursor = input.GetRowCursor(cols))
                {
                    var srcTensorGetters = GetTensorValueGetters(cursor, inputColIndices, isInputVector, inputTypes, inputShapes);

                    float loss = 0;
                    bool isDataLeft = false;
                    using (var ch = _host.Start("Training Torch model..."))
                    using (var pch = _host.StartProgressChannel("Torch training progress..."))
                    {
                        pch.SetHeader(new ProgressHeader(new[] { "Loss" }, new[] { "Epoch" }), (e) => e.SetProgress(0, epoch, _epochs));

                        while (cursor.MoveNext())
                        {
                            for (int i = 0; i < inputColIndices.Length; i++)
                            {
                                isDataLeft = true;
                                srcTensorGetters[i].BufferTrainingData();
                            }

                            if (((cursor.Position + 1) % _batchSize) == 0)
                            {
                                isDataLeft = false;
                                loss += TrainBatch(inputColIndices, inputsForTraining, srcTensorGetters);
                            }
                        }
                        if (isDataLeft)
                        {
                            isDataLeft = false;
                            ch.Warning("Not training on the last batch. The batch size is less than {0}.", _batchSize);
                        }
                        pch.Checkpoint(new double?[] { loss });
                    }
                }
            }

            nnModule.Module.Eval();

            return new TorchScorerTransformer(_host, _module, _outputColumnName, _inputColumnNames, _inputShapes, ".");
        }

        private float TrainBatch(int[] inputColIndices,
           string[] inputsForTraining,
           ITensorValueGetter[] srcTensorGetters)
        {
            float loss = 0;
            var inputTensors = new TorchTensor[inputColIndices.Length - 1];
            TorchTensor targetTensor;

            for (int i = 0; i < inputColIndices.Length - 1; i++)
            {
                inputTensors[i] = srcTensorGetters[i].GetBufferedBatchTensor();
            }

            targetTensor = srcTensorGetters[inputColIndices.Length - 1].GetBufferedBatchTensor();

            _optimizer.ZeroGrad();

            using (var prediction = _module.Forward(inputTensors))
            using (var output = _loss(prediction, targetTensor))
            {
                output.Backward();

                _optimizer.Step();

                loss = output.DataItem<float>();

                foreach (var tensor in inputTensors)
                    tensor.Dispose();
                targetTensor.Dispose();
            }

            return loss;
        }

        private (int, bool, Type) GetTrainingInputInfo(DataViewSchema inputSchema, string inputTrainingColumn, ref long[] inputShape, int batchSize)
        {
            if (!inputSchema.TryGetColumnIndex(inputTrainingColumn, out int inputColIndex))
                throw _host.Except($"Column {inputTrainingColumn} doesn't exist");

            // Check input column types
            var type = inputSchema[inputColIndex].Type;
            var rawType = type.RawType;
            var isInputVector = false;

            if (type is VectorDataViewType vectorType)
            {
                rawType = vectorType.ItemType.RawType;
                isInputVector = true;
                var colTypeDims = vectorType.Dimensions.Select(dim => (long)dim).ToArray();
                var colShapeLength = colTypeDims.Length;

                if (colShapeLength != inputShape.Length)
                    throw _host.Except($"Input shape mismatch: Input Column '{inputTrainingColumn}' vector shape length {colShapeLength} does not match expected {inputShape.Length}.");

                if (inputShape.Where(x => x == -1).Count() > 1)
                    throw _host.Except($"Shape with mode than 1 undefined dimension: Input Column '{inputTrainingColumn}'.");

                int j = 0;
                if (batchSize > 1)
                {
                    j = 1;
                    var actualShape = new long[inputShape.Length + 1];
                    actualShape[0] = batchSize;
                    for (int i = 0; i < inputShape.Length; i++)
                    {
                        actualShape[i + 1] = inputShape[i];
                    }
                    inputShape = actualShape;
                } // else there is a bug with the following j -1

                for (; j < colShapeLength + 1; j++)
                {
                    if (colTypeDims[j - 1] != inputShape[j] && inputShape[j] != -1)
                        throw _host.Except($"Input shape mismatch: Input Column '{inputTrainingColumn}' dimension {j} of size {colTypeDims[j - 1]} does not match expected size.");
                    if (inputShape[j] == -1)
                        inputShape[j] = colTypeDims[j - 1];
                }
            }
            else
            {
                if (batchSize > 1)
                {
                    inputShape = new long[] { batchSize, 1 };
                }
                else
                {
                    inputShape = new long[] { 1 };
                }
            }

            return (inputColIndex, isInputVector, rawType);
        }
    }
}
