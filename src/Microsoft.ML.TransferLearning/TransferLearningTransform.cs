
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.TransferLearning;
using NumSharp;
using Tensorflow;
using static Tensorflow.Python;

[assembly: LoadableClass(TransferLearningTransformer.Summary, typeof(IDataTransform), typeof(TransferLearningTransformer),
    typeof(TransferLearningEstimator.Options), typeof(SignatureDataTransform), TransferLearningTransformer.UserName, TransferLearningTransformer.ShortName)]

[assembly: LoadableClass(TransferLearningTransformer.Summary, typeof(IDataTransform), typeof(TransferLearningTransformer), null, typeof(SignatureLoadDataTransform),
    TransferLearningTransformer.UserName, TransferLearningTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(TransferLearningTransformer), null, typeof(SignatureLoadModel),
    TransferLearningTransformer.UserName, TransferLearningTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(TransferLearningTransformer), null, typeof(SignatureLoadRowMapper),
    TransferLearningTransformer.UserName, TransferLearningTransformer.LoaderSignature)]

[assembly: EntryPointModule(typeof(TransferLearningTransformer))]

namespace Microsoft.ML.Transforms.TransferLearning
{
    public sealed class TransferLearningTransformer : RowToRowTransformerBase
    {
        private readonly string _savedModelPath;
        private readonly string _loadedModelOutputTensorName;
        private readonly string _loadedModelInputTensorName;
        internal readonly Session Session;
        internal Graph Graph => Session.graph;

        internal const string Summary = "Transforms the data using the Transfer Learning model.";
        internal const string UserName = "TransferLearningTransform";
        internal const string ShortName = "TLTransform";
        internal const string LoaderSignature = "TransferLearningTransform";

        private Operation _trainStep;
        private Tensor _loadedModelOutput;
        private Tensor _loadedModelInput;
        private Tensor _finalTensor;
        private Tensor _bottleneckInput;
        private Tensor _crossEntropy;
        private Tensor _groundTruthInput;
        private Tensor _evaluationStep;
        private TransferLearningEstimator.Options _options;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TENSFLOW",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002,  // Added Support for Multiple Outputs and SavedModel.
                verWrittenCur: 0x00010003,  // Added Support for adding batch dimension in inputs.
                verReadableCur: 0x00010003,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TransferLearningTransformer).Assembly.FullName);
        }

        internal TransferLearningTransformer(IHostEnvironment env, TransferLearningEstimator.Options options, IDataView input) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TransferLearningTransformer)))

        {
            Host.CheckNonEmpty(options.InputColumns, nameof(options.InputColumns));
            Host.CheckNonEmpty(options.OutputColumns, nameof(options.OutputColumns));
            _options = options;

            switch (_options.Model) {
                case TransferLearningEstimator.Options.ModelType.InceptionV3:
                    throw new NotImplementedException();
                case TransferLearningEstimator.Options.ModelType.Resnet101:
                    _savedModelPath = "resnet_v2_101_299_frozen.pb";
                    //_loadedModelOutputTensorName = "resnet_v2_101/SpatialSqueeze";
                    _loadedModelInputTensorName = "input";
                    _loadedModelOutputTensorName = "output"; //If this doesn't work use the name above but should because output is an identity tensor
                    break;
            }

            Session = LoadSession(env, _savedModelPath);
            _loadedModelOutput = Graph.OperationByName(_loadedModelOutputTensorName);
            _loadedModelInput = Graph.OperationByName(_loadedModelInputTensorName);
            if (_loadedModelOutput == null || _loadedModelInput == null)
            {
                throw new Exception("Output tensor of pretrained model not found");
            }
            TransferLearning(input);
        }

        internal static Session LoadSession(IExceptionContext ectx, string modelFile = null)
        {
            var graph = tf.Graph().as_default();
            try
            {
                if (IsMetaGraph(modelFile))
                {
                    tf.train.import_meta_graph(modelFile);
                }
                else
                {
                    graph.Import(modelFile);
                }
            }
            catch (Exception ex)
            {
                if (!string.IsNullOrEmpty(modelFile))
                    throw ectx.Except($"TensorFlow exception triggered while loading model from '{modelFile}'");
#pragma warning disable MSML_NoMessagesForLoadContext
                throw ectx.ExceptDecode(ex, "Tensorflow exception triggered while loading model.");
#pragma warning restore MSML_NoMessagesForLoadContext

            }
            return new Session(graph);
        }

        internal static bool IsMetaGraph(string modelFile)
        {
            return modelFile.EndsWith(".meta");
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            var trainSaver = tf.train.Saver();
            trainSaver.save(Session, _options.SavePath);
        }

        public void TransferLearning(IDataView input)
        {
            var labelCol = input.Schema.GetColumnOrNull(_options.LabelColumn);
            var classCount = (labelCol.Value.Type as KeyDataViewType).Count;

            // Check if the last layer has already been added to the graph if not then add
            bool transferLayerExists = false;
            if (Graph.OperationByName(_options.OutputTensorName) == null)
            {
                transferLayerExists = true;
            }
            else
            {

                with(Graph, delegate
                {

                    (_trainStep, _crossEntropy, _bottleneckInput,
                     _groundTruthInput, _finalTensor) = AddFinalLayer(
                         (int) classCount, _options.OutputTensorName, _loadedModelOutput,
                         isTraining: true);
                });
            }

            with(Session, sess =>
            {
                // Initialize all weights: for the module to their pretrained values,
                // and for the newly added retraining layer to random initial values.
                var init = tf.global_variables_initializer();
                sess.run(init);

                // Create the operations we need to evaluate the accuracy of our new layer.
                if (transferLayerExists)
                {
                    _evaluationStep = Graph.OperationByName(_options.EvaluationNameScope+"/"+_options.EvaluationNameScope+"/Mean");
                }
                else
                {
                    (Tensor evaluation_step, Tensor _) = AddEvaluationStep(_finalTensor, _groundTruthInput);
                    _evaluationStep = evaluation_step;
                }

                // Merge all the summaries and write them out to the summaries_dir
                var merged = tf.summary.merge_all();
                var trainWriter = tf.summary.FileWriter("/" + _options.TrainWriterName, sess.graph);
                var validationWriter = tf.summary.FileWriter("/" + _options.ValidationWriterName, sess.graph);

                IDataView transformedValues = input; // Get output from model not really this is just to turnoff compile errors from unimplemented code
                float[] predictions = new float[4004];
                VBuffer<float>[] outputValue = new VBuffer<float>[4];
                long[] truth = { 3, 2, 1, 3 };
                for (int i = 0; i < _options.Epoch; i++)
                {
                    using (var cursor = transformedValues.GetRowCursor(transformedValues.Schema))
                    {
                        // This is where the output value is pulled from so use OutputColumnName create design for predicted label
                        var predictionValues = cursor.GetGetter<VBuffer<float>>(cursor.Schema[_options.OutputColumns[0]]);
                        int count = 0;
                        while (cursor.MoveNext())
                        {
                            predictionValues(ref outputValue[count]);
                            count++;
                            // Feed the bottlenecks and ground truth into the graph, and run a training
                            // step. Capture training summaries for TensorBoard with the `merged` op.

                        }
                        for (int index = 0; index < 4; index++)
                            // Used GetValues method instead of GetBuffer which we implemeneted so may break
                            Array.Copy(outputValue[index].GetBuffer(), 0, predictions, index * 1001, 1001);

                        NumSharp.NDArray results = sess.run(
                                  new ITensorOrOperation[] { merged, _trainStep },
                                  new FeedItem(_bottleneckInput,
                                  new NDArray(predictions, new Shape(new[] { 4, 1001 }))),
                                  new FeedItem(_groundTruthInput, truth));
                        var trainSummary = results[0];

                        results = sess.run(
                                  new ITensorOrOperation[] { _finalTensor },
                                  new FeedItem(_bottleneckInput,
                                  new NDArray(predictions, new Shape(new[] { 4, 1001 }))));

                        // TODO
                        print(results[0]);
                        trainWriter.add_summary(trainSummary, i);

                        // Every so often, print out how well the graph is training.
                        bool isLastStep = (i + 1 == _options.Epoch);
                        if ((i % _options.EvaluationStepInterval) == 0 || isLastStep)
                        {
                            results = sess.run(
                                new Tensor[] { _evaluationStep, _crossEntropy },
                                new FeedItem(_bottleneckInput,
                              new NDArray(predictions, new Shape(new[] { 4, 1001 }))),
                                new FeedItem(_groundTruthInput, truth));
                            (float train_accuracy, float cross_entropy_value) = (results[0], results[1]);
                            print($"{DateTime.Now}: Step {i + 1}: Train accuracy = {train_accuracy * 100}%,  Cross entropy = {cross_entropy_value.ToString("G4")}");

                            // Run a validation step and capture training summaries for TensorBoard
                            // with the `merged` op.
                            results = sess.run(new Tensor[] { merged, _evaluationStep },
                                new FeedItem(_bottleneckInput,
                              new NDArray(predictions, new Shape(new[] { 4, 1001 }))),
                                new FeedItem(_groundTruthInput, truth));

                            (string validation_summary, float validation_accuracy) = (results[0], results[1]);

                            validationWriter.add_summary(validation_summary, i);
                            print($"{DateTime.Now}: Step {i + 1}: Validation accuracy = {validation_accuracy * 100}% (N={len(predictions)})");

                        }
                    }
                }
            });
        }

        private (Tensor, Tensor) AddEvaluationStep(Tensor resultTensor, Tensor groundTruthTensor)
        {
            Tensor evaluationStep = null;
            Tensor correctPrediction = null;
            Tensor prediction = null;

            with(tf.name_scope(_options.EvaluationNameScope), scope =>
            {
                with(tf.name_scope(_options.PredictionComparisonNameScope), delegate
                {
                    prediction = tf.argmax(resultTensor, 1);
                    correctPrediction = tf.equal(prediction, groundTruthTensor);
                });

                with(tf.name_scope(_options.EvaluationNameScope), delegate
                {
                    evaluationStep = tf.reduce_mean(tf.cast(correctPrediction, tf.float32));
                });
            });

            tf.summary.scalar(_options.EvaluationNameScope, evaluationStep);
            return (evaluationStep, prediction);
        }

        private (Operation, Tensor, Tensor, Tensor, Tensor) AddFinalLayer(int classCount, string finalTensorName,
            Tensor bottleneckTensor, bool isTraining)
        {
            var (batch_size, bottleneck_tensor_size) = (bottleneckTensor.TensorShape.Dimensions[0], bottleneckTensor.TensorShape.Dimensions[1]);
            with(tf.name_scope(_options.FinalLayerInputNameScope), scope =>
            {
                _bottleneckInput = tf.placeholder_with_default(
                    bottleneckTensor,
                    shape: bottleneckTensor.TensorShape.Dimensions,
                    name: _options.BottleneckInputTensorName);

                _groundTruthInput = tf.placeholder(tf.int64, new TensorShape(batch_size), name: _options.GroundTruthTensorName);
            });

            // Organizing the following ops so they are easier to see in TensorBoard.
            string layerName = _options.RetrainLayerName;
            Tensor logits = null;
            with(tf.name_scope(layerName), scope =>
            {
                RefVariable layerWeights = null;
                with(tf.name_scope(_options.WeightsNameScope), delegate
                {
                    var initialValue = tf.truncated_normal(new int[] { bottleneck_tensor_size, classCount }, stddev: _options.WeightsStd);
                    layerWeights = tf.Variable(initialValue, name: _options.WeightsName);
                    VariableSummaries(layerWeights);
                });

                RefVariable layerBiases = null;
                with(tf.name_scope(_options.BiasesNameScope), delegate
                {
                    layerBiases = tf.Variable(tf.zeros((classCount)), name: _options.BiasesName);
                    VariableSummaries(layerBiases);
                });

                with(tf.name_scope(_options.MatrixMultiplicationNameScope), delegate
                {
                    logits = tf.matmul(_bottleneckInput, layerWeights) + layerBiases;
                    tf.summary.histogram(_options.MatrixMultiplicationHistogram, logits);
                });
            });

            _finalTensor = tf.nn.softmax(logits, name: finalTensorName);

            tf.summary.histogram(_options.SoftMaxHistogram, _finalTensor);

            // If this is an eval graph, we don't need to add loss ops or an optimizer.
            if (!isTraining)
                return (null, null, _bottleneckInput, _groundTruthInput, _finalTensor);

            Tensor crossEntropyMean = null;
            with(tf.name_scope(_options.CrossEntropyName), delegate
            {
                crossEntropyMean = tf.losses.sparse_softmax_cross_entropy(
                    labels: _groundTruthInput, logits: logits);
            });

            tf.summary.scalar(_options.CrossEntropyName, crossEntropyMean);

            with(tf.name_scope(_options.TrainNameScope), delegate
            {
                var optimizer = tf.train.GradientDescentOptimizer(_options.LearningRate);
                _trainStep = optimizer.minimize(crossEntropyMean);
            });

            return (_trainStep, crossEntropyMean, _bottleneckInput, _groundTruthInput,
                _finalTensor);
        }

        private void VariableSummaries(RefVariable var)
        {
            with(tf.name_scope(_options.SummariesNameScope), delegate
            {
                var mean = tf.reduce_mean(var);
                tf.summary.scalar(_options.MeanName, mean);
                Tensor stddev = null;
                with(tf.name_scope(_options.StdDevName), delegate
                {
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
                });
                tf.summary.scalar(_options.StdDevName, stddev);
                tf.summary.scalar(_options.MaxName, tf.reduce_max(var));
                tf.summary.scalar(_options.MinName, tf.reduce_min(var));
                tf.summary.histogram(_options.HistogramName, var);
            });
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema)
        {
            throw new NotImplementedException();
        }
    }

    public sealed class TransferLearningEstimator : IEstimator<TransferLearningTransformer>
    {

        internal sealed class Options : TransformInputBase
        {
            public enum ModelType {
                InceptionV3,
                Resnet101
            };

            public ModelType Model;
            public float LearningRate;
            public int BatchSize;
            public int Epoch;
            public string[] InputColumns;
            public string[] OutputColumns;
            public string LabelColumn;
            public string SavePath;
            public string OutputTensorName;
            public string EvaluationNameScope;
            public string PredictionComparisonNameScope;
            public string BottleneckInputTensorName;
            public string GroundTruthTensorName;
            public int EvaluationStepInterval;
            public string FinalLayerInputNameScope;
            public string RetrainLayerName;
            public string WeightsNameScope;
            public string WeightsName;
            public float WeightsStd;
            public string BiasesNameScope;
            public string BiasesName;
            public string MatrixMultiplicationNameScope;
            public string MatrixMultiplicationHistogram;
            public string SoftMaxHistogram;
            public string CrossEntropyName;
            public string TrainNameScope;
            public string SummariesNameScope;
            public string MeanName;
            public string StdDevName;
            public string MaxName;
            public string MinName;
            public string HistogramName;
            public string TrainWriterName;
            public string ValidationWriterName;

            internal Options(string[] inputColumns, string[] outputColumns, string labelColumn, ModelType modelType = ModelType.Resnet101,
                int batchSize = 1, int epoch = 10, float learningRate = .01f, string savePath = "TransferLearningModel",
                string outputTensorName = "FinalOutput", string evaluationNameScope = "accuracy",
                string predictionComparisonNameScope = "correct_prediction", string bottleneckInputTensorName = "BottleneckInputPlaceholder",
                string groundTruthTensorName = "GroundTruthInput", int evaluationStepInterval = 10, string finalLayerInputNameScope = "input",
                string retrainLayerName = "final_retrain_ops", string weightsNameScope = "weights", string weightsName = "final_weights",
                float weightsStd = 0.001f, string biasesNameScope = "biases", string biasesName = "final_biases",
                string matrixMultiplicationNameScope = "Wx_plus_b", string matrixMultiplicationHistogram = "pre_activations",
                string softMaxHistogram = "activations", string crossEntropyName = "cross_entropy", string trainNameScope = "train",
                string summariesNameScope = "summaries", string meanName = "mean", string stdDevName = "stddev",
                string maxName = "max", string minName = "min", string historgramName = "histogram", string trainWriterName = "train",
                string validationWriterName = "validation")
            {
                InputColumns = inputColumns;
                OutputColumns = outputColumns;
                LabelColumn = labelColumn;
                Model = modelType;
                BatchSize = batchSize;
                Epoch = epoch;
                LearningRate = learningRate;
                SavePath = savePath;
                OutputTensorName = outputTensorName;
                EvaluationNameScope = evaluationNameScope;
                PredictionComparisonNameScope = predictionComparisonNameScope;
                BottleneckInputTensorName = bottleneckInputTensorName;
                GroundTruthTensorName = groundTruthTensorName;
                EvaluationStepInterval = evaluationStepInterval;
                FinalLayerInputNameScope = finalLayerInputNameScope;
                RetrainLayerName = retrainLayerName;
                WeightsNameScope = weightsNameScope;
                WeightsName = weightsName;
                WeightsStd = weightsStd;
                BiasesNameScope = biasesNameScope;
                BiasesName = biasesName;
                MatrixMultiplicationNameScope = matrixMultiplicationNameScope;
                MatrixMultiplicationHistogram = matrixMultiplicationHistogram;
                SoftMaxHistogram = softMaxHistogram;
                CrossEntropyName = crossEntropyName;
                TrainNameScope = trainNameScope;
                SummariesNameScope = summariesNameScope;
                MeanName = meanName;
                StdDevName = stdDevName;
                MaxName = maxName;
                MinName = minName;
                HistogramName = historgramName;
                TrainWriterName = trainWriterName;
                ValidationWriterName = validationWriterName;
            }

        }

        private readonly IHost _host;
        private readonly Options _options;
        private TransferLearningTransformer _transformer;

        internal TransferLearningEstimator(IHost env, Options options)
        {
            _host = env;
            _options = options;
        }

        public TransferLearningTransformer Fit(IDataView input)
        {
            _host.CheckValue(input, nameof(input));
            if (_transformer == null)
            {
                _transformer = new TransferLearningTransformer(_host, _options, input);
            }
            // Validate input schema.
            _transformer.GetOutputSchema(input.Schema);
            return _transformer;
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }
}
