
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
    class TransferLearningTransformer : RowToRowTransformerBase
    {
        private readonly string _savedModelPath = "resnet_v2_101_299_frozen.pb";
        internal readonly Session Session;
        internal readonly DataViewType[] OutputTypes;
        internal readonly TF_DataType[] TFOutputTypes;
        internal readonly TF_DataType[] TFInputTypes;
        internal Graph Graph => Session.graph;

        internal readonly string[] Inputs;
        internal readonly string[] Outputs;

        internal static int BatchSize = 1;
        internal const string Summary = "Transforms the data using the Transfer Learning model.";
        internal const string UserName = "TransferLearningTransform";
        internal const string ShortName = "TLTransform";
        internal const string LoaderSignature = "TransferLearningTransform";

        private Operation _trainStep;
        private Tensor _finalTensor;
        private Tensor _bottleneckInput;
        private Tensor _crossEntropy;
        private Tensor _groundTruthInput;
        private Tensor _evaluationStep;
        internal string SavePath = "TransferLearningModel";
        internal float LearningRate = 0.01f;

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

        internal TransferLearningTransformer(IHostEnvironment env, Session session = null, string[] outputColumnNames, string[] inputColumnNames = null, string savedModelPath = null) :
            base(Contracts.CheckRef(env, nameof(env)).Register(nameof(TransferLearningTransformer)))

        {
            Host.CheckValue(session, nameof(session));
            Host.CheckNonEmpty(inputColumnNames, nameof(inputColumnNames));
            Host.CheckNonEmpty(outputColumnNames, nameof(outputColumnNames));
            if (session == null) {
                if (savedModelPath != null)
                    _savedModelPath = savedModelPath;
                Session = LoadSession(env, null, _savedModelPath);
            } else
            {
                Session = session;
            }
            if (inputColumnNames == null)
            {
                Inputs = new string[] { null };
            } else
            {
                Inputs = inputColumnNames;
            }
            Outputs = outputColumnNames;

            (TFInputTypes, TFInputShapes) = GetInputInfo(Host, Session, Inputs);
            (TFOutputTypes, OutputTypes) = GetOutputInfo(Host, Session, Outputs);
        }

        internal static Session LoadSession(IExceptionContext ectx, byte[] modelBytes = null, string modelFile = null)
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
            trainSaver.save(Session, SavePath);
        }

        public void TransferLearning(IDataView input, Operation bottleneckTensor, Operation inputTensor, int trainingIterations, int evalStepInterval)
        {
            var labelCol = input.Schema.Label.Value;
            var classCount = 4;
            if (labelCol.Type is KeyDataViewType labelKeyType)
                classCount = labelKeyType.GetCountAsInt32(Host);


            // Check if the last layer has already been added to the graph if not then add
            bool transferLayerExists = false;
            if (Graph.OperationByName(outputTensorName) == null)
            {
                Console.WriteLine("Transfer Learning Layer already created");
                transferLayerExists = true;
            }
            else
            {

                with(Graph, delegate
                {

                    (_trainStep, _crossEntropy, _bottleneckInput,
                     _groundTruthInput, _finalTensor) = addFinalLayer(
                         classCount, outputTensorName, bottleneckTensor,
                         is_training: true);
                });
            }


            with(Session, sess =>
            {
                // Initialize all weights: for the module to their pretrained values,
                // and for the newly added retraining layer to random initial values.
                var init = tf.global_variables_initializer();
                sess.run(init);

                // Create the operations we need to evaluate the accuracy of our new layer.
                //if (!transferLayerExists)
                if (transferLayerExists)
                {
                    _evaluationStep = Graph.OperationByName("accuracy/accuracy/Mean");
                }
                else
                {
                    (Tensor evaluation_step, Tensor _) = addEvaluationStep(_finalTensor, _groundTruthInput);
                }


                //else Tensor evaluation_step = 


                // Merge all the summaries and write them out to the summaries_dir
                var merged = tf.summary.merge_all();
                var trainWriter = tf.summary.FileWriter("/train", sess.graph);
                var validationWriter = tf.summary.FileWriter("/validation", sess.graph);




                IDataView transformedValues; // Getoutput o model
                float[] predictions = new float[4004];
                VBuffer<float>[] outputValue = new VBuffer<float>[4];
                long[] truth = { 3, 2, 1, 3 };
                for (int i = 0; i < trainingIterations; i++)
                {
                    using (var cursor = transformedValues.GetRowCursor(transformedValues.Schema))
                    {


                        var predictionValues = cursor.GetGetter<VBuffer<float>>(cursor.Schema["output"]);
                        int count = 0;
                        while (cursor.MoveNext())
                        {
                            predictionValues(ref outputValue[count]);
                            count++;
                            // Feed the bottlenecks and ground truth into the graph, and run a training
                            // step. Capture training summaries for TensorBoard with the `merged` op.

                        }
                        for (int index = 0; index < 4; index++)
                            Array.Copy(outputValue[index].GetBuffer(), 0, predictions, index * 1001, 1001);

                        NumSharp.NDArray results = sess.run(
                                  new ITensorOrOperation[] { merged, _trainStep },
                                  new FeedItem(_bottleneckInput,
                                  new NDArray(predictions, new Shape(new[] { 4, 1001 }))),
                                  new FeedItem(_groundTruthInput, truth));
                        var trainSummary = results[0];
                        Console.WriteLine("Trained");

                        results = sess.run(
                                  new ITensorOrOperation[] { _finalTensor },
                                  new FeedItem(_bottleneckInput,
                                  new NDArray(predictions, new Shape(new[] { 4, 1001 }))));

                        // TODO
                        print(results[0]);
                        trainWriter.add_summary(trainSummary, i);


                        // Every so often, print out how well the graph is training.
                        bool isLastStep = (i + 1 == trainingIterations);
                        if ((i % evalStepInterval) == 0 || isLastStep)
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

        private (Tensor, Tensor) addEvaluationStep(Tensor resultTensor, Tensor groundTruthTensor)
        {
            Tensor evaluationStep = null, correctPrediction = null, prediction = null;

            with(tf.name_scope("accuracy"), scope =>
            {
                with(tf.name_scope("correct_prediction"), delegate
                {
                    prediction = tf.argmax(resultTensor, 1);
                    correctPrediction = tf.equal(prediction, groundTruthTensor);
                });

                with(tf.name_scope("accuracy"), delegate
                {
                    evaluationStep = tf.reduce_mean(tf.cast(correctPrediction, tf.float32));
                });
            });

            tf.summary.scalar("accuracy", evaluationStep);
            return (evaluationStep, prediction);
        }



        private (Operation, Tensor, Tensor, Tensor, Tensor) addFinalLayer(int classCount, string finalTensorName,
            Tensor bottleneckTensor, bool isTraining)
        {
            var (batch_size, bottleneck_tensor_size) = (bottleneckTensor.GetShape().Dimensions[0], bottleneckTensor.GetShape().Dimensions[1]);
            with(tf.name_scope("input"), scope =>
            {
                _bottleneckInput = tf.placeholder_with_default(
                    bottleneckTensor,
                    shape: bottleneckTensor.GetShape().Dimensions,
                    name: "BottleneckInputPlaceholder");

                _groundTruthInput = tf.placeholder(tf.int64, new TensorShape(batch_size), name: "GroundTruthInput");
            });

            // Organizing the following ops so they are easier to see in TensorBoard.
            string layerName = "final_retrain_ops";
            Tensor logits = null;
            with(tf.name_scope(layerName), scope =>
            {
                RefVariable layerWeights = null;
                with(tf.name_scope("weights"), delegate
                {
                    var initialValue = tf.truncated_normal(new int[] { bottleneck_tensor_size, classCount }, stddev: 0.001f);
                    layerWeights = tf.Variable(initialValue, name: "final_weights");
                    variableSummaries(layerWeights);
                });

                RefVariable layerBiases = null;
                with(tf.name_scope("biases"), delegate
                {
                    layerBiases = tf.Variable(tf.zeros((classCount)), name: "final_biases");
                    variableSummaries(layerBiases);
                });

                with(tf.name_scope("Wx_plus_b"), delegate
                {
                    logits = tf.matmul(_bottleneckInput, layerWeights) + layerBiases;
                    tf.summary.histogram("pre_activations", logits);
                });
            });

            _finalTensor = tf.nn.softmax(logits, name: finalTensorName);


            tf.summary.histogram("activations", _finalTensor);

            // If this is an eval graph, we don't need to add loss ops or an optimizer.
            if (!isTraining)
                return (null, null, _bottleneckInput, _groundTruthInput, _finalTensor);

            Tensor crossEntropyMean = null;
            with(tf.name_scope("cross_entropy"), delegate
            {
                crossEntropyMean = tf.losses.sparse_softmax_cross_entropy(
                    labels: _groundTruthInput, logits: logits);
            });

            tf.summary.scalar("cross_entropy", crossEntropyMean);

            with(tf.name_scope("train"), delegate
            {
                var optimizer = tf.train.GradientDescentOptimizer(LearningRate);
                _trainStep = optimizer.minimize(crossEntropyMean);
            });

            return (_trainStep, crossEntropyMean, _bottleneckInput, _groundTruthInput,
                _finalTensor);
        }

        private void variableSummaries(RefVariable var)
        {
            with(tf.name_scope("summaries"), delegate
            {
                var mean = tf.reduce_mean(var);
                tf.summary.scalar("mean", mean);
                Tensor stddev = null;
                with(tf.name_scope("stddev"), delegate {
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
                });
                tf.summary.scalar("stddev", stddev);
                tf.summary.scalar("max", tf.reduce_max(var));
                tf.summary.scalar("min", tf.reduce_min(var));
                tf.summary.histogram("histogram", var);
            });
        }
    }
}
