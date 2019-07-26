# Deep Neural Network support in ML .NET
## Motivation
To improve solutions for scenarios such as image and text classification and object detection. Currently we have Tensorflow and ONNX transfomers that enable scoring of Tensorflow and ONNX models within the pipeline. We now want to enable full model training and transfer learning and create APIs that enhance and also make it easy to do the above outlined tasks. 

## Solution
ML .NET does not have a DNN training infrastructure so we plan to use Tensorflow in the backend through the C# bindings created by [Tensorflow .NET](https://github.com/SciSharp/TensorFlow.NET). 

![API layer](dnn_mlnet_layout.JPG)


### Tensorflow transform flow  
![Tensorflow transform flow](dnn_mlnet_transform.JPG)

### Transfer learning  

Using transfer learning we can leverage a well known pre-trained model to create a model that is trained on a smaller custom dataset. Here we use the rich feature extraction capability of the well known model to train a classifier layer that replaces the last layer of the well known model. Based on [this](https://arxiv.org/abs/1805.08974) publication in CVPR 2019 it seems transfer learning gives better results.

The below example shows transfer learning on resnet 101 done using ML .NET tensorflow transform. The picture consists of three graphs. The first graph is the original frozen resnet v2 101 model, the second graph is the same model converted to its meta graph file with transfer learning layer added and connected to the output of its second last layer that contains features. The third graph contains the frozen resnet v2 101 graph with the last layer replaced with the transfer learnt layer model. 

![Transfer learning 1](https://github.com/dotnet/machinelearning/blob/99e628c518303cd73e48da3a9d66ecdba75adf13/docs/specs/DNN/tl_first.jpg)  

![Transfer learning 2](https://github.com/dotnet/machinelearning/blob/99e628c518303cd73e48da3a9d66ecdba75adf13/docs/specs/DNN/tl_second.jpg)  

![Transfer learning 3](https://github.com/dotnet/machinelearning/blob/99e628c518303cd73e48da3a9d66ecdba75adf13/docs/specs/DNN/tl_third.jpg)  

## Scenarios
### Image classification APIs (Preview release in v1.3)
- **Convolutional Neural Network (CNN) Trainer**
  ```C#
    public static TensorFlowModel CNNTrainer(
        this ModelOperationsCatalog catalog,
        string outputColumnName, // Creates the name for the last layer.
        string featuresColumnName, // Input column that will be mapped to input tensor of the graph.
        string labelColumnName, // Label column that will be used for ground truth.
        string outputGraphPath, // Path to the final frozen graph.
        Architecture arch = Architecture.Resnet_V2_101, // Pre-trained model.
        int epoch = 10,
        int batchSize = 20,
        float learningRate = 0.01f,
        bool addBatchDimensionInput = false)
    
    enum Architecture
    {
      Resnet_V2_101,
      Inception_V3
    }
  ```

  Example:

  ```cs
    var mlContext = new MLContext();
    var data = GetTensorData();
    var idv = mlContext.Data.LoadFromEnumerable(data);

    // Create a ML pipeline.
    var pipeline = mlContext.Transforms.Conversion.MapValueToKey(new []{ nameof(TensorData.label) })
                    .Append(mlContext.Model.CNNTrainer(nameof(OutputScores.output), nameof(TensorData.input), nameof(TensorData.label), "myCnnModel.pb")
                      .ScoreTensorFlowModel(new[] { nameof(OutputScores.output) }, new[] { nameof(TensorData.input) }, addBatchDimensionInput: true));

    // Run the pipeline and get the transformed values.
    var estimator = pipeline.Fit(idv);
    var transformedValues = estimator.Transform(idv);

    // Retrieve model scores.
    var outScores = mlContext.Data.CreateEnumerable<OutputScores>(
        transformedValues, reuseRowObject: false);

    // Display scores. (for the sake of brevity we display scores of the
    // first 3 classes)
    foreach (var prediction in outScores)
    {
        int numClasses = 0;
        foreach (var classScore in prediction.output.Take(3))
        {
            Console.WriteLine(
                $"Class #{numClasses++} score = {classScore}");
        }
        Console.WriteLine(new string('-', 10));
    }

    // Results look like below...
    //Class #0 score = -0.8092947
    //Class #1 score = -0.3310375
    //Class #2 score = 0.1119193
    //----------
    //Class #0 score = -0.7807726
    //Class #1 score = -0.2158062
    //Class #2 score = 0.1153686
    //----------

    private const int imageHeight = 224; 
    private const int imageWidth = 224;
    private const int numChannels = 3;
    private const int inputSize = imageHeight * imageWidth * numChannels;

    /// <summary>
    /// A class to hold sample tensor data. 
    /// Member name should match the inputs that the model expects (in this
    /// case, input).
    /// </summary>
    public class TensorData
    {
        [VectorType(imageHeight, imageWidth, numChannels)]
        public float[] input { get; set; }

        string label { get; set; };
    }

    /// <summary>
    /// Method to generate sample test data. Returns 2 sample rows.
    /// </summary>
    public static TensorData[] GetTensorData()
    {
        // This can be any numerical data. Assume image pixel values.
        var image1 = Enumerable.Range(0, inputSize).Select(
            x => (float)x / inputSize).ToArray();
        
        var image2 = Enumerable.Range(0, inputSize).Select(
            x => (float)(x + 10000) / inputSize).ToArray();
        return new TensorData[] { new TensorData() { input = image1 },
            new TensorData() { input = image2 } };
    }

    /// <summary>
    /// Class to contain the output values from the transformation.
    /// </summary>
    class OutputScores
    {
        public float[] output { get; set; }
    }

  ```
  - Fine tune model by unfreezeing last few layers and training on them in addition to transfer learning.
  - Validation and test scores while training.

- **Transfer Learning Trainer**
  ```C#
    public static TensorFlowModel TransferLearning(
        this ModelOperationsCatalog catalog,
        string[] outputColumnNames,
        string[] inputColumnNames,
        string graphMetaFilePath,
        string outputGraphPath,
        string[] cutOperations,
        int[][] cutIndices,
        string[] joinOperation,
        int[][] joinIndices,
        int epoch = 10,
        int batchSize = 20,
        float learningRate = 0.01f,
        bool addBatchDimensionInput = false)
  ```
  - Option for custom *Head* of the model.

- **Model re-trainer**
  ```C#
    public TensorFlowEstimator RetrainModel(
        string[] outputColumnNames,
        string[] inputColumnNames,
        string labelColumnName,
        string tensorFlowLabel,
        string optimizationOperation,
        int epoch = 10,
        int batchSize = 20,
        string lossOperation = null,
        string metricOperation = null,
        string learningRateOperation = null,
        float learningRate = 0.01f,
        bool addBatchDimensionInput = false)
  ```
  
- **Model scorer(already exist)**
  ```C#
    public TensorFlowEstimator ScoreTensorFlowModel(
        string[] outputColumnNames,
        string[] inputColumnNames,
        bool addBatchDimensionInput = false)
  ```
### Text classification
### Object detection