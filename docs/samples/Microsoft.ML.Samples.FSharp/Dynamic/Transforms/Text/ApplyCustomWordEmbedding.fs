module Microsoft.ML.Samples.Dynamic.ApplyCustomWordEmbeddings
open Microsoft.ML
open Microsoft.ML.Data
open System
open System.IO

type TextData(value:string) = 
    member val Text = value with get,set
    new() = TextData("")

type TransformedTextData(value: string) = 
    inherit TextData(value)
    member val Features:float32 array = Array.empty with get, set

    new() = TransformedTextData("")

let Example = 
    // Create a new ML context, for ML.NET operations. It can be used for exception tracking and logging, 
    // as well as the source of randomness.
    let mlContext = MLContext()     

    // Create an empty data sample list. The 'ApplyWordEmbedding' does not require training data as
    // the estimator ('WordEmbeddingEstimator') created by 'ApplyWordEmbedding' API is not a trainable estimator.
    // The empty list is only needed to pass input schema to the pipeline.
    let emptySamples : TextData List = []

    // Convert sample list to an empty IDataView.
    let emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples)

    // Write a custom 3-dimensional word embedding model with 4 words.
    // Each line follows '<word> <float> <float> <float>' pattern.
    // Lines that do not confirm to the pattern are ignored.
    let writeFile(path:string) =
        use writer = new StreamWriter(path, false)
        writer.WriteLine("great 1.0 2.0 3.0");
        writer.WriteLine("product -1.0 -2.0 -3.0");
        writer.WriteLine("like -1 100.0 -100");
        writer.WriteLine("buy 0 0 20");

    let pathToCustomModel = @".\custommodel.txt"
    writeFile(pathToCustomModel)

    // A pipeline for converting text into a 9-dimension word embedding vector using the custom word embedding model.
    // The 'ApplyWordEmbedding' computes the minimum, average and maximum values for each token's embedding vector.
    // Tokens in 'custommodel.txt' model are represented as 3-dimension vector.
    // Therefore, the output is of 9-dimension [min, avg, max].
    //
    // The 'ApplyWordEmbedding' API requires vector of text as input.
    // The pipeline first normalizes and tokenizes text then applies word embedding transformation. 
    let textPipeline = 
        EstimatorChain()
            .Append(mlContext.Transforms.Text.NormalizeText("Text"))
            .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))  
            .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Features", pathToCustomModel, "Tokens" ))

    // Fit to data.
    let textTransformer = textPipeline.Fit(emptyDataView)

    // Create the prediction engine to get the embedding vector from the input text/string.
    let predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer)

    // Call the prediction API to convert the text into embedding vector.
    let data = new TextData("This is a great product. I would like to buy it again.");
    let prediction = predictionEngine.Predict(data);

    // Print the length of the embedding vector.
    Console.WriteLine("Number of Features: {0}", prediction.Features.Length)

    // Print the embedding vector.
    Console.Write("Features: ");
    for i = 0 to prediction.Features.Length - 1 do
        Console.Write("{0:F4} ", prediction.Features.[i])

    //  Expected output:
    //   Number of Features: 9
    //   Features: -1.0000 0.0000 -100.0000 0.0000 34.0000 -25.6667 1.0000 100.0000 20.0000



