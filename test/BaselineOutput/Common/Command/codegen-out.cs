using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace MLGeneratedCode
{
public class Program
{
/// <summary>
/// This is the input to the trained model.
///
/// In most pipelines, not all columns that are used in training are also used in scoring. Namely, the label
/// and weight columns are almost never required at scoring time. Since we don't know which columns
/// are 'optional' in this sense, all the columns are listed below.
///
/// You are free to remove any fields from the below class. If the fields are not required for scoring, the model
/// will continue to work. Otherwise, the exception will be thrown when a prediction engine is created.
///
/// </summary>
public class InputData
{
            public Single Label;

            [VectorType(5)]
            [ColumnName("F!1")]
            public Single[] Column1 = new Single[5];

            [VectorType(4)]
            public Single[] F2 = new Single[4];
}

/// <summary>
/// This is the output of the scored model, the prediction.
///
///</summary>
public class ScoredOutput
{
            public Boolean PredictedLabel;

            public Single Score;

            public Single Probability;
}

/*public static void Main(string[] args)
{
string modelPath;
modelPath = "model.zip";
PredictAsync(modelPath);
}*/

/// <summary>
/// This method demonstrates how to run prediction.
///
///</summary>
public static void Predict(string modelPath)
{
    var model = await PredictionModel.ReadAsync<InputData, ScoredOutput>(modelPath);

    var inputData = new InputData();
    // TODO: populate the example's features.

    var score = model.Predict(inputData);
    // TODO: consume the resulting score.

    var scores = model.Predict(new List<InputData> { inputData, inputData });
    // TODO: consume the resulting scores.
  }
} 
}
