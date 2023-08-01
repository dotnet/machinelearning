// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;
using Microsoft.ML.TorchSharp.Roberta;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal class QuestionAnsweringMulti
    {
        public IEstimator<ITransformer> BuildFromOption(MLContext context, QATrainer.Options param)
        {
            return context.MulticlassClassification.Trainers.QuestionAnswer(
                   contextColumnName: param.ContextColumnName,
                   questionColumnName: param.QuestionColumnName,
                   trainingAnswerColumnName: param.TrainingAnswerColumnName,
                   answerIndexColumnName: param.AnswerIndexStartColumnName,
                   predictedAnswerColumnName: param.PredictedAnswerColumnName,
                   scoreColumnName: param.ScoreColumnName,
                   batchSize: param.BatchSize,
                   maxEpochs: param.MaxEpoch,
                   topK: param.TopKAnswers,
                   architecture: BertArchitecture.Roberta);
        }
    }
}
