// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class NamedEntityRecognitionMulti
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, NamedEntityRecognitionOption param)
        {
            return context.MulticlassClassification.Trainers.NamedEntityRecognition(
                   labelColumnName: param.LabelColumnName,
                   outputColumnName: param.PredictionColumnName,
                   sentence1ColumnName: param.Sentence1ColumnName,
                   batchSize: param.BatchSize,
                   maxEpochs: param.MaxEpochs,
                   architecture: BertArchitecture.Roberta);
        }
    }
}
