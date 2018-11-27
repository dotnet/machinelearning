﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using System;

namespace Microsoft.ML.Legacy
{
    /// <summary>
    /// An item that can be added to the Learning Pipeline.
    /// </summary>
    [Obsolete]
    public interface ILearningPipelineItem
    {
        ILearningPipelineStep ApplyStep(ILearningPipelineStep previousStep, Experiment experiment);

        /// <summary>
        /// Returns the place holder for input IDataView object for the node in the execution graph.
        /// </summary>
        /// <returns></returns>
        Var<IDataView> GetInputData();
    }

    /// <summary>
    /// A data loader that can be added to the Learning Pipeline.
    /// </summary>
    [Obsolete]
    public interface ILearningPipelineLoader : ILearningPipelineItem
    {
        void SetInput(IHostEnvironment environment, Experiment experiment);
    }

    /// <summary>
    /// An item that can be added to the Learning Pipeline that can be trained and or return a IDataView.
    /// This encapsulates an IDataView(input) and ITransformModel(output) object for a transform and
    /// for a learner it will encapsulate IDataView(input) and IPredictorModel(output).
    /// </summary>
    [Obsolete]
    public interface ILearningPipelineStep
    {
    }

    [Obsolete]
    public interface ILearningPipelineDataStep : ILearningPipelineStep
    {
        Var<IDataView> Data { get; }
        Var<ITransformModel> Model { get; }
    }

    [Obsolete]
    public interface ILearningPipelinePredictorStep : ILearningPipelineStep
    {
        Var<IPredictorModel> Model { get; }
    }
}