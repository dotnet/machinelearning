// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// Interface for standard predictor model port type.
    /// </summary>
    public interface IPredictorModel
    {
        /// <summary>
        /// Save the model to the given stream.
        /// </summary>
        void Save(IHostEnvironment env, Stream stream);

        /// <summary>
        /// Extract only the transform portion of the predictor model.
        /// </summary>
        ITransformModel TransformModel { get; }

        /// <summary>
        /// Extract the predictor object out of the predictor model.
        /// </summary>
        IPredictor Predictor { get; }

        /// <summary>
        /// Apply the predictor model to the transform model and return the resulting predictor model.
        /// </summary>
        IPredictorModel Apply(IHostEnvironment env, ITransformModel transformModel);

        /// <summary>
        /// For a given input data, return role mapped data and the predictor object.
        /// The scoring entry point will hopefully know how to construct a scorer out of them.
        /// </summary>
        void PrepareData(IHostEnvironment env, IDataView input, out RoleMappedData roleMappedData, out IPredictor predictor);

        /// <summary>
        /// Returns a string array containing the label names of the label column type predictor was trained on.
        /// If the training label is a key with text key value metadata, it should return this metadata. The order of the labels should be consistent
        /// with the key values. Otherwise, it returns null.
        /// </summary>
        /// <param name="env"/>
        /// <param name="labelType">The column type of the label the predictor was trained on.</param>
        string[] GetLabelInfo(IHostEnvironment env, out ColumnType labelType);

        /// <summary>
        /// Returns the RoleMappedSchema that was used in training.
        /// </summary>
        /// <param name="env"></param>
        /// <returns></returns>
        RoleMappedSchema GetTrainingSchema(IHostEnvironment env);
    }
}