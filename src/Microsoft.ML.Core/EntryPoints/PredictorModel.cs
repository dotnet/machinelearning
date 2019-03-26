// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// Base type for standard predictor model port type.
    /// </summary>
    [BestFriend]
    internal abstract class PredictorModel
    {
        [BestFriend]
        private protected PredictorModel()
        {
        }

        /// <summary>
        /// Save the model to the given stream.
        /// </summary>
        [BestFriend]
        internal abstract void Save(IHostEnvironment env, Stream stream);

        /// <summary>
        /// Extract only the transform portion of the predictor model.
        /// </summary>
        [BestFriend]
        internal abstract TransformModel TransformModel { get; }

        /// <summary>
        /// Extract the predictor object out of the predictor model.
        /// </summary>
        [BestFriend]
        internal abstract IPredictor Predictor { get; }

        /// <summary>
        /// Apply the predictor model to the transform model and return the resulting predictor model.
        /// </summary>
        [BestFriend]
        internal abstract PredictorModel Apply(IHostEnvironment env, TransformModel transformModel);

        /// <summary>
        /// For a given input data, return role mapped data and the predictor object.
        /// The scoring entry point will hopefully know how to construct a scorer out of them.
        /// </summary>
        [BestFriend]
        internal abstract void PrepareData(IHostEnvironment env, IDataView input, out RoleMappedData roleMappedData, out IPredictor predictor);

        /// <summary>
        /// Returns a string array containing the label names of the label column type predictor was trained on.
        /// If the training label is a key with text key value annotation, it should return this annotation. The order of the labels should be consistent
        /// with the key values. Otherwise, it returns null.
        /// </summary>
        /// <param name="env"/>
        /// <param name="labelType">The column type of the label the predictor was trained on.</param>
        [BestFriend]
        internal abstract string[] GetLabelInfo(IHostEnvironment env, out DataViewType labelType);

        /// <summary>
        /// Returns the <see cref="RoleMappedSchema"/> that was used in training.
        /// </summary>
        [BestFriend]
        internal abstract RoleMappedSchema GetTrainingSchema(IHostEnvironment env);
    }
}