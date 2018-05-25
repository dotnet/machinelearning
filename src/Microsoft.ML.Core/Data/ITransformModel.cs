// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// Interface for standard transform model port type.
    /// </summary>
    public interface ITransformModel
    {
        /// <summary>
        /// The input schema that this transform model was originally instantiated on.
        /// Note that the schema may have columns that aren't needed by this transform model.
        /// If an IDataView exists with this schema, then applying this transform model to it
        /// shouldn't fail because of column type issues.
        /// REVIEW: Would be nice to be able to trim this to the minimum needed somehow. Note
        /// however that doing so may cause issues for composing transform models. For example,
        /// if transform model A needs column X and model B needs Y, that is NOT produced by A,
        /// then trimming A's input schema would cause composition to fail.
        /// </summary>
        ISchema InputSchema { get; }

        /// <summary>
        /// The resulting schema once applied to this model. The <see cref="InputSchema"/> might have
        /// columns that are not needed by this transform and these columns will be seen in the 
        /// <see cref="OutputSchema"/> produced by this transform.
        /// </summary>
        ISchema OutputSchema { get; }

        /// <summary>
        /// Apply the transform(s) in the model to the given input data.
        /// </summary>
        IDataView Apply(IHostEnvironment env, IDataView input);

        /// <summary>
        /// Apply the transform(s) in the model to the given transform model.
        /// </summary>
        ITransformModel Apply(IHostEnvironment env, ITransformModel input);

        /// <summary>
        /// Save the model to the given stream.
        /// </summary>
        void Save(IHostEnvironment env, Stream stream);

        /// <summary>
        /// Returns the transform model as an <see cref="IRowToRowMapper"/> that can output a row
        /// given a row with the same schema as <see cref="InputSchema"/>.
        /// </summary>
        /// <returns>The transform model as an <see cref="IRowToRowMapper"/>. If not all transforms
        /// in the pipeline are <see cref="IRowToRowMapper"/> then it returns null.</returns>
        IRowToRowMapper AsRowToRowMapper(IExceptionContext ectx);
    }
}
