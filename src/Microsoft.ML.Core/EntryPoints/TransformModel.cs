// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// Interface for standard transform model port type.
    /// </summary>
    [BestFriend]
    internal abstract class TransformModel
    {
        [BestFriend]
        private protected TransformModel()
        {
        }

        /// <summary>
        /// The input schema that this transform model was originally instantiated on.
        /// Note that the schema may have columns that aren't needed by this transform model.
        /// If an <see cref="IDataView"/> exists with this schema, then applying this transform model to it
        /// shouldn't fail because of column type issues.
        /// </summary>
        // REVIEW: Would be nice to be able to trim this to the minimum needed somehow. Note
        // however that doing so may cause issues for composing transform models. For example,
        // if transform model A needs column X and model B needs Y, that is NOT produced by A,
        // then trimming A's input schema would cause composition to fail.
        [BestFriend]
        internal abstract DataViewSchema InputSchema { get; }

        /// <summary>
        /// The output schema that this transform model was originally instantiated on. The schema resulting
        /// from <see cref="Apply(IHostEnvironment, TransformModel)"/> may differ from this, similarly to how
        /// <see cref="InputSchema"/> may differ from the schema of dataviews we apply this transform model to.
        /// </summary>
        [BestFriend]
        internal abstract DataViewSchema OutputSchema { get; }

        /// <summary>
        /// Apply the transform(s) in the model to the given input data.
        /// </summary>
        [BestFriend]
        internal abstract IDataView Apply(IHostEnvironment env, IDataView input);

        /// <summary>
        /// Apply the transform(s) in the model to the given transform model.
        /// </summary>
        [BestFriend]
        internal abstract TransformModel Apply(IHostEnvironment env, TransformModel input);

        /// <summary>
        /// Save the model to the given stream.
        /// </summary>
        [BestFriend]
        internal abstract void Save(IHostEnvironment env, Stream stream);

        /// <summary>
        /// Returns the transform model as an <see cref="IRowToRowMapper"/> that can output a row
        /// given a row with the same schema as <see cref="InputSchema"/>.
        /// </summary>
        /// <returns>The transform model as an <see cref="IRowToRowMapper"/>. If not all transforms
        /// in the pipeline are <see cref="IRowToRowMapper"/> then it returns <see langword="null"/>.</returns>
        [BestFriend]
        internal abstract IRowToRowMapper AsRowToRowMapper(IExceptionContext ectx);
    }
}
