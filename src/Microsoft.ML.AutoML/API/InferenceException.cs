// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.AutoML
{
    /// <summary>
    /// Type of exception encountered by AutoML.
    /// </summary>
    public enum InferenceExceptionType
    {
        /// <summary>
        /// Exception that occurs when AutoML is inferring the data type of a column.
        /// </summary>
        ColumnDataType,

        /// <summary>
        /// Exception that occurs when AutoML is attempting to split a dataset into distinct columns.
        /// </summary>
        ColumnSplit,
    }

    /// <summary>
    /// Exception thrown by AutoML.
    /// </summary>
    public sealed class InferenceException : Exception
    {
        /// <summary>
        /// Type of AutoML exception that occurred.
        /// </summary>
        public InferenceExceptionType InferenceExceptionType;

        internal InferenceException(InferenceExceptionType inferenceType, string message)
        : base(message)
        {
        }

        internal InferenceException(InferenceExceptionType inferenceType, string message, Exception inner)
            : base(message, inner)
        {
        }
    }

}
