using System;

namespace Microsoft.ML.Auto
{
    public enum InferenceType
    {
        Seperator,
        Header,
        Label,
        Task,
        ColumnDataKind,
        ColumnPurpose,
        Tranform,
        Trainer,
        Hyperparams,
        ColumnSplit
    }

    public class InferenceException : Exception
    {
        public InferenceType InferenceType;

        public InferenceException(InferenceType inferenceType, string message)
        : base(message)
        {
        }

        public InferenceException(InferenceType inferenceType, string message, Exception inner)
            : base(message, inner)
        {
        }
    }

}
