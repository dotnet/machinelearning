using Float = System.Single;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Core.Data;

namespace Microsoft.ML.Runtime.Data
{
    public sealed partial class TextLoader
    {
        private sealed class TextReconciler : ReaderReconciler<IMultiStreamSource>
        {
            public static readonly TextReconciler Inst = new TextReconciler();

            public override IDataReaderEstimator<IMultiStreamSource, IDataReader<IMultiStreamSource>> Reconcile(
                PipelineColumn[] toOutput, Dictionary<PipelineColumn, string> outputNames)
            {
                //return new FakeReaderEstimator<IMultiStreamSource>();
                return null;
            }
        }

        public sealed class Context
        {
            private class MyScalar<T> : Scalar<T>
            {
                public readonly int Ordinal;

                public MyScalar(int ordinal)
                    : base(TextReconciler.Inst, null)
                {
                    Ordinal = ordinal;
                }
            }

            private class MyVector<T> : Vector<T>
            {
                public readonly int? Min;
                public readonly int? Max;

                public MyVector(int? min, int? max)
                    : base(TextReconciler.Inst, null)
                {
                    Min = min;
                    Max = max;
                }
            }

            public Scalar<bool> LoadBool(int ordinal) => Load<bool>(ordinal);
            public Vector<bool> LoadBool(int minOrdinal, int? maxOrdinal) => Load<bool>(minOrdinal, maxOrdinal);
            public Scalar<float> LoadFloat(int ordinal) => Load<float>(ordinal);
            public Vector<float> LoadFloat(int minOrdinal, int? maxOrdinal) => Load<float>(minOrdinal, maxOrdinal);
            public Scalar<double> LoadDouble(int ordinal) => Load<double>(ordinal);
            public Vector<double> LoadDouble(int minOrdinal, int? maxOrdinal) => Load<double>(minOrdinal, maxOrdinal);
            public Scalar<string> LoadText(int ordinal) => Load<string>(ordinal);
            public Vector<string> LoadText(int minOrdinal, int? maxOrdinal) => Load<string>(minOrdinal, maxOrdinal);

            private Scalar<T> Load<T>(int ordinal)
            {
                Contracts.CheckParam(ordinal >= 0, nameof(ordinal), "Should be non-negative");
                return new MyScalar<T>(ordinal);
            }

            private Vector<T> Load<T>(int minOrdinal, int? maxOrdinal)
            {
                Contracts.CheckParam(minOrdinal >= 0, nameof(minOrdinal), "Should be non-negative");
                var v = maxOrdinal >= minOrdinal;
                Contracts.CheckParam(!(maxOrdinal < minOrdinal), nameof(maxOrdinal), "If specified, cannot be less than " + nameof(minOrdinal));
                return new MyVector<T>(minOrdinal, maxOrdinal);
            }
        }
    }
}
