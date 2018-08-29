// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.StaticPipe.Runtime;
using Microsoft.ML.Data.StaticPipe;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    public sealed partial class TextLoader
    {
        public static DataReader<IMultiStreamSource, TTupleShape> CreateReader<TTupleShape>(
            IHostEnvironment env, Func<Context, TTupleShape> func, IMultiStreamSource files = null,
            bool hasHeader = false, char separator = '\t', bool allowQuoting = true, bool allowSparse = true,
            bool trimWhitspace = false)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(func, nameof(func));
            env.CheckValueOrNull(files);

            // Populate all args except the columns.
            var args = new Arguments();
            args.AllowQuoting = allowQuoting;
            args.AllowSparse = allowSparse;
            args.HasHeader = hasHeader;
            args.SeparatorChars = new[] { separator };
            args.TrimWhitespace = trimWhitspace;

            var rec = new TextReconciler(args, files);
            var ctx = new Context(rec);

            using (var ch = env.Start("Initializing " + nameof(TextLoader)))
            {
                var readerEst = StaticPipeUtils.ReaderEstimatorAnalyzerHelper(env, ch, ctx, rec, func);
                Contracts.AssertValue(readerEst);
                var reader = readerEst.Fit(files);
                ch.Done();
                return reader;
            }
        }

        private sealed class TextReconciler : ReaderReconciler<IMultiStreamSource>
        {
            private readonly Arguments _args;
            private readonly IMultiStreamSource _files;

            public TextReconciler(Arguments args, IMultiStreamSource files)
            {
                Contracts.AssertValue(args);
                Contracts.AssertValueOrNull(files);

                _args = args;
                _files = files;
            }

            public override IDataReaderEstimator<IMultiStreamSource, IDataReader<IMultiStreamSource>> Reconcile(
                IHostEnvironment env, PipelineColumn[] toOutput, IReadOnlyDictionary<PipelineColumn, string> outputNames)
            {
                Contracts.AssertValue(env);
                Contracts.AssertValue(toOutput);
                Contracts.AssertValue(outputNames);
                Contracts.Assert(_args.Column == null);

                Column Create(PipelineColumn pipelineCol)
                {
                    var pipelineArgCol = (IPipelineArgColumn)pipelineCol;
                    var argCol = pipelineArgCol.Create();
                    argCol.Name = outputNames[pipelineCol];
                    return argCol;
                }

                var cols = _args.Column = new Column[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                    cols[i] = Create(toOutput[i]);

                var orig = new TextLoader(env, _args, _files);
                return new TrivialReaderEstimator<IMultiStreamSource, TextLoader>(orig);
            }
        }

        private interface IPipelineArgColumn
        {
            /// <summary>
            /// Creates a <see cref="Column"/> object corresponding to the <see cref="PipelineColumn"/>, with everything
            /// filled in except <see cref="ColInfo.Name"/>.
            /// </summary>
            Column Create();
        }

        public sealed class Context
        {
            private readonly Reconciler _rec;

            internal Context(Reconciler rec)
            {
                Contracts.AssertValue(rec);
                _rec = rec;
            }

            public Scalar<bool> LoadBool(int ordinal) => Load<bool>(DataKind.BL, ordinal);
            public Vector<bool> LoadBool(int minOrdinal, int? maxOrdinal) => Load<bool>(DataKind.BL, minOrdinal, maxOrdinal);
            public Scalar<float> LoadFloat(int ordinal) => Load<float>(DataKind.R4, ordinal);
            public Vector<float> LoadFloat(int minOrdinal, int? maxOrdinal) => Load<float>(DataKind.R4, minOrdinal, maxOrdinal);
            public Scalar<double> LoadDouble(int ordinal) => Load<double>(DataKind.R8, ordinal);
            public Vector<double> LoadDouble(int minOrdinal, int? maxOrdinal) => Load<double>(DataKind.R8, minOrdinal, maxOrdinal);
            public Scalar<string> LoadText(int ordinal) => Load<string>(DataKind.TX, ordinal);
            public Vector<string> LoadText(int minOrdinal, int? maxOrdinal) => Load<string>(DataKind.TX, minOrdinal, maxOrdinal);

            private Scalar<T> Load<T>(DataKind kind, int ordinal)
            {
                Contracts.CheckParam(ordinal >= 0, nameof(ordinal), "Should be non-negative");
                return new MyScalar<T>(_rec, kind, ordinal);
            }

            private Vector<T> Load<T>(DataKind kind, int minOrdinal, int? maxOrdinal)
            {
                Contracts.CheckParam(minOrdinal >= 0, nameof(minOrdinal), "Should be non-negative");
                var v = maxOrdinal >= minOrdinal;
                Contracts.CheckParam(!(maxOrdinal < minOrdinal), nameof(maxOrdinal), "If specified, cannot be less than " + nameof(minOrdinal));
                return new MyVector<T>(_rec, kind, minOrdinal, maxOrdinal);
            }

            private class MyScalar<T> : Scalar<T>, IPipelineArgColumn
            {
                private readonly DataKind _kind;
                private readonly int _ordinal;

                public MyScalar(Reconciler rec, DataKind kind, int ordinal)
                    : base(rec, null)
                {
                    _kind = kind;
                    _ordinal = ordinal;
                }

                public Column Create()
                {
                    return new Column()
                    {
                        Type = _kind,
                        Source = new[] { new Range(_ordinal) },
                    };
                }
            }

            private class MyVector<T> : Vector<T>, IPipelineArgColumn
            {
                private readonly DataKind _kind;
                private readonly int _min;
                private readonly int? _max;

                public MyVector(Reconciler rec, DataKind kind, int min, int? max)
                    : base(rec, null)
                {
                    _kind = kind;
                    _min = min;
                    _max = max;
                }

                public Column Create()
                {
                    return new Column()
                    {
                        Type = _kind,
                        Source = new[] { new Range(_min, _max) },
                    };
                }
            }
        }
    }
}

