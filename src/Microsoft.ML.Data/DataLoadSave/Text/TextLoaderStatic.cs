// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Core.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;

namespace Microsoft.ML.Runtime.Data
{
    public sealed partial class TextLoader
    {
        /// <summary>
        /// Configures a reader for text files.
        /// </summary>
        /// <typeparam name="TShape">The type shape parameter, which must be a valid-schema shape. As a practical
        /// matter this is generally not explicitly defined from the user, but is instead inferred from the return
        /// type of the <paramref name="func"/> where one takes an input <see cref="Context"/> and uses it to compose
        /// a shape-type instance describing what the columns are and how to load them from the file.</typeparam>
        /// <param name="env">The environment.</param>
        /// <param name="func">The delegate that describes what fields to read from the text file, as well as
        /// describing their input type. The way in which it works is that the delegate is fed a <see cref="Context"/>,
        /// and the user composes a shape type with <see cref="PipelineColumn"/> instances out of that <see cref="Context"/>.
        /// The resulting data will have columns with the names corresponding to their names in the shape type.</param>
        /// <param name="files">Input files. If <c>null</c> then no files are read, but this means that options or
        /// configurations that require input data for initialization (for example, <paramref name="hasHeader"/> or
        /// <see cref="Context.LoadFloat(int, int?)"/>) with a <c>null</c> second argument.</param>
        /// <param name="hasHeader">Data file has header with feature names.</param>
        /// <param name="separator">Text field separator.</param>
        /// <param name="allowQuoting">Whether the input -may include quoted values, which can contain separator
        /// characters, colons, and distinguish empty values from missing values. When true, consecutive separators
        /// denote a missing value and an empty value is denoted by <c>""</c>. When false, consecutive separators
        /// denote an empty value.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations.</param>
        /// <param name="trimWhitspace">Remove trailing whitespace from lines.</param>
        /// <returns>A configured statically-typed reader for text files.</returns>
        public static DataReader<IMultiStreamSource, TShape> CreateReader<[IsShape] TShape>(
            IHostEnvironment env,  Func<Context, TShape> func, IMultiStreamSource files = null,
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
                return readerEst.Fit(files);
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

        /// <summary>
        /// Context object by which a user can indicate what fields they want to read from a text file, and what data type they ought to have.
        /// Instances of this class are never made but the user, but rather are fed into the delegate in
        /// <see cref="TextLoader.CreateReader{TShape}(IHostEnvironment, Func{Context, TShape}, IMultiStreamSource, bool, char, bool, bool, bool)"/>.
        /// </summary>
        public sealed class Context
        {
            private readonly Reconciler _rec;

            internal Context(Reconciler rec)
            {
                Contracts.AssertValue(rec);
                _rec = rec;
            }

            /// <summary>
            /// Reads a scalar Boolean column from a single field in the text file.
            /// </summary>
            /// <param name="ordinal">The zero-based index of the field to read from.</param>
            /// <returns>The column representation.</returns>
            public Scalar<bool> LoadBool(int ordinal) => Load<bool>(DataKind.BL, ordinal);

            /// <summary>
            /// Reads a vector Boolean column from a range of fields in the text file.
            /// </summary>
            /// <param name="minOrdinal">The zero-based inclusive lower index of the field to read from.</param>
            /// <param name="maxOrdinal">The zero-based inclusive upper index of the field to read from.
            /// Note that if this is <c>null</c>, it will read to the end of the line. The file(s)
            /// will be inspected to get the length of the type.</param>
            /// <returns>The column representation.</returns>
            public Vector<bool> LoadBool(int minOrdinal, int? maxOrdinal) => Load<bool>(DataKind.BL, minOrdinal, maxOrdinal);

            /// <summary>
            /// Reads a scalar single-precision floating point column from a single field in the text file.
            /// </summary>
            /// <param name="ordinal">The zero-based index of the field to read from.</param>
            /// <returns>The column representation.</returns>
            public Scalar<float> LoadFloat(int ordinal) => Load<float>(DataKind.R4, ordinal);

            /// <summary>
            /// Reads a vector single-precision column from a range of fields in the text file.
            /// </summary>
            /// <param name="minOrdinal">The zero-based inclusive lower index of the field to read from.</param>
            /// <param name="maxOrdinal">The zero-based inclusive upper index of the field to read from.
            /// Note that if this is <c>null</c>, it will read to the end of the line. The file(s)
            /// will be inspected to get the length of the type.</param>
            /// <returns>The column representation.</returns>
            public Vector<float> LoadFloat(int minOrdinal, int? maxOrdinal) => Load<float>(DataKind.R4, minOrdinal, maxOrdinal);

            /// <summary>
            /// Reads a scalar double-precision floating point column from a single field in the text file.
            /// </summary>
            /// <param name="ordinal">The zero-based index of the field to read from.</param>
            /// <returns>The column representation.</returns>
            public Scalar<double> LoadDouble(int ordinal) => Load<double>(DataKind.R8, ordinal);

            /// <summary>
            /// Reads a vector double-precision column from a range of fields in the text file.
            /// </summary>
            /// <param name="minOrdinal">The zero-based inclusive lower index of the field to read from.</param>
            /// <param name="maxOrdinal">The zero-based inclusive upper index of the field to read from.
            /// Note that if this is <c>null</c>, it will read to the end of the line. The file(s)
            /// will be inspected to get the length of the type.</param>
            /// <returns>The column representation.</returns>
            public Vector<double> LoadDouble(int minOrdinal, int? maxOrdinal) => Load<double>(DataKind.R8, minOrdinal, maxOrdinal);

            /// <summary>
            /// Reads a scalar textual column from a single field in the text file.
            /// </summary>
            /// <param name="ordinal">The zero-based index of the field to read from.</param>
            /// <returns>The column representation.</returns>
            public Scalar<string> LoadText(int ordinal) => Load<string>(DataKind.TX, ordinal);

            /// <summary>
            /// Reads a vector textual column from a range of fields in the text file.
            /// </summary>
            /// <param name="minOrdinal">The zero-based inclusive lower index of the field to read from.</param>
            /// <param name="maxOrdinal">The zero-based inclusive upper index of the field to read from.
            /// Note that if this is <c>null</c>, it will read to the end of the line. The file(s)
            /// will be inspected to get the length of the type.</param>
            /// <returns>The column representation.</returns>
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

    public static class LocalPathReader
    {
        public static IDataView Read(this IDataReader<IMultiStreamSource> reader, params string[] path)
        {
            return reader.Read(new MultiFileSource(path));
        }

        public static DataView<TShape> Read<TShape>(this DataReader<IMultiStreamSource, TShape> reader, params string[] path)
        {
            return reader.Read(new MultiFileSource(path));
        }
    }
}