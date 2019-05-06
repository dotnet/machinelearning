// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.StaticPipe
{
    public static class TextLoaderStatic
    {
        /// <summary>
        /// Configures a loader for text files.
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
        /// <param name="separator">Text field separator.</param>
        /// <param name="hasHeader">Data file has header with feature names.</param>
        /// <param name="allowQuoting">Whether the input -may include quoted values, which can contain separator
        /// characters, colons, and distinguish empty values from missing values. When true, consecutive separators
        /// denote a missing value and an empty value is denoted by <c>""</c>. When false, consecutive separators
        /// denote an empty value.</param>
        /// <param name="allowSparse">Whether the input may include sparse representations.</param>
        /// <param name="trimWhitspace">Remove trailing whitespace from lines.</param>
        /// <returns>A configured statically-typed loader for text files.</returns>
        public static DataLoader<IMultiStreamSource, TShape> CreateLoader<[IsShape] TShape>(
            IHostEnvironment env, Func<Context, TShape> func, IMultiStreamSource files = null,
            char separator = '\t', bool hasHeader = false, bool allowQuoting = true, bool allowSparse = true,
            bool trimWhitspace = false)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(func, nameof(func));
            env.CheckValueOrNull(files);

            // Populate all args except the columns.
            var args = new TextLoader.Options();
            args.AllowQuoting = allowQuoting;
            args.AllowSparse = allowSparse;
            args.HasHeader = hasHeader;
            args.Separators = new[] { separator };
            args.TrimWhitespace = trimWhitspace;

            var rec = new TextReconciler(args, files);
            var ctx = new Context(rec);

            using (var ch = env.Start("Initializing " + nameof(TextLoader)))
            {
                var loaderEst = StaticPipeUtils.LoaderEstimatorAnalyzerHelper(env, ch, ctx, rec, func);
                Contracts.AssertValue(loaderEst);
                return loaderEst.Fit(files);
            }
        }

        private sealed class TextReconciler : LoaderReconciler<IMultiStreamSource>
        {
            private readonly TextLoader.Options _args;
            private readonly IMultiStreamSource _files;

            public TextReconciler(TextLoader.Options options, IMultiStreamSource files)
            {
                Contracts.AssertValue(options);
                Contracts.AssertValueOrNull(files);

                _args = options;
                _files = files;
            }

            public override IDataLoaderEstimator<IMultiStreamSource, IDataLoader<IMultiStreamSource>> Reconcile(
                IHostEnvironment env, PipelineColumn[] toOutput, IReadOnlyDictionary<PipelineColumn, string> outputNames)
            {
                Contracts.AssertValue(env);
                Contracts.AssertValue(toOutput);
                Contracts.AssertValue(outputNames);
                Contracts.Assert(_args.Columns == null);

                TextLoader.Column Create(PipelineColumn pipelineCol)
                {
                    var pipelineArgCol = (IPipelineArgColumn)pipelineCol;
                    var argCol = pipelineArgCol.Create();
                    argCol.Name = outputNames[pipelineCol];
                    return argCol;
                }

                var cols = _args.Columns = new TextLoader.Column[toOutput.Length];
                for (int i = 0; i < toOutput.Length; ++i)
                    cols[i] = Create(toOutput[i]);

                var orig = new TextLoader(env, _args, _files);
                return new TrivialLoaderEstimator<IMultiStreamSource, TextLoader>(orig);
            }
        }

        private interface IPipelineArgColumn
        {
            /// <summary>
            /// Creates a <see cref="TextLoader.Column"/> object corresponding to the <see cref="PipelineColumn"/>, with everything
            /// filled in except <see cref="TextLoader.ColInfo.Name"/>.
            /// </summary>
            TextLoader.Column Create();
        }

        /// <summary>
        /// Context object by which a user can indicate what fields they want to read from a text file, and what data type they ought to have.
        /// Instances of this class are never made but the user, but rather are fed into the delegate in
        /// <see cref="CreateLoader{TShape}(IHostEnvironment, Func{Context, TShape}, IMultiStreamSource, char, bool, bool, bool, bool)"/>.
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
            public Scalar<bool> LoadBool(int ordinal) => Load<bool>(InternalDataKind.BL, ordinal);

            /// <summary>
            /// Reads a vector Boolean column from a range of fields in the text file.
            /// </summary>
            /// <param name="minOrdinal">The zero-based inclusive lower index of the field to read from.</param>
            /// <param name="maxOrdinal">The zero-based inclusive upper index of the field to read from.
            /// Note that if this is <c>null</c>, it will read to the end of the line. The file(s)
            /// will be inspected to get the length of the type.</param>
            /// <returns>The column representation.</returns>
            public Vector<bool> LoadBool(int minOrdinal, int? maxOrdinal) => Load<bool>(InternalDataKind.BL, minOrdinal, maxOrdinal);

            /// <summary>
            /// Create a representation for a key loaded from TextLoader as an unsigned integer (32 bits).
            /// </summary>
            /// <param name="ordinal">The zero-based index of the field to read from.</param>
            /// <param name="keyCount">If specified, it's the count or cardinality of valid key values.
            /// Using null initalizes <paramref name="keyCount"/> to uint.MaxValue</param>
            /// <returns>The column representation.</returns>
            public Key<uint> LoadKey(int ordinal, ulong? keyCount) => Load<uint>(InternalDataKind.U4, ordinal, keyCount);

            /// <summary>
            /// Reads a scalar single-precision floating point column from a single field in the text file.
            /// </summary>
            /// <param name="ordinal">The zero-based index of the field to read from.</param>
            /// <returns>The column representation.</returns>
            public Scalar<float> LoadFloat(int ordinal) => Load<float>(InternalDataKind.R4, ordinal);

            /// <summary>
            /// Reads a vector single-precision column from a range of fields in the text file.
            /// </summary>
            /// <param name="minOrdinal">The zero-based inclusive lower index of the field to read from.</param>
            /// <param name="maxOrdinal">The zero-based inclusive upper index of the field to read from.
            /// Note that if this is <c>null</c>, it will read to the end of the line. The file(s)
            /// will be inspected to get the length of the type.</param>
            /// <returns>The column representation.</returns>
            public Vector<float> LoadFloat(int minOrdinal, int? maxOrdinal) => Load<float>(InternalDataKind.R4, minOrdinal, maxOrdinal);

            /// <summary>
            /// Reads a scalar double-precision floating point column from a single field in the text file.
            /// </summary>
            /// <param name="ordinal">The zero-based index of the field to read from.</param>
            /// <returns>The column representation.</returns>
            public Scalar<double> LoadDouble(int ordinal) => Load<double>(InternalDataKind.R8, ordinal);

            /// <summary>
            /// Reads a vector double-precision column from a range of fields in the text file.
            /// </summary>
            /// <param name="minOrdinal">The zero-based inclusive lower index of the field to read from.</param>
            /// <param name="maxOrdinal">The zero-based inclusive upper index of the field to read from.
            /// Note that if this is <c>null</c>, it will read to the end of the line. The file(s)
            /// will be inspected to get the length of the type.</param>
            /// <returns>The column representation.</returns>
            public Vector<double> LoadDouble(int minOrdinal, int? maxOrdinal) => Load<double>(InternalDataKind.R8, minOrdinal, maxOrdinal);

            /// <summary>
            /// Reads a scalar textual column from a single field in the text file.
            /// </summary>
            /// <param name="ordinal">The zero-based index of the field to read from.</param>
            /// <returns>The column representation.</returns>
            public Scalar<string> LoadText(int ordinal) => Load<string>(InternalDataKind.TX, ordinal);

            /// <summary>
            /// Reads a vector textual column from a range of fields in the text file.
            /// </summary>
            /// <param name="minOrdinal">The zero-based inclusive lower index of the field to read from.</param>
            /// <param name="maxOrdinal">The zero-based inclusive upper index of the field to read from.
            /// Note that if this is <c>null</c>, it will read to the end of the line. The file(s)
            /// will be inspected to get the length of the type.</param>
            /// <returns>The column representation.</returns>
            public Vector<string> LoadText(int minOrdinal, int? maxOrdinal) => Load<string>(InternalDataKind.TX, minOrdinal, maxOrdinal);

            private Scalar<T> Load<T>(InternalDataKind kind, int ordinal)
            {
                Contracts.CheckParam(ordinal >= 0, nameof(ordinal), "Should be non-negative");
                return new MyScalar<T>(_rec, kind, ordinal);
            }

            private Vector<T> Load<T>(InternalDataKind kind, int minOrdinal, int? maxOrdinal)
            {
                Contracts.CheckParam(minOrdinal >= 0, nameof(minOrdinal), "Should be non-negative");
                var v = maxOrdinal >= minOrdinal;
                Contracts.CheckParam(!(maxOrdinal < minOrdinal), nameof(maxOrdinal), "If specified, cannot be less than " + nameof(minOrdinal));
                return new MyVector<T>(_rec, kind, minOrdinal, maxOrdinal);
            }

            private Key<T> Load<T>(InternalDataKind kind, int ordinal, ulong? keyCount)
            {
                Contracts.CheckParam(ordinal >= 0, nameof(ordinal), "Should be non-negative");
                return new MyKey<T>(_rec, kind, ordinal, keyCount);
            }

            /// <summary>
            /// A data type used to bridge <see cref="PipelineColumn"/> and <see cref="TextLoader.Column"/>. It can be used as <see cref="PipelineColumn"/>
            /// in static-typed pipelines and provides <see cref="MyKey{T}.Create"/> for translating itself into <see cref="TextLoader.Column"/>.
            /// </summary>
            private class MyKey<T> : Key<T>, IPipelineArgColumn
            {
                // The storage type that the targeted content would be loaded as.
                private readonly InternalDataKind _kind;
                // The position where the key value gets read from.
                private readonly int _oridinal;
                // The count or cardinality of valid key values. Its value is null if unbounded.
                private readonly ulong? _keyCount;

                // Contstuct a representation for a key-typed column loaded from a text file. Key values are assumed to be contiguous.
                public MyKey(Reconciler rec, InternalDataKind kind, int oridinal, ulong? keyCount=null)
                    : base(rec, null)
                {
                    _kind = kind;
                    _oridinal = oridinal;
                    _keyCount = keyCount;
                }

                // Translate the internal variable representation to columns of TextLoader.
                public TextLoader.Column Create()
                {
                    return new TextLoader.Column()
                    {
                        Type = _kind,
                        Source = new[] { new TextLoader.Range(_oridinal) },
                        KeyCount = _keyCount.HasValue ? new KeyCount(_keyCount.GetValueOrDefault()) : new KeyCount()
                    };
                }
            }

            private class MyScalar<T> : Scalar<T>, IPipelineArgColumn
            {
                private readonly InternalDataKind _kind;
                private readonly int _ordinal;

                public MyScalar(Reconciler rec, InternalDataKind kind, int ordinal)
                    : base(rec, null)
                {
                    _kind = kind;
                    _ordinal = ordinal;
                }

                public TextLoader.Column Create()
                {
                    return new TextLoader.Column()
                    {
                        Type = _kind,
                        Source = new[] { new TextLoader.Range(_ordinal) },
                    };
                }
            }

            private class MyVector<T> : Vector<T>, IPipelineArgColumn
            {
                private readonly InternalDataKind _kind;
                private readonly int _min;
                private readonly int? _max;

                public MyVector(Reconciler rec, InternalDataKind kind, int min, int? max)
                    : base(rec, null)
                {
                    _kind = kind;
                    _min = min;
                    _max = max;
                }

                public TextLoader.Column Create()
                {
                    return new TextLoader.Column()
                    {
                        Type = _kind,
                        Source = new[] { new TextLoader.Range(_min, _max) },
                    };
                }
            }
        }
    }
}