// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// This applies the user provided RefPredicate to a column and drops rows that map to false. It automatically
    /// injects a standard conversion from the actual type of the source column to typeSrc (if needed).
    /// </summary>
    internal static class LambdaFilter
    {
        public static IDataView Create<TSrc>(IHostEnvironment env, string name, IDataView input,
            string src, DataViewType typeSrc, InPredicate<TSrc> predicate)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(name, nameof(name));
            env.CheckValue(input, nameof(input));
            env.CheckNonEmpty(src, nameof(src));
            env.CheckValue(typeSrc, nameof(typeSrc));
            env.CheckValue(predicate, nameof(predicate));

            if (typeSrc.RawType != typeof(TSrc))
            {
                throw env.ExceptParam(nameof(predicate),
                    "The source column type '{0}' doesn't match the input type of the predicate", typeSrc);
            }

            int colSrc;
            bool tmp = input.Schema.TryGetColumnIndex(src, out colSrc);
            if (!tmp)
                throw env.ExceptParam(nameof(src), "The input data doesn't have a column named '{0}'", src);
            var typeOrig = input.Schema[colSrc].Type;

            // REVIEW: Ideally this should support vector-type conversion. It currently doesn't.
            bool ident;
            Delegate conv;
            if (typeOrig.SameSizeAndItemType(typeSrc))
            {
                ident = true;
                conv = null;
            }
            else if (!Conversions.Instance.TryGetStandardConversion(typeOrig, typeSrc, out conv, out ident))
            {
                throw env.ExceptParam(nameof(predicate),
                    "The type of column '{0}', '{1}', cannot be converted to the input type of the predicate '{2}'",
                    src, typeOrig, typeSrc);
            }

            IDataView impl;
            if (ident)
                impl = new Impl<TSrc, TSrc>(env, name, input, colSrc, predicate);
            else
            {
                Func<IHostEnvironment, string, IDataView, int,
                    InPredicate<int>, ValueMapper<int, int>, Impl<int, int>> del = CreateImpl<int, int>;
                var meth = del.GetMethodInfo().GetGenericMethodDefinition()
                    .MakeGenericMethod(typeOrig.RawType, typeof(TSrc));
                impl = (IDataView)meth.Invoke(null, new object[] { env, name, input, colSrc, predicate, conv });
            }

            return new OpaqueDataView(impl);
        }

        private static Impl<T1, T2> CreateImpl<T1, T2>(
            IHostEnvironment env, string name, IDataView input, int colSrc,
            InPredicate<T2> pred, ValueMapper<T1, T2> conv)
        {
            return new Impl<T1, T2>(env, name, input, colSrc, pred, conv);
        }

        private sealed class Impl<T1, T2> : FilterBase
        {
            private readonly int _colSrc;
            private readonly InPredicate<T2> _pred;
            private readonly ValueMapper<T1, T2> _conv;

            public Impl(IHostEnvironment env, string name, IDataView input,
                int colSrc, InPredicate<T2> pred, ValueMapper<T1, T2> conv = null)
                : base(env, name, input)
            {
                Host.AssertValue(pred);
                Host.Assert(conv != null | typeof(T1) == typeof(T2));
                Host.Assert(0 <= colSrc & colSrc < Source.Schema.Count);

                _colSrc = colSrc;
                _pred = pred;
                _conv = conv;
            }

            private protected override void SaveModel(ModelSaveContext ctx)
            {
                Host.Assert(false, "Shouldn't serialize this!");
                throw Host.ExceptNotSupp("Shouldn't serialize this");
            }

            protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
            {
                Host.AssertValue(predicate);
                // This transform has no preference.
                return null;
            }

            protected override DataViewRowCursor GetRowCursorCore(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                Host.AssertValueOrNull(rand);
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

                bool[] active;
                Func<int, bool> inputPred = GetActive(predicate, out active);

                var inputCols = Source.Schema.Where(x => inputPred(x.Index));
                var input = Source.GetRowCursor(inputCols, rand);
                return new Cursor(this, input, active);
            }

            public override DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {

                Host.CheckValueOrNull(rand);
                var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, OutputSchema);

                bool[] active;
                Func<int, bool> inputPred = GetActive(predicate, out active);
                var inputCols = Source.Schema.Where(x => inputPred(x.Index));
                var inputs = Source.GetRowCursorSet(inputCols, n, rand);
                Host.AssertNonEmpty(inputs);

                // No need to split if this is given 1 input cursor.
                var cursors = new DataViewRowCursor[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                    cursors[i] = new Cursor(this, inputs[i], active);
                return cursors;
            }

            private Func<int, bool> GetActive(Func<int, bool> predicate, out bool[] active)
            {
                Host.AssertValue(predicate);
                active = new bool[Source.Schema.Count];
                bool[] activeInput = new bool[Source.Schema.Count];
                for (int i = 0; i < active.Length; i++)
                    activeInput[i] = active[i] = predicate(i);
                activeInput[_colSrc] = true;
                return col => activeInput[col];
            }

            // REVIEW: Should this cache the source value like MissingValueFilter does?
            private sealed class Cursor : LinkedRowFilterCursorBase
            {
                private readonly ValueGetter<T1> _getSrc;
                private readonly InPredicate<T1> _pred;
                private T1 _src;

                public Cursor(Impl<T1, T2> parent, DataViewRowCursor input, bool[] active)
                    : base(parent.Host, input, parent.OutputSchema, active)
                {
                    _getSrc = Input.GetGetter<T1>(Input.Schema[parent._colSrc]);
                    if (parent._conv == null)
                    {
                        Ch.Assert(typeof(T1) == typeof(T2));
                        _pred = (InPredicate<T1>)(Delegate)parent._pred;
                    }
                    else
                    {
                        T2 val = default(T2);
                        var pred = parent._pred;
                        var conv = parent._conv;
                        _pred =
                            (in T1 src) =>
                            {
                                conv(in _src, ref val);
                                return pred(in val);
                            };
                    }
                }

                protected override bool Accept()
                {
                    _getSrc(ref _src);
                    return _pred(in _src);
                }
            }

        }
    }
}
