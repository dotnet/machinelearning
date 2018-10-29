// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This applies the user provided RefPredicate to a column and drops rows that map to false. It automatically
    /// injects a standard conversion from the actual type of the source column to typeSrc (if needed).
    /// </summary>
    public static class LambdaFilter
    {
        public static IDataView Create<TSrc>(IHostEnvironment env, string name, IDataView input,
            string src, ColumnType typeSrc, InPredicate<TSrc> predicate)
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
            var typeOrig = input.Schema.GetColumnType(colSrc);

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
                Host.Assert(0 <= colSrc & colSrc < Source.Schema.ColumnCount);

                _colSrc = colSrc;
                _pred = pred;
                _conv = conv;
            }

            public override void Save(ModelSaveContext ctx)
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

            protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
            {
                Host.AssertValue(predicate, "predicate");
                Host.AssertValueOrNull(rand);

                bool[] active;
                Func<int, bool> inputPred = GetActive(predicate, out active);
                var input = Source.GetRowCursor(inputPred, rand);
                return new RowCursor(this, input, active);
            }

            public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
                Func<int, bool> predicate, int n, IRandom rand = null)
            {
                Host.CheckValue(predicate, nameof(predicate));
                Host.CheckValueOrNull(rand);

                bool[] active;
                Func<int, bool> inputPred = GetActive(predicate, out active);
                var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
                Host.AssertNonEmpty(inputs);

                // No need to split if this is given 1 input cursor.
                var cursors = new IRowCursor[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                    cursors[i] = new RowCursor(this, inputs[i], active);
                return cursors;
            }

            private Func<int, bool> GetActive(Func<int, bool> predicate, out bool[] active)
            {
                Host.AssertValue(predicate);
                active = new bool[Source.Schema.ColumnCount];
                bool[] activeInput = new bool[Source.Schema.ColumnCount];
                for (int i = 0; i < active.Length; i++)
                    activeInput[i] = active[i] = predicate(i);
                activeInput[_colSrc] = true;
                return col => activeInput[col];
            }

            // REVIEW: Should this cache the source value like MissingValueFilter does?
            private sealed class RowCursor : LinkedRowFilterCursorBase
            {
                private readonly ValueGetter<T1> _getSrc;
                private readonly InPredicate<T1> _pred;
                private T1 _src;

                public RowCursor(Impl<T1, T2> parent, IRowCursor input, bool[] active)
                    : base(parent.Host, input, parent.Schema, active)
                {
                    _getSrc = Input.GetGetter<T1>(parent._colSrc);
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
                                conv(ref _src, ref val);
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
