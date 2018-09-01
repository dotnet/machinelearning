// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Data.StaticPipe.Runtime
{
    /// <summary>
    /// Utility functions useful for the internal implementations of the key pipeline utilities.
    /// </summary>
    internal static class StaticPipeInternalUtils
    {
        /// <summary>
        /// Given a type which is a <see cref="ValueTuple"/> tree with <see cref="PipelineColumn"/> leaves, return an instance of that
        /// type which has appropriate instances of <see cref="PipelineColumn"/> that use the returned reconciler.
        /// </summary>
        /// <param name="fakeReconciler">This is a data-reconciler that always reconciles to a <c>null</c> object</param>
        /// <typeparam name="T">A type of either <see cref="ValueTuple"/> or one of the major <see cref="PipelineColumn"/> subclasses
        /// (e.g., <see cref="Scalar{T}"/>, <see cref="Vector{T}"/>, etc.)</typeparam>
        /// <returns>An instance of <typeparamref name="T"/> where all <see cref="PipelineColumn"/> fields have the provided reconciler</returns>
        public static T MakeAnalysisInstance<T>(out ReaderReconciler<int> fakeReconciler)
        {
            var rec = new AnalyzeUtil.Rec();
            fakeReconciler = rec;
            return (T)AnalyzeUtil.MakeAnalysisInstanceCore<T>(rec);
        }

        private static class AnalyzeUtil
        {
            public sealed class Rec : ReaderReconciler<int>
            {
                public Rec() : base() { }

                public override IDataReaderEstimator<int, IDataReader<int>> Reconcile(
                    IHostEnvironment env, PipelineColumn[] toOutput, IReadOnlyDictionary<PipelineColumn, string> outputNames)
                {
                    Contracts.AssertValue(env);
                    foreach (var col in toOutput)
                        env.Assert(col.ReconcilerObj == this);
                    return null;
                }
            }

            private static Reconciler _reconciler = new Rec();

            private sealed class AScalar<T> : Scalar<T> { public AScalar(Rec rec) : base(rec, null) { } }
            private sealed class AVector<T> : Vector<T> { public AVector(Rec rec) : base(rec, null) { } }
            private sealed class ANormVector<T> : NormVector<T> { public ANormVector(Rec rec) : base(rec, null) { } }
            private sealed class AVarVector<T> : VarVector<T> { public AVarVector(Rec rec) : base(rec, null) { } }
            private sealed class AKey<T> : Key<T> { public AKey(Rec rec) : base(rec, null) { } }
            private sealed class AKey<T, TV> : Key<T, TV> { public AKey(Rec rec) : base(rec, null) { } }
            private sealed class AVarKey<T> : VarKey<T> { public AVarKey(Rec rec) : base(rec, null) { } }

            private static PipelineColumn MakeScalar<T>(Rec rec) => new AScalar<T>(rec);
            private static PipelineColumn MakeVector<T>(Rec rec) => new AVector<T>(rec);
            private static PipelineColumn MakeNormVector<T>(Rec rec) => new ANormVector<T>(rec);
            private static PipelineColumn MakeVarVector<T>(Rec rec) => new AVarVector<T>(rec);
            private static PipelineColumn MakeKey<T>(Rec rec) => new AKey<T>(rec);
            private static Key<T, TV> MakeKey<T, TV>(Rec rec) => new AKey<T, TV>(rec);
            private static PipelineColumn MakeVarKey<T>(Rec rec) => new AVarKey<T>(rec);

            private static MethodInfo[] _valueTupleCreateMethod = InitValueTupleCreateMethods();

            private static MethodInfo[] InitValueTupleCreateMethods()
            {
                const string methodName = nameof(ValueTuple.Create);
                var methods = typeof(ValueTuple).GetMethods()
                    .Where(m => m.Name == methodName && m.ContainsGenericParameters)
                    .OrderBy(m => m.GetGenericArguments().Length).Take(7)
                    .Append(typeof(AnalyzeUtil).GetMethod(nameof(UnstructedCreate))).ToArray();
                return methods;
            }

            /// <summary>
            /// Note that we use this instead of <see cref="ValueTuple.Create{T1, T2, T3, T4, T5, T6, T7, T8}(T1, T2, T3, T4, T5, T6, T7, T8)"/>
            /// for the eight-item because that method will embed the last element into a one-element tuple,
            /// which is embedded in the original. The actual physical representation, which is what is relevant here,
            /// has no real conveniences around its creation.
            /// </summary>
            public static ValueTuple<T1, T2, T3, T4, T5, T6, T7, TRest>
                UnstructedCreate<T1, T2, T3, T4, T5, T6, T7, TRest>(
                T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, TRest restTuple)
                where TRest : struct
            {
                return new ValueTuple<T1, T2, T3, T4, T5, T6, T7, TRest>(v1, v2, v3, v4, v5, v6, v7, restTuple);
            }

            public static object MakeAnalysisInstanceCore<T>(Rec rec)
            {
                var t = typeof(T);
                if (typeof(PipelineColumn).IsAssignableFrom(t))
                {
                    if (t.IsGenericType)
                    {
                        var genP = t.GetGenericArguments();
                        var genT = t.GetGenericTypeDefinition();

                        if (genT == typeof(Scalar<>))
                            return Utils.MarshalInvoke(MakeScalar<int>, genP[0], rec);
                        if (genT == typeof(Vector<>))
                            return Utils.MarshalInvoke(MakeVector<int>, genP[0], rec);
                        if (genT == typeof(NormVector<>))
                            return Utils.MarshalInvoke(MakeNormVector<int>, genP[0], rec);
                        if (genT == typeof(VarVector<>))
                            return Utils.MarshalInvoke(MakeVarVector<int>, genP[0], rec);
                        if (genT == typeof(Key<>))
                            return Utils.MarshalInvoke(MakeKey<uint>, genP[0], rec);
                        if (genT == typeof(Key<,>))
                        {
                            Func<Rec, PipelineColumn> f = MakeKey<uint, int>;
                            return f.Method.GetGenericMethodDefinition().MakeGenericMethod(genP).Invoke(null, new object[] { rec });
                        }
                        if (genT == typeof(VarKey<>))
                            return Utils.MarshalInvoke(MakeVector<int>, genP[0], rec);
                    }
                    throw Contracts.Except($"Type {t} is a {nameof(PipelineColumn)} yet does not appear to be directly one of " +
                        $"the official types. This is commonly due to a mistake by the component author and can be addressed by " +
                        $"upcasting the instance in the tuple definition to one of the official types.");
                }
                // If it's not a pipeline column type then we suppose it is a value tuple.

                if (t.IsGenericType && ValueTupleUtils.IsValueTuple(t))
                {
                    var genT = t.GetGenericTypeDefinition();
                    var genP = t.GetGenericArguments();
                    if (1 <= genP.Length && genP.Length <= 8)
                    {
                        // First recursively create the sub-analysis objects.
                        object[] subArgs = genP.Select(subType => Utils.MarshalInvoke(MakeAnalysisInstanceCore<int>, subType, rec)).ToArray();
                        // Next create the tuple.
                        return _valueTupleCreateMethod[subArgs.Length - 1].MakeGenericMethod(genP).Invoke(null, subArgs);
                    }
                }
                throw Contracts.Except($"Type {t} is neither a {nameof(PipelineColumn)} subclass nor a value tuple. Other types are not permitted.");
            }
        }

        private struct Info
        {
            public readonly Type Type;
            public readonly object Item;

            public Info(Type type, object item)
            {
                Type = type;
                Item = item;
            }
        }

        public static KeyValuePair<string, Type>[] GetNamesTypes<T>(ParameterInfo pInfo)
            => GetNamesTypes<T, PipelineColumn>(pInfo);

        public static KeyValuePair<string, Type>[] GetNamesTypes<T, TLeaf>(ParameterInfo pInfo)
        {
            Contracts.CheckValue(pInfo, nameof(pInfo));
            if (typeof(T) != pInfo.ParameterType)
                throw Contracts.ExceptParam(nameof(pInfo), "Type mismatch with " + typeof(T).Name);
            var result = NameUtil<TLeaf>.GetNames<T>(default, pInfo);
            var retVal = new KeyValuePair<string, Type>[result.Length];
            for (int i = 0; i < result.Length; ++i)
            {
                retVal[i] = new KeyValuePair<string, Type>(result[i].name, result[i].type);
                Contracts.Assert(result[i].value == default);
            }
            return retVal;
        }

        public static KeyValuePair<string, PipelineColumn>[] GetNamesValues<T>(T record, ParameterInfo pInfo)
            => GetNamesValues<T, PipelineColumn>(record, pInfo);

        private static KeyValuePair<string, TLeaf>[] GetNamesValues<T, TLeaf>(T record, ParameterInfo pInfo)
        {
            Contracts.CheckValue(pInfo, nameof(pInfo));
            Contracts.CheckParam(typeof(T) == pInfo.ParameterType, nameof(pInfo), "Type mismatch with " + nameof(record));
            var result = NameUtil<TLeaf>.GetNames<T>(record, pInfo);
            var retVal = new KeyValuePair<string, TLeaf>[result.Length];
            for (int i = 0; i < result.Length; ++i)
                retVal[i] = new KeyValuePair<string, TLeaf>(result[i].name, result[i].value);
            return retVal;
        }

        /// <summary>
        /// Utility for extracting names out of value-tuple tree structures.
        /// </summary>
        /// <typeparam name="TLeaf"></typeparam>
        private static class NameUtil<TLeaf>
        {
            /// <summary>
            /// A utility for exacting name/type/value triples out of a value-tuple based tree structure.
            ///
            /// For example: If <typeparamref name="TLeaf"/> were <see cref="int"/> then the value-tuple
            /// <c>(a: 1, b: (c: 2, d: 3), e: 4)</c> would result in the return array where the name/value
            /// pairs were <c>[("a", 1), ("b.c", 2), ("b.d", 3), "e", 4]</c>, in some order, and the type
            /// is <c>typeof(int)</c>.
            ///
            /// Note that the type returned in the triple is the type as declared in the tuple, which will
            /// be a derived type of <typeparamref name="TLeaf"/>, and in turn the type of the value will be
            /// of a type derived from that type.
            ///
            /// This method will throw if anything other than value-tuples or <typeparamref name="TLeaf"/>
            /// instances are detected during its execution.
            /// </summary>
            /// <typeparam name="T">The type to extract on.</typeparam>
            /// <param name="record">The instance to extract values out of.</param>
            /// <param name="pInfo">A type parameter associated with this, usually extracted out of some
            /// delegate over this value tuple type. Note that names in value-tuples are an illusion perpetrated
            /// by the C# compiler, and are not accessible though <typeparamref name="T"/> by reflection, which
            /// is why it is necessary to engage in trickery like passing in a delegate over those types, which
            /// does retain the information on the names.</param>
            /// <returns>The list of name/type/value triples extracted out of the tree like-structure</returns>
            public static (string name, Type type, TLeaf value)[] GetNames<T>(T record, ParameterInfo pInfo)
            {
                Contracts.AssertValue(pInfo);
                Contracts.Assert(typeof(T) == pInfo.ParameterType);
                // Record can only be null if it isn't the value tuple type.

                if (typeof(TLeaf).IsAssignableFrom(typeof(T)))
                    return new[] { ("Data", typeof(T), (TLeaf)(object)record) };

                // The structure of names for value tuples is somewhat unusual. All names in a nested structure of value
                // tuples is arranged in a roughly depth-first structure, unless we consider tuple cardinality greater
                // than seven (which is physically stored in a tuple of cardinality eight, with the so-called `Rest`
                // field iteratively holding "more" values. So what appears to be a ten-tuple is really an eight-tuple,
                // with the first seven items holding the first seven items of the original tuple, and another value
                // tuple in `Rest` holding the remaining three items.

                // Anyway: the names are given in depth-first fashion with all items in a tuple being assigned
                // contiguously to the items (so for any n-tuple, there is an contiguous n-length segment in the names
                // array corresponding to the names). This also applies to the "virtual" >7 tuples, which are for this
                // purpose considered "one" tuple, which has some interesting implications on valid traversals of the
                // structure.

                var tupleNames = pInfo.GetCustomAttribute<TupleElementNamesAttribute>()?.TransformNames;
                var accumulated = new List<(string, Type, TLeaf)>();
                RecurseNames<T>(record, tupleNames, 0, null, accumulated);
                return accumulated.ToArray();
            }

            /// <summary>
            /// Helper method for <see cref="GetNamesValues{T, TRoot}(T, ParameterInfo)"/>, that given a <see cref="ValueTuple"/>
            /// <paramref name="record"/> will either append triples to <paramref name="accum"/> (if the item is of type
            /// <typeparamref name="TLeaf"/>), or recurse on this function (if the item is a <see cref="ValueTuple"/>),
            /// or otherwise throw an error.
            /// </summary>
            /// <typeparam name="T">The type we are recursing on, should be a <see cref="ValueTuple"/> of some sort</typeparam>
            /// <param name="record">The <see cref="ValueTuple"/> we are extracting on. Note that this is <see cref="object"/>
            /// just for the sake of ease of using
            /// <see cref="Utils.MarshalInvoke{TArg1, TArg2, TArg3, TArg4, TArg5, TRet}(Func{TArg1, TArg2, TArg3, TArg4, TArg5, TRet}, Type, TArg1, TArg2, TArg3, TArg4, TArg5)"/></param>.
            /// <param name="names">The names list extracted from the <see cref="TupleElementNamesAttribute"/> attribute, or <c>null</c>
            /// if no such attribute could be found.</param>
            /// <param name="namesOffset">The offset into <paramref name="names"/> where <paramref name="record"/>'s names begin.</param>
            /// <param name="namePrefix"><c>null</c> for the root level structure, or the appendation of <c>.</c> suffixed names
            /// of the path of value-tuples down to this item.</param>
            /// <param name="accum">The list into which the names are being added</param>
            /// <returns>The total number of items added to <paramref name="accum"/></returns>
            private static int RecurseNames<T>(object record, IList<string> names, int namesOffset, string namePrefix, List<(string, Type, TLeaf)> accum)
            {
                if (!ValueTupleUtils.IsValueTuple(typeof(T)))
                {
                    throw Contracts.Except($"Expected to find structure composed of {typeof(ValueTuple)} and {typeof(TLeaf)} " +
                        $" but during traversal of the structure an item of {typeof(T)} was found instead");
                }
                Contracts.AssertValue(record);
                Contracts.Assert(record is T);
                Contracts.AssertValueOrNull(names);
                Contracts.Assert(names == null || namesOffset <= names.Count);
                Contracts.AssertValueOrNull(namePrefix);
                Contracts.AssertValue(accum);

                var tupleItems = new List<Info>();

                ValueTupleUtils.ApplyActionToTuple((T)record, (index, type, item)
                    => tupleItems.Add(new Info(type, item)));
                int total = tupleItems.Count;

                for (int i = 0; i < tupleItems.Count; ++i)
                {
                    string name = names?[namesOffset + i] ?? $"Item{i + 1}";
                    if (!string.IsNullOrEmpty(namePrefix))
                        name = namePrefix + name;

                    if (typeof(TLeaf).IsAssignableFrom(tupleItems[i].Type))
                        accum.Add((name, tupleItems[i].Type, (TLeaf)tupleItems[i].Item));
                    else
                    {
                        total += Utils.MarshalInvoke(RecurseNames<int>, tupleItems[i].Type,
                            tupleItems[i].Item, names, namesOffset + total, name + ".", accum);
                    }
                }

                return total;
            }
        }

        private static class ValueTupleUtils
        {
            public static bool IsValueTuple(Type t)
            {
                Type genT = t.IsGenericType ? t.GetGenericTypeDefinition() : t;
                return genT == typeof(ValueTuple<>) || genT == typeof(ValueTuple<,>) || genT == typeof(ValueTuple<,,>)
                    || genT == typeof(ValueTuple<,,,>) || genT == typeof(ValueTuple<,,,,>) || genT == typeof(ValueTuple<,,,,,>)
                    || genT == typeof(ValueTuple<,,,,,,>) || genT == typeof(ValueTuple<,,,,,,,>);
            }

            public delegate void TupleItemAction(int index, Type itemType, object item);

            public static void ApplyActionToTuple<T>(T tuple, TupleItemAction action)
            {
                Contracts.CheckValue(action, nameof(action));
                ApplyActionToTuple<T>(tuple, 0, action);
            }

            internal static void ApplyActionToTuple<T>(object tuple, int root, TupleItemAction action)
            {
                Contracts.AssertValue(action);
                Contracts.Assert(root >= 0);

                var tType = typeof(T);
                if (tType.IsGenericType)
                    tType = tType.GetGenericTypeDefinition();

                if (typeof(ValueTuple<>) == tType)
                    MarshalInvoke<ValueTuple<int>>(Process, tuple, root, action);
                else if (typeof(ValueTuple<,>) == tType)
                    MarshalInvoke<ValueTuple<int, int>>(Process, tuple, root, action);
                else if (typeof(ValueTuple<,,>) == tType)
                    MarshalInvoke<ValueTuple<int, int, int>>(Process, tuple, root, action);
                else if (typeof(ValueTuple<,,,>) == tType)
                    MarshalInvoke<ValueTuple<int, int, int, int>>(Process, tuple, root, action);
                else if (typeof(ValueTuple<,,,,>) == tType)
                    MarshalInvoke<ValueTuple<int, int, int, int, int>>(Process, tuple, root, action);
                else if (typeof(ValueTuple<,,,,,>) == tType)
                    MarshalInvoke<ValueTuple<int, int, int, int, int, int>>(Process, tuple, root, action);
                else if (typeof(ValueTuple<,,,,,,>) == tType)
                    MarshalInvoke<ValueTuple<int, int, int, int, int, int, int>>(Process, tuple, root, action);
                else if (typeof(ValueTuple<,,,,,,,>) == tType)
                    MarshalInvoke<ValueTuple<int, int, int, int, int, int, int, ValueTuple<int>>>(Process, tuple, root, action);
                else
                {
                    // This will fall through here if this was either not a generic type or is a value tuple type.
                    throw Contracts.ExceptParam(nameof(tuple), $"Item should have been a {nameof(ValueTuple)} but was instead {tType}");
                }
            }

            private delegate void Processor<T>(T val, int root, TupleItemAction action);

            private static void Process<T1>(ValueTuple<T1> val, int root, TupleItemAction action)
            {
                action(root++, typeof(T1), val.Item1);
            }

            private static void Process<T1, T2>(ValueTuple<T1, T2> val, int root, TupleItemAction action)
            {
                action(root++, typeof(T1), val.Item1);
                action(root++, typeof(T2), val.Item2);
            }

            private static void Process<T1, T2, T3>(ValueTuple<T1, T2, T3> val, int root, TupleItemAction action)
            {
                action(root++, typeof(T1), val.Item1);
                action(root++, typeof(T2), val.Item2);
                action(root++, typeof(T3), val.Item3);
            }

            private static void Process<T1, T2, T3, T4>(ValueTuple<T1, T2, T3, T4> val, int root, TupleItemAction action)
            {
                action(root++, typeof(T1), val.Item1);
                action(root++, typeof(T2), val.Item2);
                action(root++, typeof(T3), val.Item3);
                action(root++, typeof(T4), val.Item4);
            }

            private static void Process<T1, T2, T3, T4, T5>(ValueTuple<T1, T2, T3, T4, T5> val, int root, TupleItemAction action)
            {
                action(root++, typeof(T1), val.Item1);
                action(root++, typeof(T2), val.Item2);
                action(root++, typeof(T3), val.Item3);
                action(root++, typeof(T4), val.Item4);
                action(root++, typeof(T5), val.Item5);
            }

            private static void Process<T1, T2, T3, T4, T5, T6>(ValueTuple<T1, T2, T3, T4, T5, T6> val, int root, TupleItemAction action)
            {
                action(root++, typeof(T1), val.Item1);
                action(root++, typeof(T2), val.Item2);
                action(root++, typeof(T3), val.Item3);
                action(root++, typeof(T4), val.Item4);
                action(root++, typeof(T5), val.Item5);
                action(root++, typeof(T6), val.Item6);
            }

            private static void Process<T1, T2, T3, T4, T5, T6, T7>(ValueTuple<T1, T2, T3, T4, T5, T6, T7> val, int root, TupleItemAction action)
            {
                action(root++, typeof(T1), val.Item1);
                action(root++, typeof(T2), val.Item2);
                action(root++, typeof(T3), val.Item3);
                action(root++, typeof(T4), val.Item4);
                action(root++, typeof(T5), val.Item5);
                action(root++, typeof(T6), val.Item6);
                action(root++, typeof(T7), val.Item7);
            }

            private static void Process<T1, T2, T3, T4, T5, T6, T7, TRest>(ValueTuple<T1, T2, T3, T4, T5, T6, T7, TRest> val, int root, TupleItemAction action)
                where TRest : struct
            {
                action(root++, typeof(T1), val.Item1);
                action(root++, typeof(T2), val.Item2);
                action(root++, typeof(T3), val.Item3);
                action(root++, typeof(T4), val.Item4);
                action(root++, typeof(T5), val.Item5);
                action(root++, typeof(T6), val.Item6);
                action(root++, typeof(T7), val.Item7);
                ApplyActionToTuple<TRest>(val.Rest, root++, action);
            }

            private static void MarshalInvoke<T>(Processor<T> del, object arg, int root, TupleItemAction action)
            {
                Contracts.AssertValue(del);
                Contracts.Assert(del.Method.IsGenericMethod);
                var argType = arg.GetType();
                Contracts.Assert(argType.IsGenericType);
                var argGenTypes = argType.GetGenericArguments();
                // The argument generic types should be compatible with the delegate's generic types.
                Contracts.Assert(del.Method.GetGenericArguments().Length == argGenTypes.Length);
                // Reconstruct the delegate generic types so it adheres to the args generic types.
                var newDel = del.Method.GetGenericMethodDefinition().MakeGenericMethod(argGenTypes);

                var result = newDel.Invoke(null, new object[] { arg, root, action });
            }
        }
    }
}
