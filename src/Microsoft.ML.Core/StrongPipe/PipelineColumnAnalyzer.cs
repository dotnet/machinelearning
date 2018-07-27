using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.ML.Core.StrongPipe.Columns;
using Microsoft.ML.Runtime;
using System;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Core.StrongPipe
{
    internal static class PipelineColumnAnalyzer
    {
        public static void Analyze<TIn, TOut>(Func<TIn, TOut> func)
        {
            bool singletonIn = typeof(PipelineColumn).IsAssignableFrom(typeof(TIn));
            bool singletonOut = typeof(PipelineColumn).IsAssignableFrom(typeof(TOut));

            var analysis = CreateAnalysisInstance<TIn>();
            var analysisOut = func(analysis);

            var inNames = GetNames<TIn, PipelineColumn>(analysis, func.Method.GetParameters()[0]);
            var outNames = GetNames<TOut, PipelineColumn>(analysisOut, func.Method.ReturnParameter);
        }

        public interface IIsAnalysisColumn { }

        /// <summary>
        /// Given a type which is a <see cref="ValueTuple"/> tree with <see cref="PipelineColumn"/> leaves, return an instance of that
        /// type which has appropriate instances of <see cref="PipelineColumn"/> that also implement the marker interface
        /// <see cref="IIsAnalysisColumn"/>.
        /// </summary>
        /// <typeparam name="T">A type of either <see cref="ValueTuple"/> or one of the major <see cref="PipelineColumn"/> subclasses
        /// (e.g., <see cref="Scalar{T}"/>, <see cref="Vector{T}"/>, etc.)</typeparam>
        /// <returns>An instance of <typeparamref name="T"/> where all <see cref="PipelineColumn"/> fields have instances implementing
        /// <see cref="IIsAnalysisColumn"/> filled in</returns>
        public static T CreateAnalysisInstance<T>()
            => (T)AnalyzeUtil.CreateAnalysisInstanceCore<T>();

        private static class AnalyzeUtil
        {
            private sealed class Rec : Reconciler
            {
            }

            private static Reconciler _reconciler = new Rec();

            private sealed class AScalar<T> : Scalar<T>, IIsAnalysisColumn { public AScalar() : base(_reconciler, null) { } }
            private sealed class AVector<T> : Vector<T>, IIsAnalysisColumn { public AVector() : base(_reconciler, null) { } }
            private sealed class AVarVector<T> : VarVector<T>, IIsAnalysisColumn { public AVarVector() : base(_reconciler, null) { } }
            private sealed class AKey<T> : Key<T>, IIsAnalysisColumn { public AKey() : base(_reconciler, null) { } }
            private sealed class AKey<T, TV> : Key<T, TV>, IIsAnalysisColumn { public AKey() : base(_reconciler, null) { } }
            private sealed class AVarKey<T> : VarKey<T>, IIsAnalysisColumn { public AVarKey() : base(_reconciler, null) { } }

            private static PipelineColumn CreateScalar<T>() => new AScalar<T>();
            private static PipelineColumn CreateVector<T>() => new AVector<T>();
            private static PipelineColumn CreateVarVector<T>() => new AVarVector<T>();
            private static PipelineColumn CreateKey<T>() => new AKey<T>();
            private static Key<T, TV> CreateKey<T, TV>() => new AKey<T, TV>();
            private static PipelineColumn CreateVarKey<T>() => new AVarKey<T>();

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

            public static object CreateAnalysisInstanceCore<T>()
            {
                var t = typeof(T);
                if (typeof(PipelineColumn).IsAssignableFrom(t))
                {
                    if (t.IsGenericType)
                    {
                        var genP = t.GetGenericArguments();
                        var genT = t.GetGenericTypeDefinition();

                        if (genT == typeof(Scalar<>))
                            return Utils.MarshalInvoke(CreateScalar<int>, genP[0]);
                        if (genT == typeof(Vector<>))
                            return Utils.MarshalInvoke(CreateVector<int>, genP[0]);
                        if (genT == typeof(VarVector<>))
                            return Utils.MarshalInvoke(CreateVarVector<int>, genP[0]);
                        if (genT == typeof(Key<>))
                            return Utils.MarshalInvoke(CreateKey<uint>, genP[0]);
                        if (genT == typeof(Key<,>))
                        {
                            Func<PipelineColumn> f = CreateKey<uint, int>;
                            return f.Method.GetGenericMethodDefinition().MakeGenericMethod(genP).Invoke(null, new object[0]);
                        }
                        if (genT == typeof(VarKey<>))
                            return Utils.MarshalInvoke(CreateVector<int>, genP[0]);
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
                        object[] subArgs = genP.Select(subType => Utils.MarshalInvoke(CreateAnalysisInstanceCore<int>, subType)).ToArray();
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

        internal static KeyValuePair<string, PipelineColumn>[] GetNames<T>(T record, ParameterInfo pInfo)
            => GetNames<T, PipelineColumn>(record, pInfo);

        internal static KeyValuePair<string, TLeaf>[] GetNames<T, TLeaf>(T record, ParameterInfo pInfo)
        {
            Contracts.CheckValue(pInfo, nameof(pInfo));
            Contracts.CheckParam(typeof(T) == pInfo.ParameterType, nameof(pInfo), "Type mismatch with " + nameof(record));
            return NameUtil<TLeaf>.GetNames<T>(record, pInfo);
        }

        /// <summary>
        /// Utility for extracting names out of value-tuple tree structures.
        /// </summary>
        /// <typeparam name="TLeaf"></typeparam>
        private static class NameUtil<TLeaf>
        {
            /// <summary>
            /// A utility for exacting name/value pairs out of a value-tuple based tree structure.
            ///
            /// For example: If <typeparamref name="TLeaf"/> were <see cref="int"/> then the value-tuple
            /// <c>(a: 1, b: (c: 2, d: 3), e: 4)</c> would result in the return array of
            /// <see cref="KeyValuePair{TKey, TValue}"/> with <see cref="string"/> keys and <see cref="int"/> values
            /// <c>[("a": 1), ("b.c": 2), ("b.d": 3), "e": 4]</c>, in some order.
            ///
            /// This method will throw if anything other than value-tuples or <typeparamref name="TLeaf"/>
            /// instances are detected during its execution.
            /// </summary>
            /// <typeparam name="T">The type to extract on, </typeparam>
            /// <param name="record">The instance to extract values out of</param>
            /// <param name="pInfo">A type parameter associated with this, usually extracted out of some
            /// delegate over this value tuple type. Note that names in value-tuples are an illusion perpetrated
            /// by the C# compiler</param>
            /// <returns>The list of name/value pairs extracted out of the tree like-structure</returns>
            public static KeyValuePair<string, TLeaf>[] GetNames<T>(T record, ParameterInfo pInfo)
            {
                Contracts.AssertValue(pInfo);
                Contracts.Assert(typeof(T) == pInfo.ParameterType);
                Contracts.Assert(typeof(T).IsValueType || record != null);

                if (typeof(TLeaf).IsAssignableFrom(record.GetType()))
                    return new[] { new KeyValuePair<string, TLeaf>("Data", (TLeaf)(object)record) };

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
                var accumulated = new List<KeyValuePair<string, TLeaf>>();
                RecurseNames<T>(record, tupleNames, 0, null, accumulated);
                return accumulated.ToArray();
            }

            /// <summary>
            /// Helper method for <see cref="GetNames{T, TRoot}(T, ParameterInfo)"/>, that given a <see cref="ValueTuple"/>
            /// <paramref name="record"/> will either append pairs to <paramref name="accum"/> (if the item is of type
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
            private static int RecurseNames<T>(object record, IList<string> names, int namesOffset, string namePrefix, List<KeyValuePair<string, TLeaf>> accum)
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
                        accum.Add(new KeyValuePair<string, TLeaf>(name, (TLeaf)tupleItems[i].Item));
                    else
                    {
                        total += Utils.MarshalInvoke(RecurseNames<int>, tupleItems[i].Type,
                            tupleItems[i].Item, names, namesOffset + total, name + ".", accum);
                    }
                }

                return total;
            }
        }

    }
}
