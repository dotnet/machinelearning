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
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.StaticPipe.Runtime
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
        /// (for example, <see cref="Scalar{T}"/>, <see cref="Vector{T}"/>, etc.)</typeparam>
        /// <returns>An instance of <typeparamref name="T"/> where all <see cref="PipelineColumn"/> fields have the provided reconciler</returns>
        public static T MakeAnalysisInstance<T>(out ReaderReconciler<int> fakeReconciler)
        {
            var rec = new AnalyzeUtil.Rec();
            fakeReconciler = rec;
            return (T)AnalyzeUtil.MakeAnalysisInstanceCore<T>(rec, new HashSet<Type>());
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
            private sealed class ACustom<T> : Custom<T> { public ACustom(Rec rec) : base(rec, null) { } }

            private static PipelineColumn MakeScalar<T>(Rec rec) => new AScalar<T>(rec);
            private static PipelineColumn MakeVector<T>(Rec rec) => new AVector<T>(rec);
            private static PipelineColumn MakeNormVector<T>(Rec rec) => new ANormVector<T>(rec);
            private static PipelineColumn MakeVarVector<T>(Rec rec) => new AVarVector<T>(rec);
            private static PipelineColumn MakeKey<T>(Rec rec) => new AKey<T>(rec);
            private static Key<T, TV> MakeKey<T, TV>(Rec rec) => new AKey<T, TV>(rec);
            private static PipelineColumn MakeVarKey<T>(Rec rec) => new AVarKey<T>(rec);
            private static PipelineColumn MakeCustom<T>(Rec rec) => new ACustom<T>(rec);

            private static MethodInfo[] _valueTupleCreateMethod = InitValueTupleCreateMethods();

            private static MethodInfo[] InitValueTupleCreateMethods()
            {
                const string methodName = nameof(ValueTuple.Create);
                var methods = typeof(ValueTuple).GetMethods()
                    .Where(m => m.Name == methodName && m.ContainsGenericParameters)
                    .OrderBy(m => m.GetGenericArguments().Length).Take(7)
                    .ToArray().AppendElement(typeof(AnalyzeUtil).GetMethod(nameof(UnstructedCreate)));
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

            public static object MakeAnalysisInstanceCore<T>(Rec rec, HashSet<Type> encountered)
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
                        if (genT == typeof(Custom<>))
                            return Utils.MarshalInvoke(MakeCustom<int>, genP[0], rec);
                    }
                    throw Contracts.Except($"Type {t} is a {nameof(PipelineColumn)} yet does not appear to be directly one of " +
                        $"the official types. This is commonly due to a mistake by the component author and can be addressed by " +
                        $"upcasting the instance in the tuple definition to one of the official types.");
                }
                // If it's not a pipeline column type, perhaps it is a value-tuple.

                if (ValueTupleUtils.IsValueTuple(t))
                {
                    var genT = t.GetGenericTypeDefinition();
                    var genP = t.GetGenericArguments();
                    Contracts.Assert(1 <= genP.Length && genP.Length <= 8);
                    // First recursively create the sub-analysis objects.
                    object[] subArgs = genP.Select(subType => Utils.MarshalInvoke(MakeAnalysisInstanceCore<int>, subType, rec, encountered)).ToArray();
                    // Next create the tuple.
                    return _valueTupleCreateMethod[subArgs.Length - 1].MakeGenericMethod(genP).Invoke(null, subArgs);
                }
                else
                {
                    // If neither of these, perhaps it's a supported type of property-bearing class. Either way, this is the sort
                    // of class we have to be careful about since there could be some recursively defined types.
                    if (!encountered.Add(t))
                        throw Contracts.Except($"Recursively defined type {t} encountered.");
                    var func = GetContainerMaker(t, out Type[] inputTypes);
                    object[] subArgs = inputTypes.Select(subType => Utils.MarshalInvoke(MakeAnalysisInstanceCore<int>, subType, rec, encountered)).ToArray();
                    encountered.Remove(t);
                    return func(subArgs);
                }

                //throw Contracts.Except($"Type {t} is neither a {nameof(PipelineColumn)} subclass nor a value tuple. Other types are not permitted.");
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

        /// <summary>
        /// Given a schema shape defining instance, return the pairs of names and values, based on a recursive
        /// traversal of the structure. If in that list the value <c>a.b.c</c> is paired with an item <c>x</c>,
        /// then programmatically <paramref name="record"/> when accessed as <paramref name="record"/><c>.a.b.c</c>
        /// would be that item <c>x</c>.
        /// </summary>
        /// <typeparam name="T">The schema shape defining type.</typeparam>
        /// <param name="record">The instance of that schema shape defining type, whose items will
        /// populate the <see cref="KeyValuePair{TKey, TValue}.Value"/> fields of the returned items.</param>
        /// <param name="pInfo">It is an implementation detail of the value-tuple type that the names
        /// are not associated with the type at all, but there is instead an illusion propagated within
        /// Visual Studio, that works via attributes. Programmatic access to this is limited, except that
        /// a <see cref="TupleElementNamesAttribute"/> is attached to the type in appropriate places, for example,
        /// in a delegate one of the parameters, or the return parameter, or somesuch. If present, the names
        /// will be extracted from that structure, and if not the default names of <c>Item1</c>, <c>Item2</c>,
        /// etc. will be used. Note that non-value-tuple tupes do not have this problem.</param>
        /// <returns>The set of names and corresponding values.</returns>
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
        /// A sort of extended version of <see cref="Type.IsAssignableFrom(Type)"/> that accounts
        /// for the presence of the <see cref="Vector{T}"/>, <see cref="VarVector{T}"/> and <see cref="NormVector{T}"/> types. />
        /// </summary>
        /// <param name="to">Can we assign to this type?</param>
        /// <param name="from">From that type?</param>
        /// <returns></returns>
        public static bool IsAssignableFromStaticPipeline(this Type to, Type from)
        {
            Contracts.AssertValue(to);
            Contracts.AssertValue(from);
            if (to.IsAssignableFrom(from))
                return true;
            // The only exception to the above test are the vector types. These are generic types.
            if (!to.IsGenericType || !from.IsGenericType)
                return false;
            var gto = to.GetGenericTypeDefinition();
            var gfrom = from.GetGenericTypeDefinition();

            // If either of the types is not one of the vector types, we can just stop right here.
            if ((gto != typeof(Vector<>) && gto != typeof(VarVector<>) && gto != typeof(NormVector<>)) ||
                (gfrom != typeof(Vector<>) && gfrom != typeof(VarVector<>) && gfrom != typeof(NormVector<>)))
            {
                return false;
            }

            // First check the value types. If those don't match, no sense going any further.
            var ato = to.GetGenericArguments();
            var afrom = from.GetGenericArguments();
            Contracts.Assert(Utils.Size(ato) == 1);
            Contracts.Assert(Utils.Size(afrom) == 1);

            if (!ato[0].IsAssignableFrom(afrom[0]))
                return false;

            // We have now confirmed at least the compatibility of the item types. Next we must confirm the same of the vector type.
            // Variable sized vectors must match in their types, norm vector can be considered assignable to vector.

            // If either is a var vector, the other must be as well.
            if (gto == typeof(VarVector<>))
                return gfrom == typeof(VarVector<>);

            // We can assign from NormVector<> to Vector<>, but not the other way around. So we only fail if we are trying to assign Vector<> to NormVector<>.
            return gfrom != typeof(Vector<>) || gto != typeof(NormVector<>);
        }

        /// <summary>
        /// Utility for extracting names out of shape-shape structures.
        /// </summary>
        /// <typeparam name="TLeaf">The base type in the base world.</typeparam>
        private static class NameUtil<TLeaf>
        {
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
            /// delegate over this type. In the case of value-tuples specifically, note that names in value-tuples
            /// are an illusion perpetrated by the C# compiler, and are not accessible though <typeparamref name="T"/>
            /// by reflection, which is why it is necessary to engage in trickery like passing in a delegate over
            /// those types, which does retain the information on the names.</param>
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
                Contracts.AssertValueOrNull(record);
                Contracts.Assert(record == null || record is T);
                Contracts.Assert(record == null || !typeof(TLeaf).IsAssignableFrom(record.GetType()));
                Contracts.AssertValueOrNull(names);
                Contracts.Assert(names == null || namesOffset <= names.Count);
                Contracts.AssertValueOrNull(namePrefix);
                Contracts.AssertValue(accum);

                var ttype = typeof(T);
                if (ValueTupleUtils.IsValueTuple(ttype))
                {
                    record = record ?? Activator.CreateInstance(ttype);
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

                // Otherwise it may be a class. Let's first check.
                string error = VerifyIsSupportedContainingType(ttype);
                if (error != null)
                {
                    if (string.IsNullOrEmpty(namePrefix))
                        throw Contracts.Except(error);
                    throw Contracts.Except($"Problem with {namePrefix}: {error}");
                }

                var props = ttype.GetProperties(BindingFlags.Public | BindingFlags.Instance);
                foreach (var prop in props)
                {
                    var propValue = record == null ? null : prop.GetValue(record);
                    var name = namePrefix + prop.Name;

                    if (typeof(TLeaf).IsAssignableFrom(prop.PropertyType))
                        accum.Add((name, prop.PropertyType, (TLeaf)propValue));
                    else
                    {
                        // It may be that the property itself points to a value-tuple. Get it, just in case.
                        var tupleNames = prop.GetCustomAttribute<TupleElementNamesAttribute>()?.TransformNames;
                        // Do not incremenet the total in this case. This was not a value-tuple, and any internal thing
                        // that was a value-tuple should not result in an increment on the count. Correspondingly, we also
                        // start the recursion again insofar as the offset is concerned.
                        Utils.MarshalInvoke(RecurseNames<int>, prop.PropertyType,
                            propValue, tupleNames, 0, name + ".", accum);
                    }
                }
                return 0;
            }
        }

        /// <summary>
        /// Verifies whether the given type is a supported containing type. This returns the error
        /// message rather than throwing itself so that the caller can report the error in the way that
        /// is most appropriate to their context.
        /// </summary>
        /// <param name="type">The type to check.</param>
        /// <returns>A non-null answer </returns>
        private static string VerifyIsSupportedContainingType(Type type)
        {
            Contracts.AssertValue(type);
            if (type.IsValueType)
                return $"Type {type.Name} is a value-type, not a reference type.";

            // Somehow, BindingsFlags.Public does not find the public constructor. Who knows why.
            var constructors = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance);
            if (constructors.Length != 1)
                return $"Type {type.Name} for schema shape should have exactly one public constructor.";
            var constructor = constructors[0];
            var parameters = constructor.GetParameters();

            // Let's do a small minor smoke test on the type to see that there is a one to one correspondence between
            // the types of the properties, and the types of the parameters in the constructor.
            var counters = new Dictionary<Type, int>();

            // We allow one of two ways to create the instance. We can either have an empty constructor with all settable properties,
            // or a constructor with as many parameters as properties, where there is a correspondence between the types.
            var props = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
            foreach (var prop in props)
            {
                // Skip all non-public properties.
                if (prop.GetAccessors().All(a => !a.IsPublic && !a.IsStatic))
                    continue;
                if (!prop.CanRead)
                    return $"Type {type.Name} for schema shape has non-readable property {prop.Name}.";
                if (prop.CanWrite != (parameters.Length == 0))
                {
                    if (prop.CanWrite)
                        return $"Type {type.Name} for schema shape has writable property {prop.Name}, but also has a non-empty constructor.";
                    return $"Type {type.Name} for schema shape has non-writable property {prop.Name}, but also has an empty constructor.";
                }
                counters.TryGetValue(prop.PropertyType, out int currCount);
                counters[prop.PropertyType] = currCount + 1;
            }
            // Next let's check the types of the constructor properties, if any, and make sure there is a correspondence.
            if (parameters.Length > 0)
            {
                foreach (var p in parameters)
                {
                    if (!counters.TryGetValue(p.ParameterType, out int c) || c == 0)
                        return $"Constructor parameter {p.Name} is of type {p.ParameterType.Name} which appeared more often than we found a corresponding property.";
                    counters[p.ParameterType]--;
                }
            }
            return null;
        }

        /// <summary>
        /// Creates a unified means of creating an instance of the given containing schema-shape type,
        /// whether it is the type that has a constructor that has all the inputs enumerated and getter
        /// // methods,
        /// </summary>
        /// <param name="type"></param>
        /// <param name="inputTypes">The types we expect when constructing. Note that </param>
        /// <returns></returns>
        public static Func<object[], object> GetContainerMaker(Type type, out Type[] inputTypes)
        {
            Contracts.AssertValue(type);
            var typeError = VerifyIsSupportedContainingType(type);
            if (typeError != null)
                throw Contracts.ExceptParam(nameof(type), typeError);
            var constructor = type.GetConstructors().First(p => p.IsPublic);
            var parameters = constructor.GetParameters();
            var props = type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);

            Func<object[], object> retval;
            if (parameters.Length == 0)
            {
                // This kind of functions like a quasi-constructor that sets all teh objects.
                Action<object, object[]> allSetter = null;

                inputTypes = new Type[props.Length];
                // All properties must have setters.
                for (int i = 0; i < props.Length; ++i)
                {
                    inputTypes[i] = props[i].PropertyType;
                    int ii = i;
                    allSetter +=
                        (obj, inputs) =>
                        {
                            Contracts.Assert(inputs.Length == props.Length);
                            Contracts.Assert(props[ii].CanWrite);
                            Contracts.Assert(props[ii].PropertyType.IsAssignableFrom(inputs[ii].GetType()));
                            props[ii].SetValue(obj, inputs[ii]);
                        };
                }
                retval =
                    inputs =>
                    {
                        var obj = constructor.Invoke(new object[0]);
                        allSetter?.Invoke(obj, inputs);
                        return obj;
                    };
            }
            else
            {
                // Otherwise it's the constructor variant.
                inputTypes = Utils.BuildArray(parameters.Length, i => parameters[i].ParameterType);
                retval = constructor.Invoke;
            }
            // In either case, we would like there to be a check after this to ensure that no funny-business
            // went on with the initialization, and that every public property is in fact reference equatable.

            return inputs =>
                {
                    var inputSet = new HashSet<object>(inputs);
                    var obj = retval(inputs);
                    foreach (var prop in props)
                    {
                        var propValue = prop.GetValue(obj);
                        if (!inputSet.Remove(propValue))
                            throw Contracts.Except($"While making {type.Name} instance, unexpected value found in property {prop.Name}.");
                    }
                    return obj;
                };
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
