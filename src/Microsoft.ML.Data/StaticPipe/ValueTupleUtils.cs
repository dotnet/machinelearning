// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data.StaticPipe
{
    internal static class ValueTupleUtils
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
