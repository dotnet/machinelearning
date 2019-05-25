// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;

namespace Microsoft.ML.AutoML
{
    internal static class MLNetUtils
    {
        public static int[] GetRandomPermutation(Random rand, int size)
        {
            var res = GetIdentityPermutation(size);
            Shuffle<int>(rand, res);
            return res;
        }

        public static int[] GetIdentityPermutation(int size)
        {
            var res = new int[size];
            for (int i = 0; i < size; i++)
                res[i] = i;
            return res;
        }

        public static void Shuffle<T>(Random rand, Span<T> rgv)
        {
            for (int iv = 0; iv < rgv.Length; iv++)
                Swap(ref rgv[iv], ref rgv[iv + rand.Next(rgv.Length - iv)]);
        }

        public static void Swap<T>(ref T a, ref T b)
        {
            T temp = a;
            a = b;
            b = temp;
        }

        public static int Size<T>(T[] x)
        {
            return x == null ? 0 : x.Length;
        }

        private static MethodInfo MarshalInvokeCheckAndCreate<TRet>(Type genArg, Delegate func)
        {
            var method = MarshalActionInvokeCheckAndCreate(genArg, func);
            if (method.ReturnType != typeof(TRet))
            {
                throw new ArgumentException("Cannot be generic on return type");
            }
            return method;
        }

        private static MethodInfo MarshalActionInvokeCheckAndCreate(Type genArg, Delegate func)
        {
            var meth = func.GetMethodInfo();
            meth = meth.GetGenericMethodDefinition().MakeGenericMethod(genArg);
            return meth;
        }

        /// <summary>
        /// Given a generic method with a single type parameter, re-create the generic method on a new type,
        /// then reinvoke the method and return the result. A common pattern throughout the code base is to
        /// have some sort of generic method, whose parameters and return value are, as defined, non-generic,
        /// but whose code depends on some sort of generic type parameter. This utility method exists to make
        /// this common pattern more convenient, and also safer so that the arguments, if any, can be type
        /// checked at compile time instead of at runtime.
        ///
        /// Because it is strongly typed, this can only be applied to methods whose return type
        /// is known at compile time, that is, that do not depend on the type parameter of the method itself.
        /// </summary>
        /// <typeparam name="TRet">The return value</typeparam>
        /// <param name="func">A delegate that should be a generic method with a single type parameter.
        /// The generic method definition will be extracted, then a new method will be created with the
        /// given type parameter, then the method will be invoked.</param>
        /// <param name="genArg">The new type parameter for the generic method</param>
        /// <returns>The return value of the invoked function</returns>
        public static TRet MarshalInvoke<TRet>(Func<TRet> func, Type genArg)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, null);
        }

        /// <summary>
        /// A two-argument version of <see cref="MarshalInvoke{TRet}"/>.
        /// </summary>
        public static TRet MarshalInvoke<TArg1, TArg2, TRet>(Func<TArg1, TArg2, TRet> func, Type genArg, TArg1 arg1, TArg2 arg2)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2 });
        }
    }
}
