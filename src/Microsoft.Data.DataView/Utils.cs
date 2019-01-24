// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Reflection;

namespace Microsoft.ML.Data
{
    internal static class Utils
    {
        public static void MarshalActionInvoke<TArg1>(Action<TArg1> act, Type genArg, TArg1 arg1)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, act);
            meth.Invoke(act.Target, new object[] { arg1 });
        }

        public static void MarshalActionInvoke<TArg1, TArg2, TArg3, TArg4>(Action<TArg1, TArg2, TArg3, TArg4> act, Type genArg, TArg1 arg1, TArg2 arg2, TArg3 arg3, TArg4 arg4)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, act);
            meth.Invoke(act.Target, new object[] { arg1, arg2, arg3, arg4 });
        }

        public static TRet MarshalInvoke<TArg1, TArg2, TRet>(Func<TArg1, TArg2, TRet> func, Type genArg, TArg1 arg1, TArg2 arg2)
        {
            var meth = MarshalInvokeCheckAndCreate<TRet>(genArg, func);
            return (TRet)meth.Invoke(func.Target, new object[] { arg1, arg2 });
        }

        private static MethodInfo MarshalInvokeCheckAndCreate<TRet>(Type genArg, Delegate func)
        {
            var meth = MarshalActionInvokeCheckAndCreate(genArg, func);
            Debug.Assert(meth.ReturnType == typeof(TRet));
            return meth;
        }

        private static MethodInfo MarshalActionInvokeCheckAndCreate(Type genArg, Delegate func)
        {
            var meth = func.GetMethodInfo();
            meth = meth.GetGenericMethodDefinition().MakeGenericMethod(genArg);
            return meth;
        }
    }
}
