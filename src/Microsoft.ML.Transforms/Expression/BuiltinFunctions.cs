// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using Microsoft.ML.Data;
using Microsoft.ML.Data.Conversion;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    using BL = System.Boolean;
    using I4 = System.Int32;
    using I8 = System.Int64;
    using R4 = Single;
    using R8 = Double;
    using TX = ReadOnlyMemory<char>;

    internal static class FunctionProviderUtils
    {
        /// <summary>
        /// Returns whether the given object is non-null and an NA value for one of the standard types.
        /// </summary>
        public static bool IsNA(object v)
        {
            if (v == null)
                return false;
            Type type = v.GetType();
            if (type == typeof(R4))
                return R4.IsNaN((R4)v);
            if (type == typeof(R8))
                return R8.IsNaN((R8)v);
            Contracts.Assert(type == typeof(BL) || type == typeof(I4) || type == typeof(I8) || type == typeof(TX),
                "Unexpected constant value type!");
            return false;
        }

        /// <summary>
        /// Returns the standard NA value for the given standard type.
        /// </summary>
        public static object GetNA(Type type)
        {
            if (type == typeof(R4))
                return R4.NaN;
            if (type == typeof(R8))
                return R8.NaN;
            Contracts.Assert(false, "Unexpected constant value type!");
            return null;
        }

        /// <summary>
        /// Helper method to bundle one or more MethodInfos into an array.
        /// </summary>
        public static MethodInfo[] Ret(params MethodInfo[] funcs)
        {
            Contracts.AssertValue(funcs);
            return funcs;
        }

        /// <summary>
        /// Returns the MethodInfo for the given delegate.
        /// </summary>
        public static MethodInfo Fn<T1>(Func<T1> fn)
        {
            Contracts.AssertValue(fn);
            Contracts.Assert(fn.Target == null);
            return fn.GetMethodInfo();
        }

        /// <summary>
        /// Returns the MethodInfo for the given delegate.
        /// </summary>
        public static MethodInfo Fn<T1, T2>(Func<T1, T2> fn)
        {
            Contracts.AssertValue(fn);
            Contracts.Assert(fn.Target == null);
            return fn.GetMethodInfo();
        }

        /// <summary>
        /// Returns the MethodInfo for the given delegate.
        /// </summary>
        public static MethodInfo Fn<T1, T2, T3>(Func<T1, T2, T3> fn)
        {
            Contracts.AssertValue(fn);
            Contracts.Assert(fn.Target == null);
            return fn.GetMethodInfo();
        }

        /// <summary>
        /// Returns the MethodInfo for the given delegate.
        /// </summary>
        public static MethodInfo Fn<T1, T2, T3, T4>(Func<T1, T2, T3, T4> fn)
        {
            Contracts.AssertValue(fn);
            Contracts.Assert(fn.Target == null);
            return fn.GetMethodInfo();
        }
    }

    /// <summary>
    /// The standard builtin functions for ExprTransform.
    /// </summary>
    internal sealed class BuiltinFunctions : IFunctionProvider
    {
        private static volatile BuiltinFunctions _instance;
        public static BuiltinFunctions Instance
        {
            get
            {
                if (_instance == null)
                    Interlocked.CompareExchange(ref _instance, new BuiltinFunctions(), null);
                return _instance;
            }
        }

        public string NameSpace { get { return "global"; } }

        /// <summary>
        /// Returns the MethodInfo for <see cref="Id{T}(T)"/>
        /// </summary>
        private static MethodInfo Id<T>()
        {
            Action<T> fn = Id<T>;
            Contracts.Assert(fn.Target == null);
            return fn.GetMethodInfo();
        }

        // This is an "identity" function.
        private static void Id<T>(T src) { }

        public MethodInfo[] Lookup(string name)
        {
            switch (name)
            {
                case "pi":
                    return FunctionProviderUtils.Ret(FunctionProviderUtils.Fn<R8>(Pi));

                case "na":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(NA),
                        FunctionProviderUtils.Fn<R8, R8>(NA));
                case "default":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, I4>(Default),
                        FunctionProviderUtils.Fn<I8, I8>(Default),
                        FunctionProviderUtils.Fn<R4, R4>(Default),
                        FunctionProviderUtils.Fn<R8, R8>(Default),
                        FunctionProviderUtils.Fn<BL, BL>(Default),
                        FunctionProviderUtils.Fn<TX, TX>(Default));

                case "abs":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, I4>(Math.Abs),
                        FunctionProviderUtils.Fn<I8, I8>(Math.Abs),
                        FunctionProviderUtils.Fn<R4, R4>(Math.Abs),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Abs));
                case "sign":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, I4>(Sign),
                        FunctionProviderUtils.Fn<I8, I8>(Sign),
                        FunctionProviderUtils.Fn<R4, R4>(Sign),
                        FunctionProviderUtils.Fn<R8, R8>(Sign));
                case "exp":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Exp),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Exp));
                case "ln":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Log),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Log));
                case "log":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Log),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Log),
                        FunctionProviderUtils.Fn<R4, R4, R4>(Log),
                        FunctionProviderUtils.Fn<R8, R8, R8>(Math.Log));

                case "deg":
                case "degrees":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Deg),
                        FunctionProviderUtils.Fn<R8, R8>(Deg));
                case "rad":
                case "radians":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Rad),
                        FunctionProviderUtils.Fn<R8, R8>(Rad));

                case "sin":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Sin),
                        FunctionProviderUtils.Fn<R8, R8>(Sin));
                case "sind":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(SinD),
                        FunctionProviderUtils.Fn<R8, R8>(SinD));
                case "cos":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Cos),
                        FunctionProviderUtils.Fn<R8, R8>(Cos));
                case "cosd":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(CosD),
                        FunctionProviderUtils.Fn<R8, R8>(CosD));
                case "tan":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Tan),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Tan));
                case "tand":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(TanD),
                        FunctionProviderUtils.Fn<R8, R8>(TanD));

                case "asin":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Asin),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Asin));
                case "acos":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Acos),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Acos));
                case "atan":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Atan),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Atan));
                case "atan2":
                case "atanyx":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4, R4>(Atan2),
                        FunctionProviderUtils.Fn<R8, R8, R8>(Atan2));

                case "sinh":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Sinh),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Sinh));
                case "cosh":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Cosh),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Cosh));
                case "tanh":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Tanh),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Tanh));

                case "sqrt":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Sqrt),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Sqrt));

                case "trunc":
                case "truncate":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Truncate),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Truncate));
                case "floor":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Floor),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Floor));
                case "ceil":
                case "ceiling":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Ceiling),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Ceiling));
                case "round":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, R4>(Round),
                        FunctionProviderUtils.Fn<R8, R8>(Math.Round));

                case "min":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, I4, I4>(Math.Min),
                        FunctionProviderUtils.Fn<I8, I8, I8>(Math.Min),
                        FunctionProviderUtils.Fn<R4, R4, R4>(Math.Min),
                        FunctionProviderUtils.Fn<R8, R8, R8>(Math.Min));
                case "max":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, I4, I4>(Math.Max),
                        FunctionProviderUtils.Fn<I8, I8, I8>(Math.Max),
                        FunctionProviderUtils.Fn<R4, R4, R4>(Math.Max),
                        FunctionProviderUtils.Fn<R8, R8, R8>(Math.Max));

                case "len":
                    return FunctionProviderUtils.Ret(FunctionProviderUtils.Fn<TX, I4>(Len));
                case "lower":
                    return FunctionProviderUtils.Ret(FunctionProviderUtils.Fn<TX, TX>(Lower));
                case "upper":
                    return FunctionProviderUtils.Ret(FunctionProviderUtils.Fn<TX, TX>(Upper));
                case "right":
                    return FunctionProviderUtils.Ret(FunctionProviderUtils.Fn<TX, I4, TX>(Right));
                case "left":
                    return FunctionProviderUtils.Ret(FunctionProviderUtils.Fn<TX, I4, TX>(Left));
                case "mid":
                    return FunctionProviderUtils.Ret(FunctionProviderUtils.Fn<TX, I4, I4, TX>(Mid));

                case "concat":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<TX>(Empty),
                        Id<TX>(),
                        FunctionProviderUtils.Fn<TX, TX, TX>(Concat),
                        FunctionProviderUtils.Fn<TX, TX, TX, TX>(Concat),
                        FunctionProviderUtils.Fn<TX[], TX>(Concat));

                case "isna":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<R4, BL>(IsNA),
                        FunctionProviderUtils.Fn<R8, BL>(IsNA));

                case "bool":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<TX, BL>(ToBL),
                        Id<BL>());
                case "int":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I8, I4>(Convert.ToInt32),
                        FunctionProviderUtils.Fn<R4, I4>(Convert.ToInt32),
                        FunctionProviderUtils.Fn<R8, I4>(Convert.ToInt32),
                        FunctionProviderUtils.Fn<BL, I4>(Convert.ToInt32),
                        FunctionProviderUtils.Fn<TX, I4>(ToI4),
                        Id<I4>());
                case "long":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, I8>(Convert.ToInt64),
                        FunctionProviderUtils.Fn<R4, I8>(Convert.ToInt64),
                        FunctionProviderUtils.Fn<R8, I8>(Convert.ToInt64),
                        FunctionProviderUtils.Fn<BL, I8>(Convert.ToInt64),
                        FunctionProviderUtils.Fn<TX, I8>(ToI8),
                        Id<I8>());
                case "float":
                case "single":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, R4>(Convert.ToSingle),
                        FunctionProviderUtils.Fn<I8, R4>(Convert.ToSingle),
                        FunctionProviderUtils.Fn<R4, R4>(ToR4),
                        FunctionProviderUtils.Fn<R8, R4>(ToR4),
                        FunctionProviderUtils.Fn<BL, R4>(Convert.ToSingle),
                        FunctionProviderUtils.Fn<TX, R4>(ToR4));
                case "double":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, R8>(Convert.ToDouble),
                        FunctionProviderUtils.Fn<I8, R8>(Convert.ToDouble),
                        FunctionProviderUtils.Fn<R4, R8>(ToR8),
                        FunctionProviderUtils.Fn<R8, R8>(ToR8),
                        FunctionProviderUtils.Fn<BL, R8>(Convert.ToDouble),
                        FunctionProviderUtils.Fn<TX, R8>(ToR8));
                case "text":
                    return FunctionProviderUtils.Ret(
                        FunctionProviderUtils.Fn<I4, TX>(ToTX),
                        FunctionProviderUtils.Fn<I8, TX>(ToTX),
                        FunctionProviderUtils.Fn<R4, TX>(ToTX),
                        FunctionProviderUtils.Fn<R8, TX>(ToTX),
                        FunctionProviderUtils.Fn<BL, TX>(ToTX),
                        Id<TX>());
            }

            return null;
        }

        public object ResolveToConstant(string name, MethodInfo fn, object[] values)
        {
            Contracts.CheckNonEmpty(name, nameof(name));
            Contracts.CheckValue(fn, nameof(fn));
            Contracts.CheckParam(Utils.Size(values) > 0, nameof(values), "Expected values to have positive length");
            Contracts.CheckParam(!values.All(x => x != null), nameof(values), "Expected values to contain at least one null");

            switch (name)
            {
                case "na":
                    {
                        Contracts.Assert(values.Length == 1);

                        Type type = fn.ReturnType;
                        if (type == typeof(R4))
                            return R4.NaN;
                        if (type == typeof(R8))
                            return R8.NaN;
                        return null;
                    }
                case "default":
                    {
                        Contracts.Assert(values.Length == 1);

                        Type type = fn.ReturnType;
                        if (type == typeof(I4))
                            return default(I4);
                        if (type == typeof(I8))
                            return default(I8);
                        if (type == typeof(R4))
                            return default(R4);
                        if (type == typeof(R8))
                            return default(R8);
                        if (type == typeof(BL))
                            return default(BL);
                        if (type == typeof(TX))
                            return default(TX);
                        Contracts.Assert(false, "Unexpected return type!");
                        return null;
                    }
            }

            // By default, constant NA arguments produce an NA result. Note that this is not true for isna,
            // but those functions will get here only if values contains a single null, not an NA.
            for (int i = 0; i < values.Length; i++)
            {
                if (FunctionProviderUtils.IsNA(values[i]))
                {
                    Contracts.Assert(values.Length > 1);
                    return FunctionProviderUtils.GetNA(fn.ReturnType);
                }
            }

            return null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 Pi()
        {
            return Math.PI;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 NA(R4 a)
        {
            return R4.NaN;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 NA(R8 a)
        {
            return R8.NaN;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static I4 Default(I4 a)
        {
            return default(I4);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static I8 Default(I8 a)
        {
            return default(I8);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Default(R4 a)
        {
            return default(R4);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 Default(R8 a)
        {
            return default(R8);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Default(BL a)
        {
            return default(BL);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Default(TX a)
        {
            return default(TX);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Sign(R4 a)
        {
            // Preserves NaN. Unfortunately, it also preserves negative zero,
            // but perhaps that is a good thing?
            return a > 0 ? +1 : a < 0 ? -1 : a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 Sign(R8 a)
        {
            // Preserves NaN. Unfortunately, it also preserves negative zero,
            // but perhaps that is a good thing?
            return a > 0 ? +1 : a < 0 ? -1 : a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static I4 Sign(I4 a)
        {
            return a > 0 ? +1 : a < 0 ? -1 : a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static I8 Sign(I8 a)
        {
            return a > 0 ? +1 : a < 0 ? -1 : a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Pow(R4 a, R4 b)
        {
            return (R4)Math.Pow(a, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Exp(R4 a)
        {
            return (R4)Math.Exp(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Log(R4 a)
        {
            return (R4)Math.Log(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Log(R4 a, R4 b)
        {
            return (R4)Math.Log(a, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Deg(R4 a)
        {
            return (R4)(a * (180 / Math.PI));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 Deg(R8 a)
        {
            return a * (180 / Math.PI);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Rad(R4 a)
        {
            return (R4)(a * (Math.PI / 180));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 Rad(R8 a)
        {
            return a * (Math.PI / 180);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Sin(R4 a)
        {
            return (R4)Math.Sin(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 Sin(R8 a)
        {
            return MathUtils.Sin(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 SinD(R4 a)
        {
            return (R4)Math.Sin(a * (Math.PI / 180));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 SinD(R8 a)
        {
            return MathUtils.Sin(a * (Math.PI / 180));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Cos(R4 a)
        {
            return (R4)Math.Cos(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 Cos(R8 a)
        {
            return MathUtils.Cos(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 CosD(R4 a)
        {
            return (R4)Math.Cos(a * (Math.PI / 180));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 CosD(R8 a)
        {
            return MathUtils.Cos(a * (Math.PI / 180));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Tan(R4 a)
        {
            return (R4)Math.Tan(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 TanD(R4 a)
        {
            return (R4)Math.Tan(a * (Math.PI / 180));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 TanD(R8 a)
        {
            return Math.Tan(a * (Math.PI / 180));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Asin(R4 a)
        {
            return (R4)Math.Asin(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Acos(R4 a)
        {
            return (R4)Math.Acos(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Atan(R4 a)
        {
            return (R4)Math.Atan(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Atan2(R4 a, R4 b)
        {
            // According to the documentation of Math.Atan2: if x and y are either System.Double.PositiveInfinity
            // or System.Double.NegativeInfinity, the method returns System.Double.NaN, but this seems to not be the case.
            if (R4.IsInfinity(a) && R4.IsInfinity(b))
                return R4.NaN;
            return (R4)Math.Atan2(a, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 Atan2(R8 a, R8 b)
        {
            // According to the documentation of Math.Atan2: if x and y are either System.Double.PositiveInfinity
            // or System.Double.NegativeInfinity, the method returns System.Double.NaN, but this seems to not be the case.
            if (R8.IsInfinity(a) && R8.IsInfinity(b))
                return R8.NaN;
            return Math.Atan2(a, b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Sinh(R4 a)
        {
            return (R4)Math.Sinh(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Cosh(R4 a)
        {
            return (R4)Math.Cosh(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Tanh(R4 a)
        {
            return (R4)Math.Tanh(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Sqrt(R4 a)
        {
            return (R4)Math.Sqrt(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Truncate(R4 a)
        {
            return (R4)Math.Truncate(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Floor(R4 a)
        {
            return (R4)Math.Floor(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Ceiling(R4 a)
        {
            return (R4)Math.Ceiling(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 Round(R4 a)
        {
            return (R4)Math.Round(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Lower(TX a)
        {
            if (a.IsEmpty)
                return a;
            var sb = new StringBuilder();
            ReadOnlyMemoryUtils.AddLowerCaseToStringBuilder(a.Span, sb);
            return sb.ToString().AsMemory();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Upper(TX a)
        {
            if (a.IsEmpty)
                return a;
            var dst = new char[a.Length];
            a.Span.ToUpperInvariant(dst);
            return new TX(dst);
        }

        // Special case some common Concat sizes, for better efficiency.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Empty()
        {
            return TX.Empty;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Concat(TX a, TX b)
        {
            if (a.IsEmpty)
                return b;
            if (b.IsEmpty)
                return a;
            var dst = new char[a.Length + b.Length];
            a.Span.CopyTo(dst);
            b.Span.CopyTo(new Span<char>(dst, a.Length, b.Length));
            return new TX(dst);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Concat(TX a, TX b, TX c)
        {
            var dst = new char[a.Length + b.Length + c.Length];
            a.Span.CopyTo(dst);
            b.Span.CopyTo(new Span<char>(dst, a.Length, b.Length));
            c.Span.CopyTo(new Span<char>(dst, a.Length + b.Length, c.Length));
            return new TX(dst);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Concat(TX[] a)
        {
            Contracts.AssertValue(a);

            int len = 0;
            for (int i = 0; i < a.Length; i++)
                len += a[i].Length;
            if (len == 0)
                return TX.Empty;

            var sb = new StringBuilder(len);
            for (int i = 0; i < a.Length; i++)
                sb.AppendSpan(a[i].Span);
            return sb.ToString().AsMemory();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static I4 Len(TX a)
        {
            return a.Length;
        }

        /// <summary>
        /// Given an index meant to index into a given sequence normalize it according to
        /// these rules: negative indices get <paramref name="len"/> added to them, and
        /// then the index is clamped the range 0 to <paramref name="len"/> inclusive,
        /// and that result is returned. (For those familiar with Python, this is the same
        /// as the logic for slice normalization.)
        /// </summary>
        /// <param name="i">The index to normalize</param>
        /// <param name="len">The length of the sequence</param>
        /// <returns>The normalized version of the index, a non-positive value no greater
        /// than <paramref name="len"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int NormalizeIndex(int i, int len)
        {
            Contracts.Assert(0 <= len);
            if (i < 0)
            {
                if ((i += len) < 0)
                    return 0;
            }
            else if (i > len)
                return len;
            return i;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Right(TX a, I4 min)
        {
            if (a.IsEmpty)
                return a;
            return a.Slice(NormalizeIndex(min, a.Length));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Left(TX a, I4 lim)
        {
            if (a.IsEmpty)
                return a;
            return a.Slice(0, NormalizeIndex(lim, a.Length));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TX Mid(TX a, I4 min, I4 lim)
        {
            if (a.IsEmpty)
                return a;
            int im = NormalizeIndex(min, a.Length);
            int il = NormalizeIndex(lim, a.Length);
            if (im >= il)
                return TX.Empty;
            return a.Slice(im, il - im);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL IsNA(R4 a)
        {
            return R4.IsNaN(a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL IsNA(R8 a)
        {
            return R8.IsNaN(a);
        }

        public static BL ToBL(TX a)
        {
            BL res = default(BL);
            Conversions.DefaultInstance.Convert(in a, ref res);
            return res;
        }

        public static I4 ToI4(TX a)
        {
            I4 res = default(I4);
            Conversions.DefaultInstance.Convert(in a, ref res);
            return res;
        }

        public static I8 ToI8(TX a)
        {
            I8 res = default(I8);
            Conversions.DefaultInstance.Convert(in a, ref res);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 ToR4(R4 a)
        {
            // Note that the cast is intentional and NOT a no-op. It forces the JIT
            // to narrow to R4 when it might be tempted to keep intermediate
            // computations in larger precision.
            return (R4)a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R4 ToR4(R8 a)
        {
            return (R4)a;
        }

        public static R4 ToR4(TX a)
        {
            R4 res = default(R4);
            Conversions.DefaultInstance.Convert(in a, ref res);
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 ToR8(R4 a)
        {
            return (R8)a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static R8 ToR8(R8 a)
        {
            // Note that the cast is intentional and NOT a no-op. It forces the JIT
            // to narrow to R4 when it might be tempted to keep intermediate
            // computations in larger precision.
            return a;
        }

        public static R8 ToR8(TX a)
        {
            R8 res = default(R8);
            Conversions.DefaultInstance.Convert(in a, ref res);
            return res;
        }

        public static TX ToTX(I4 src) => src.ToString().AsMemory();
        public static TX ToTX(I8 src) => src.ToString().AsMemory();
        public static TX ToTX(R4 src) => src.ToString("R", CultureInfo.InvariantCulture).AsMemory();
        public static TX ToTX(R8 src) => src.ToString("G17", CultureInfo.InvariantCulture).AsMemory();
        public static TX ToTX(BL src)
        {
            if (!src)
                return "0".AsMemory();
            else
                return "1".AsMemory();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL Equals(TX first, TX second)
        {
            return first.Span.SequenceEqual(second.Span);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL NotEquals(TX first, TX second)
        {
            return !Equals(first, second);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte Not(bool b)
        {
            return !b ? (byte)1 : default;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL False()
        {
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BL True()
        {
            return true;
        }

        /// <summary>
        /// Raise a to the b power. Special cases:
        /// * a^(negative value) => 0
        /// * In case of overflow, return I4.MinValue or I4.MaxValue, based on whether the result would have been
        /// negative or positive.
        /// </summary>
        public static I4 Pow(I4 a, I4 b)
        {
            if (a == 1)
                return 1;
            switch (b)
            {
                case 0:
                    return 1;
                case 1:
                    return a;
                case 2:
                    return a * a;
            }
            if (a == -1)
                return (b & 1) == 0 ? 1 : -1;
            if (b < 0)
                return 0;

            bool neg = false;
            if (a < 0)
            {
                a = -a;
                neg = (b & 1) != 0;
            }

            // Since the abs of the base is at least two, the exponent must be less than 31.
            if (b >= 31)
                return neg ? I4.MinValue : I4.MaxValue;

            if (a == 0)
            {
                if (b == 0)
                    return 1;
                return 0;
            }

            Contracts.Assert(a >= 2);

            // Since the exponent is at least three, the base must be <= 1290.
            Contracts.Assert(b >= 3);
            if (a > 1290)
                return neg ? I4.MinValue : I4.MaxValue;

            // REVIEW: Should we use a checked context and exception catching like I8 does?
            ulong u = (ulong)(uint)a;
            ulong result = 1;
            for (; ; )
            {
                if ((b & 1) != 0 && (result *= u) > I4.MaxValue)
                    return neg ? I4.MinValue : I4.MaxValue;
                b >>= 1;
                if (b == 0)
                    break;
                if ((u *= u) > I4.MaxValue)
                    return neg ? I4.MinValue : I4.MaxValue;
            }
            Contracts.Assert(result <= I4.MaxValue);

            var res = (I4)result;
            if (neg)
                res = -res;
            return res;
        }

        /// <summary>
        /// Raise a to the b power. Special cases:
        /// * a^(negative value) => 0
        /// * In case of overflow, return I8.MinValue or I8.MaxValue, based on whether the result would have been
        /// negative or positive.
        /// </summary>
        public static I8 Pow(I8 a, I8 b)
        {
            if (a == 1)
                return 1;
            switch (b)
            {
                case 0:
                    return 1;
                case 1:
                    return a;
                case 2:
                    return a * a;
            }
            if (a == -1)
                return (b & 1) == 0 ? 1 : -1;
            if (b < 0)
                return 0;

            bool neg = false;
            if (a < 0)
            {
                a = -a;
                neg = (b & 1) != 0;
            }

            // Since the abs of the base is at least two, the exponent must be less than 63.
            if (b >= 63)
                return neg ? I8.MinValue : I8.MaxValue;

            if (a == 0)
            {
                if (b == 0)
                    return 1;
                return 0;
            }

            Contracts.Assert(a >= 2);

            // Since the exponent is at least three, the base must be < 2^21.
            Contracts.Assert(b >= 3);
            if (a >= (1L << 21))
                return neg ? I8.MinValue : I8.MaxValue;

            long res = 1;
            long x = a;
            // REVIEW: Is the catch too slow in the overflow case?
            try
            {
                checked
                {
                    for (; ; )
                    {
                        if ((b & 1) != 0)
                            res *= x;
                        b >>= 1;
                        if (b == 0)
                            break;
                        x *= x;
                    }
                }
            }
            catch (OverflowException)
            {
                return neg ? I8.MinValue : I8.MaxValue;
            }
            Contracts.Assert(res > 0);

            if (neg)
                res = -res;
            return res;
        }
    }
}
