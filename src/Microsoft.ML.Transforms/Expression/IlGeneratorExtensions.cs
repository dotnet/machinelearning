// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using System.Reflection.Emit;
using Microsoft.ML.Runtime;

#pragma warning disable MSML_GeneralName // The names are derived from .NET OpCode names. These do not adhere to .NET naming standards.
namespace Microsoft.ML.Transforms
{
    /// <summary>
    /// Helper extension methods for using ILGenerator.
    /// Rather than typing out something like:
    ///   il.Emit(OpCodes.Ldarg_0);
    ///   il.Emit(OpCodes.Ldarg_1);
    ///   il.Emit(OpCodes.Ldc_I4, i);
    ///   il.Emit(OpCodes.Ldelem_Ref);
    ///   il.Emit(OpCodes.Stfld, literalFields[i]);
    /// You can do:
    ///   il
    ///       .Ldarg(0)
    ///       .Ldarg(1)
    ///       .Ldc_I4(i)
    ///       .Ldelem_Ref()
    ///       .Stfld(literalFields[i]);
    /// It also provides some type safety over the Emit methods by ensuring
    /// that you don't pass any args when using Add or that you only
    /// pass a Label when using Br.
    /// </summary>
    internal static class ILGeneratorExtensions
    {
        public static ILGenerator Add(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Add);
            return il;
        }

        public static ILGenerator Beq(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Beq, label);
            return il;
        }

        public static ILGenerator Bge(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Bge, label);
            return il;
        }

        public static ILGenerator Bge_Un(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Bge_Un, label);
            return il;
        }

        public static ILGenerator Bgt(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Bgt, label);
            return il;
        }

        public static ILGenerator Bgt_Un(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Bgt_Un, label);
            return il;
        }

        public static ILGenerator Ble(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Ble, label);
            return il;
        }

        public static ILGenerator Ble_Un(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Ble_Un, label);
            return il;
        }

        public static ILGenerator Blt(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Blt, label);
            return il;
        }

        public static ILGenerator Blt_Un(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Blt_Un, label);
            return il;
        }

        public static ILGenerator Bne_Un(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Bne_Un, label);
            return il;
        }

        public static ILGenerator Br(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Br, label);
            return il;
        }

        public static ILGenerator Brfalse(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Brfalse, label);
            return il;
        }

        public static ILGenerator Brtrue(this ILGenerator il, Label label)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Brtrue, label);
            return il;
        }

        public static ILGenerator Call(this ILGenerator il, MethodInfo info)
        {
            Contracts.AssertValue(il);
            Contracts.AssertValue(info);
            Contracts.Assert(!info.IsVirtual);
            il.Emit(OpCodes.Call, info);
            return il;
        }

        public static ILGenerator Ceq(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Ceq);
            return il;
        }

        public static ILGenerator Cgt(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Cgt);
            return il;
        }

        public static ILGenerator Cgt_Un(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Cgt_Un);
            return il;
        }

        public static ILGenerator Clt(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Clt);
            return il;
        }

        public static ILGenerator Clt_Un(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Clt_Un);
            return il;
        }

        public static ILGenerator Conv_I8(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Conv_I8);
            return il;
        }

        public static ILGenerator Conv_R4(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Conv_R4);
            return il;
        }

        public static ILGenerator Conv_R8(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Conv_R8);
            return il;
        }

        public static ILGenerator Div(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Div);
            return il;
        }

        public static ILGenerator Dup(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Dup);
            return il;
        }

        public static ILGenerator Ldarg(this ILGenerator il, int arg)
        {
            Contracts.AssertValue(il);
            Contracts.Assert(0 <= arg && arg <= short.MaxValue);

            switch (arg)
            {
                case 0:
                    il.Emit(OpCodes.Ldarg_0);
                    break;
                case 1:
                    il.Emit(OpCodes.Ldarg_1);
                    break;
                case 2:
                    il.Emit(OpCodes.Ldarg_2);
                    break;
                case 3:
                    il.Emit(OpCodes.Ldarg_3);
                    break;
                default:
                    if (arg <= byte.MaxValue)
                        il.Emit(OpCodes.Ldarg_S, (byte)arg);
                    else
                        il.Emit(OpCodes.Ldarg, (short)arg);
                    break;
            }
            return il;
        }

        public static ILGenerator Ldc_I4(this ILGenerator il, int arg)
        {
            Contracts.AssertValue(il);

            switch (arg)
            {
                case -1:
                    il.Emit(OpCodes.Ldc_I4_M1);
                    break;
                case 0:
                    il.Emit(OpCodes.Ldc_I4_0);
                    break;
                case 1:
                    il.Emit(OpCodes.Ldc_I4_1);
                    break;
                case 2:
                    il.Emit(OpCodes.Ldc_I4_2);
                    break;
                case 3:
                    il.Emit(OpCodes.Ldc_I4_3);
                    break;
                case 4:
                    il.Emit(OpCodes.Ldc_I4_4);
                    break;
                case 5:
                    il.Emit(OpCodes.Ldc_I4_5);
                    break;
                case 6:
                    il.Emit(OpCodes.Ldc_I4_6);
                    break;
                case 7:
                    il.Emit(OpCodes.Ldc_I4_7);
                    break;
                case 8:
                    il.Emit(OpCodes.Ldc_I4_8);
                    break;
                default:
                    // REVIEW: Docs say use ILGenerator.Emit(OpCode, byte) even though the value is signed
                    if (sbyte.MinValue <= arg && arg <= sbyte.MaxValue)
                        il.Emit(OpCodes.Ldc_I4_S, (byte)arg);
                    else
                        il.Emit(OpCodes.Ldc_I4, arg);
                    break;
            }
            return il;
        }

        public static ILGenerator Ldc_I8(this ILGenerator il, long arg)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Ldc_I8, arg);
            return il;
        }

        public static ILGenerator Ldc_R4(this ILGenerator il, float arg)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Ldc_R4, arg);
            return il;
        }

        public static ILGenerator Ldc_R8(this ILGenerator il, double arg)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Ldc_R8, arg);
            return il;
        }

        public static ILGenerator Ldloc(this ILGenerator il, LocalBuilder builder)
        {
            Contracts.AssertValue(il);
            Contracts.AssertValue(builder);
            il.Emit(OpCodes.Ldloc, builder);
            return il;
        }

        public static ILGenerator Ldstr(this ILGenerator il, string str)
        {
            Contracts.AssertValue(il);
            Contracts.AssertValue(str);
            il.Emit(OpCodes.Ldstr, str);
            return il;
        }

        public static ILGenerator Mul(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Mul);
            return il;
        }

        public static ILGenerator Mul_Ovf(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Mul_Ovf);
            return il;
        }

        public static ILGenerator Neg(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Neg);
            return il;
        }

        public static ILGenerator Newarr(this ILGenerator il, Type type)
        {
            Contracts.AssertValue(il);
            Contracts.AssertValue(type);
            il.Emit(OpCodes.Newarr, type);
            return il;
        }

        public static ILGenerator Pop(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Pop);
            return il;
        }

        public static ILGenerator Rem(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Rem);
            return il;
        }

        public static ILGenerator Ret(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Ret);
            return il;
        }

        public static ILGenerator Stelem(this ILGenerator il, Type type)
        {
            Contracts.AssertValue(il);
            Contracts.AssertValue(type);
            il.Emit(OpCodes.Stelem, type);
            return il;
        }

        public static ILGenerator Stloc(this ILGenerator il, LocalBuilder builder)
        {
            Contracts.AssertValue(il);
            Contracts.AssertValue(builder);
            il.Emit(OpCodes.Stloc, builder);
            return il;
        }

        public static ILGenerator Sub(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Sub);
            return il;
        }

        public static ILGenerator Xor(this ILGenerator il)
        {
            Contracts.AssertValue(il);
            il.Emit(OpCodes.Xor);
            return il;
        }
    }
}
#pragma warning restore MSML_GeneralName
