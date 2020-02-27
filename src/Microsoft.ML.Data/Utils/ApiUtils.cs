// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    internal delegate void Peek<in TRow, TValue>(TRow row, long position, ref TValue value);

    internal delegate void Poke<TRow, TValue>(TRow dst, TValue src);

    internal static class ApiUtils
    {
        private static OpCode GetAssignmentOpCode(Type t, IEnumerable<Attribute> attributes)
        {
            // REVIEW: This should be a Dictionary<Type, OpCode> based solution.
            // DvTypes, strings, arrays, all nullable types, VBuffers and RowId.
            if (t == typeof(ReadOnlyMemory<char>) || t == typeof(string) || t.IsArray ||
                (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(VBuffer<>)) ||
                (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Nullable<>)) ||
                t == typeof(DateTime) || t == typeof(DateTimeOffset) || t == typeof(TimeSpan) ||
                t == typeof(DataViewRowId) || DataViewTypeManager.Knows(t, attributes))
            {
                return OpCodes.Stobj;
            }

            // Simple primitive types.
            if (t == typeof(Single))
                return OpCodes.Stind_R4;
            if (t == typeof(Double))
                return OpCodes.Stind_R8;
            if (t == typeof(sbyte) || t == typeof(byte) || t == typeof(bool))
                return OpCodes.Stind_I1;
            if (t == typeof(short) || t == typeof(ushort))
                return OpCodes.Stind_I2;
            if (t == typeof(int) || t == typeof(uint))
                return OpCodes.Stind_I4;
            if (t == typeof(long) || t == typeof(ulong))
                return OpCodes.Stind_I8;
            throw Contracts.ExceptNotSupp("Type '{0}' is not supported.", t.FullName);
        }

        /// <summary>
        /// Each of the specialized 'peek' methods copies the appropriate field value of an instance of T
        /// into the provided buffer. So, the call is 'peek(userObject, ref destination)' and the logic is
        /// indentical to 'destination = userObject.##FIELD##', where ##FIELD## is defined per peek method.
        /// </summary>
        internal static Delegate GeneratePeek<TOwn, TRow>(InternalSchemaDefinition.Column column)
        {
            switch (column.MemberInfo)
            {
                case FieldInfo fieldInfo:
                    Type fieldType = fieldInfo.FieldType;

                    var assignmentOpCode = GetAssignmentOpCode(fieldType, fieldInfo.GetCustomAttributes());
                    Func<FieldInfo, OpCode, Delegate> func = GeneratePeek<TOwn, TRow, int>;
                    var methInfo = func.GetMethodInfo().GetGenericMethodDefinition()
                        .MakeGenericMethod(typeof(TOwn), typeof(TRow), fieldType);
                    return (Delegate)methInfo.Invoke(null, new object[] { fieldInfo, assignmentOpCode });

                case PropertyInfo propertyInfo:
                    Type propertyType = propertyInfo.PropertyType;

                    var assignmentOpCodeProp = GetAssignmentOpCode(propertyType, propertyInfo.GetCustomAttributes());
                    Func<PropertyInfo, OpCode, Delegate> funcProp = GeneratePeek<TOwn, TRow, int>;
                    var methInfoProp = funcProp.GetMethodInfo().GetGenericMethodDefinition()
                        .MakeGenericMethod(typeof(TOwn), typeof(TRow), propertyType);
                    return (Delegate)methInfoProp.Invoke(null, new object[] { propertyInfo, assignmentOpCodeProp });

                default:
                    Contracts.Assert(false);
                    throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");

            }
        }

        private static Delegate GeneratePeek<TOwn, TRow, TValue>(FieldInfo fieldInfo, OpCode assignmentOpCode)
        {
            // REVIEW: It seems like we really should cache these, instead of generating them per cursor.
            Type[] args = { typeof(TOwn), typeof(TRow), typeof(long), typeof(TValue).MakeByRefType() };
            var mb = new DynamicMethod("Peek", null, args, typeof(TOwn), true);
            var il = mb.GetILGenerator();

            il.Emit(OpCodes.Ldarg_3);               // push arg3
            il.Emit(OpCodes.Ldarg_1);               // push arg1
            il.Emit(OpCodes.Ldfld, fieldInfo);      // push [stack top].[fieldInfo]
            // Stobj needs to coupled with a type.
            if (assignmentOpCode == OpCodes.Stobj)  // [stack top-1] = [stack top]
                il.Emit(assignmentOpCode, fieldInfo.FieldType);
            else
                il.Emit(assignmentOpCode);
            il.Emit(OpCodes.Ret);                   // ret

            return mb.CreateDelegate(typeof(Peek<TRow, TValue>));
        }

        private static Delegate GeneratePeek<TOwn, TRow, TValue>(PropertyInfo propertyInfo, OpCode assignmentOpCode)
        {
            // REVIEW: It seems like we really should cache these, instead of generating them per cursor.
            Type[] args = { typeof(TOwn), typeof(TRow), typeof(long), typeof(TValue).MakeByRefType() };
            var mb = new DynamicMethod("Peek", null, args, typeof(TOwn), true);
            var il = mb.GetILGenerator();
            var minfo = propertyInfo.GetGetMethod();
            var opcode = (minfo.IsVirtual || minfo.IsAbstract) ? OpCodes.Callvirt : OpCodes.Call;

            il.Emit(OpCodes.Ldarg_3);               // push arg3
            il.Emit(OpCodes.Ldarg_1);               // push arg1
            il.Emit(opcode, minfo);                 // call [stack top].get_[propertyInfo]()
            // Stobj needs to coupled with a type.
            if (assignmentOpCode == OpCodes.Stobj)  // [stack top-1] = [stack top]
                il.Emit(assignmentOpCode, propertyInfo.PropertyType);
            else
                il.Emit(assignmentOpCode);
            il.Emit(OpCodes.Ret);                   // ret

            return mb.CreateDelegate(typeof(Peek<TRow, TValue>));
        }

        /// <summary>
        /// Each of the specialized 'poke' methods sets the appropriate field value of an instance of T
        /// to the provided value. So, the call is 'peek(userObject, providedValue)' and the logic is
        /// identical to 'userObject.##FIELD## = providedValue', where ##FIELD## is defined per poke method.
        /// </summary>
        internal static Delegate GeneratePoke<TOwn, TRow>(InternalSchemaDefinition.Column column)
        {
            switch (column.MemberInfo)
            {
                case FieldInfo fieldInfo:
                    Type fieldType = fieldInfo.FieldType;

                    var assignmentOpCode = GetAssignmentOpCode(fieldType, fieldInfo.GetCustomAttributes());
                    Func<FieldInfo, OpCode, Delegate> func = GeneratePoke<TOwn, TRow, int>;
                    var methInfo = func.GetMethodInfo().GetGenericMethodDefinition()
                        .MakeGenericMethod(typeof(TOwn), typeof(TRow), fieldType);
                    return (Delegate)methInfo.Invoke(null, new object[] { fieldInfo, assignmentOpCode });

                case PropertyInfo propertyInfo:
                    Type propertyType = propertyInfo.PropertyType;

                    var assignmentOpCodeProp = GetAssignmentOpCode(propertyType, propertyInfo.GetCustomAttributes());
                    Func<PropertyInfo, Delegate> funcProp = GeneratePoke<TOwn, TRow, int>;
                    var methInfoProp = funcProp.GetMethodInfo().GetGenericMethodDefinition()
                        .MakeGenericMethod(typeof(TOwn), typeof(TRow), propertyType);
                    return (Delegate)methInfoProp.Invoke(null, new object[] { propertyInfo });

                default:
                    Contracts.Assert(false);
                    throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
            }
        }

        private static Delegate GeneratePoke<TOwn, TRow, TValue>(FieldInfo fieldInfo, OpCode assignmentOpCode)
        {
            Type[] args = { typeof(TOwn), typeof(TRow), typeof(TValue) };
            var mb = new DynamicMethod("Poke", null, args, typeof(TOwn), true);
            var il = mb.GetILGenerator();

            il.Emit(OpCodes.Ldarg_1);               // push arg1
            il.Emit(OpCodes.Ldflda, fieldInfo);     // push addr([stack top].[fieldInfo])
            il.Emit(OpCodes.Ldarg_2);               // push arg2
            // Stobj needs to coupled with a type.
            if (assignmentOpCode == OpCodes.Stobj)  // [stack top-1] = [stack top]
                il.Emit(assignmentOpCode, fieldInfo.FieldType);
            else
                il.Emit(assignmentOpCode);
            il.Emit(OpCodes.Ret);                   // ret
            return mb.CreateDelegate(typeof(Poke<TRow, TValue>), null);
        }

        private static Delegate GeneratePoke<TOwn, TRow, TValue>(PropertyInfo propertyInfo)
        {
            Type[] args = { typeof(TOwn), typeof(TRow), typeof(TValue) };
            var mb = new DynamicMethod("Poke", null, args, typeof(TOwn), true);
            var il = mb.GetILGenerator();
            var minfo = propertyInfo.GetSetMethod();
            var opcode = (minfo.IsVirtual || minfo.IsAbstract) ? OpCodes.Callvirt : OpCodes.Call;

            il.Emit(OpCodes.Ldarg_1);               // push arg1
            il.Emit(OpCodes.Ldarg_2);               // push arg2
            il.Emit(opcode, minfo);                 // call [stack top-1].set_[propertyInfo]([stack top])
            il.Emit(OpCodes.Ret);                   // ret
            return mb.CreateDelegate(typeof(Poke<TRow, TValue>), null);
        }
    }
}
