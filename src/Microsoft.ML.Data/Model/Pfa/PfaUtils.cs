// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.Model.Pfa
{
    public static class PfaUtils
    {
        public static JObject AddReturn(this JObject toEdit, string name, JToken value)
        {
            Contracts.CheckValueOrNull(toEdit);
            Contracts.CheckValue(name, nameof(name));
            Contracts.CheckValue(value, nameof(value));

            if (toEdit == null)
                toEdit = new JObject();
            toEdit.Add(name, value);
            return toEdit;
        }

        /// <summary>
        /// Generic facilities for calling a function.
        /// </summary>
        /// <param name="func">The function to call</param>
        /// <param name="prms">The parameters for the function</param>
        /// <returns></returns>
        public static JObject Call(string func, params JToken[] prms)
        {
            Contracts.CheckNonWhiteSpace(func, nameof(func));
            Contracts.CheckValue(prms, nameof(prms));
            var retval = new JObject();
            retval[func] = new JArray(prms);
            return retval;
        }

        public static JObject FuncRef(string func)
        {
            Contracts.CheckNonWhiteSpace(func, nameof(func));
            return ((JObject)null).AddReturn("fcn", func);
        }

        public static JObject Param(string name, JToken type)
        {
            var retval = new JObject();
            retval[name] = type;
            return retval;
        }

        public static JObject Index(JToken arrayOrMap, JToken key)
        {
            var retval = new JObject();
            retval["attr"] = arrayOrMap;
            var path = new JArray();
            path.Add(key);
            retval["path"] = path;
            return retval;
        }

        public static JObject String(string str)
        {
            Contracts.CheckValue(str, nameof(str));
            return ((JObject)null).AddReturn("type", Type.String).AddReturn("value", str);
        }

        public static JObject For(JObject initBlock, JObject whileBlock, JObject stepBlock, JObject doBlock)
        {
            var retval = new JObject();
            retval["for"] = initBlock;
            retval["while"] = whileBlock;
            retval["step"] = stepBlock;
            retval["do"] = doBlock;
            return retval;
        }

        public static JObject If(JToken condition, JToken thenBlock, JToken elseBlock)
        {
            var retval = new JObject();
            retval["if"] = condition;
            retval["then"] = thenBlock;
            if (elseBlock != null)
                retval["else"] = elseBlock;
            return retval;
        }

        /// <summary>
        /// Builds a "cast" statement to the two vector types.
        /// </summary>
        /// <param name="itemType">The type of the item in the vector</param>
        /// <param name="src">The token we are casting</param>
        /// <param name="asMapName">The name for the token as it will appear in the <paramref name="mapDo"/></param>
        /// <param name="mapDo">The map case expression</param>
        /// <param name="asArrName">The name for the token as it will appear in the <paramref name="arrDo"/></param>
        /// <param name="arrDo">The array case expression</param>
        /// <returns>The cast/case expression</returns>
        public static JObject VectorCase(JToken itemType, JToken src, string asMapName, JToken mapDo, string asArrName, JToken arrDo)
        {
            JObject jobj = null;
            var cases = new JArray();
            cases.Add(jobj.AddReturn("as", Type.Map(itemType)).AddReturn(
                "named", asMapName).AddReturn("do", mapDo));
            cases.Add(jobj.AddReturn("as", Type.Array(itemType)).AddReturn(
                "named", asArrName).AddReturn("do", arrDo));
            return jobj.AddReturn("cast", src).AddReturn("cases", cases);
        }

        public static JObject Cell(string name)
        {
            Contracts.CheckNonWhiteSpace(name, nameof(name));
            return ((JObject)null).AddReturn("cell", name);
        }

        public static class Type
        {
            public static readonly JToken Int = "int";
            public static readonly JToken Long = "long";
            public static readonly JToken Float = "float";
            public static readonly JToken Double = "double";
            public static readonly JToken Bool = "boolean";
            public static readonly JToken String = "string";
            public static readonly JToken Null = "null";

            public static JToken Map(JToken valueType)
            {
                Contracts.CheckValue(valueType, nameof(valueType));
                var retval = new JObject();
                retval["type"] = "map";
                retval["values"] = valueType;
                return retval;
            }

            public static JToken Array(JToken itemType)
            {
                Contracts.CheckValue(itemType, nameof(itemType));
                var retval = new JObject();
                retval["type"] = "array";
                retval["items"] = itemType;
                return retval;
            }

            public static JToken Union(params JToken[] types)
            {
                Contracts.CheckParam(Utils.Size(types) >= 2, nameof(types), "Union must have at least two types");
                return new JArray(types);
            }

            public static JToken Vector(JToken itemType)
            {
                Contracts.CheckValue(itemType, nameof(itemType));
                return Union(Map(itemType), Array(itemType));
            }

            public static JToken PfaTypeOrNullForColumnType(ColumnType type)
            {
                Contracts.CheckValue(type, nameof(type));
                if (type.IsVector)
                {
                    // We represent vectors as the union of array (for dense) and map (for sparse),
                    // of the appropriate item type.
                    var itemType = PfaTypeOrNullCore(type.ItemType);
                    if (itemType == null)
                        return null;
                    return Array(itemType);
                }
                return PfaTypeOrNullCore(type.ItemType);
            }

            private static JToken PfaTypeOrNullCore(ColumnType itemType)
            {
                Contracts.AssertValue(itemType);

                if (!itemType.IsPrimitive)
                    return null;

                if (itemType.IsKey)
                {
                    // Keys will retain the property that they are just numbers,
                    // with 0 representing missing.
                    if (itemType.KeyCount > 0 || itemType.RawKind != DataKind.U8)
                        return Int;
                    return Long;
                }

                switch (itemType.RawKind)
                {
                    case DataKind.I1:
                    case DataKind.U1:
                    case DataKind.I2:
                    case DataKind.U2:
                    case DataKind.I4:
                        return Int;
                    case DataKind.U4:
                    case DataKind.I8:
                    case DataKind.U8:
                        return Long;
                    case DataKind.R4:
                    // REVIEW: This should really be float. But, for the
                    // sake of the POC, we use double since all the PFA convenience
                    // libraries operate over doubles.
                    case DataKind.R8:
                        return Double;
                    case DataKind.BL:
                        return Bool;
                    case DataKind.TX:
                        return String;
                    default:
                        return null;
                }
            }

            public static JToken DefaultTokenOrNull(PrimitiveType itemType)
            {
                Contracts.CheckValue(itemType, nameof(itemType));

                if (itemType.IsKey)
                    return 0;

                switch (itemType.RawKind)
                {
                    case DataKind.I1:
                    case DataKind.U1:
                    case DataKind.I2:
                    case DataKind.U2:
                    case DataKind.I4:
                    case DataKind.U4:
                    case DataKind.I8:
                    case DataKind.U8:
                        return 0;
                    case DataKind.R4:
                    // REVIEW: This should really be float. But, for the
                    // sake of the POC, we use double since all the PFA convenience
                    // libraries operate over doubles.
                    case DataKind.R8:
                        return 0.0;
                    case DataKind.BL:
                        return false;
                    case DataKind.TX:
                        return String("");
                    default:
                        return null;
                }
            }
        }

        /// <summary>
        /// This ensures that there is a function formatted as "count_type" (for example, "count_double"),
        /// that takes either a map or array and returns the number of items in that map or array.
        /// </summary>
        /// <param name="ctx">The context to check for the existence of this</param>
        /// <param name="itemType">The item type this will operate on</param>
        public static string EnsureCount(this PfaContext ctx, JToken itemType)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.CheckValue(itemType, nameof(itemType));
            var name = "count_" + itemType.ToString();
            if (ctx.ContainsFunc(name))
                return "u." + name;
            ctx.AddFunc(name, new JArray(Param("a", Type.Vector(itemType))), Type.Int,
                VectorCase(itemType, "a", "ma", Call("map.len", "ma"), "aa", Call("a.len", "aa")));
            return "u." + name;
        }

        /// <summary>
        /// A string -> bool function for determining whether a string has content.
        /// </summary>
        public static string EnsureHasChars(this PfaContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            const string name = "hasChars";
            if (ctx.ContainsFunc(name))
                return "u." + name;
            ctx.AddFunc(name, new JArray(Param("str", Type.String)), Type.Bool,
                Call(">", Call("s.len", "str"), 0));
            return "u." + name;
        }

        public static string EnsureNewArray(this PfaContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            const string name = "hasChars";
            const string refname = "u." + name;
            if (ctx.ContainsFunc(name))
                return refname;
            var arrType = Type.Array(Type.Double);

            JObject jobj = null;

            JArray elseBlock = new JArray();
            elseBlock.Add(jobj.AddReturn("let", jobj.AddReturn("halfsize",
                Call(refname, Call("//", "size", 2)))));
            elseBlock.Add(jobj.AddReturn("let", jobj.AddReturn("fullsize",
                Call("a.concat", "halfsize", "halfsize"))));
            elseBlock.Add(If(
                Call("==", Call("&", "size", 1), 1),
                Call("a.append", "fullsize", 0.0), "fullsize"));

            ctx.AddFunc(name, new JArray(Param("size", Type.Int)), arrType,
                If(Call("==", "size", 0), jobj.AddReturn("type", arrType).AddReturn("value", new JArray()),
                elseBlock));
            return refname;
        }

        public static string EnsureAdd(this PfaContext ctx, JToken itemType)
            => EnsureOpCore(ctx, "add", "+", itemType);
        public static string EnsureSub(this PfaContext ctx, JToken itemType)
            => EnsureOpCore(ctx, "sub", "-", itemType);
        public static string EnsureMul(this PfaContext ctx, JToken itemType)
            => EnsureOpCore(ctx, "mul", "*", itemType);
        public static string EnsureDiv(this PfaContext ctx, JToken itemType)
            => EnsureOpCore(ctx, "div", "/", itemType);

        private static string EnsureOpCore(PfaContext ctx, string funcPrefix, string binOp, JToken itemType, JToken returnType = null)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            Contracts.AssertNonEmpty(funcPrefix);
            Contracts.AssertNonEmpty(binOp);
            Contracts.CheckValue(itemType, nameof(itemType));
            Contracts.CheckValueOrNull(returnType);
            returnType = returnType ?? itemType;

            var name = funcPrefix + "_" + itemType.ToString();
            if (ctx.ContainsFunc(name))
                return "u." + name;
            ctx.AddFunc(name, new JArray(Param("a", itemType), Param("b", itemType)), returnType, Call(binOp, "a", "b"));
            return "u." + name;
        }
    }
}