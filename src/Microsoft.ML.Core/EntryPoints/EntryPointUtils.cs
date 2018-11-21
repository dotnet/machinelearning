// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.EntryPoints
{
    public static class EntryPointUtils
    {
        private static bool IsValueWithinRange<T>(TlcModule.RangeAttribute range, object obj)
        {
            T val;
            if (obj is Optional<T> asOptional)
                val = asOptional.Value;
            else
                val = (T)obj;

            return
                (range.Min == null || ((IComparable)range.Min).CompareTo(val) <= 0) &&
                (range.Inf == null || ((IComparable)range.Inf).CompareTo(val) < 0) &&
                (range.Max == null || ((IComparable)range.Max).CompareTo(val) >= 0) &&
                (range.Sup == null || ((IComparable)range.Sup).CompareTo(val) > 0);
        }

        public static bool IsValueWithinRange(this TlcModule.RangeAttribute range, object val)
        {
            Contracts.AssertValue(range);
            Contracts.AssertValue(val);
            Func<TlcModule.RangeAttribute, object, bool> fn = IsValueWithinRange<int>;
            // Avoid trying to cast double as float. If range
            // was specified using floats, but value being checked
            // is double, change range to be of type double
            if (range.Type == typeof(float) && val is double)
                range.CastToDouble();
            return Utils.MarshalInvoke(fn, range.Type, range, val);
        }

        /// <summary>
        /// Performs checks on an EntryPoint input class equivilent to the checks that are done
        /// when parsing a JSON EntryPoint graph.
        ///
        /// Call this method from EntryPoint methods to ensure that range and required checks are performed
        /// in a consistent manner when EntryPoints are created directly from code.
        /// </summary>
        public static void CheckInputArgs(IExceptionContext ectx, object args)
        {
            foreach (var fieldInfo in args.GetType().GetFields())
            {
                var attr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault()
                    as ArgumentAttribute;
                if (attr == null || attr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;

                var fieldVal = fieldInfo.GetValue(args);
                var fieldType = fieldInfo.FieldType;

                // Optionals are either left in their Implicit constructed state or
                // a new Explicit optional is constructed. They should never be set
                // to null.
                if (fieldType.IsGenericType && fieldType.GetGenericTypeDefinition() == typeof(Optional<>) && fieldVal == null)
                    throw ectx.Except("Field '{0}' is Optional<> and set to null instead of an explicit value.", fieldInfo.Name);

                if (attr.IsRequired)
                {
                    bool equalToDefault;
                    if (fieldType.IsGenericType && fieldType.GetGenericTypeDefinition() == typeof(Optional<>))
                        equalToDefault = !((Optional)fieldVal).IsExplicit;
                    else
                        equalToDefault = fieldType.IsValueType ? Activator.CreateInstance(fieldType).Equals(fieldVal) : fieldVal == null;

                    if (equalToDefault)
                        throw ectx.Except("Field '{0}' is required but is not set.", fieldInfo.Name);
                }

                var rangeAttr = fieldInfo.GetCustomAttributes(typeof(TlcModule.RangeAttribute), false).FirstOrDefault()
                    as TlcModule.RangeAttribute;
                if (rangeAttr != null && fieldVal != null && !rangeAttr.IsValueWithinRange(fieldVal))
                    throw ectx.Except("Field '{0}' is set to a value that falls outside the range bounds.", fieldInfo.Name);
            }
        }

        public static IHost CheckArgsAndCreateHost(IHostEnvironment env, string hostName, object input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(hostName);
            host.CheckValue(input, nameof(input));
            CheckInputArgs(host, input);
            return host;
        }

        /// <summary>
        /// Searches for the given column name in the schema. This method applies a
        /// common policy that throws an exception if the column is not found
        /// and the column name was explicitly specified. If the column is not found
        /// and the column name was not explicitly specified, it returns null.
        /// </summary>
        public static string FindColumnOrNull(IExceptionContext ectx, Schema schema, Optional<string> value)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(schema, nameof(schema));
            ectx.CheckValue(value, nameof(value));

            if (value == "")
                return null;
            int col;
            if (!schema.TryGetColumnIndex(value, out col))
            {
                if (value.IsExplicit)
                    throw ectx.Except("Column '{0}' not found", value);
                return null;
            }
            return value;
        }

        /// <summary>
        /// Converts EntryPoint Optional{T} types into nullable types, with the
        /// implicit value being converted to the null value.
        /// </summary>
        public static T? AsNullable<T>(this Optional<T> opt) where T : struct
        {
            if (opt.IsExplicit)
                return opt.Value;
            else
                return null;
        }
    }
}
