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
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.EntryPoints.JsonUtils
{
    /// <summary>
    /// The class that creates and wraps around an instance of an input object and gradually populates all fields, keeping track of missing 
    /// required values. The values can be set from their JSON representation (during the graph parsing stage), as well as directly 
    /// (in the process of graph execution).
    /// </summary>
    public sealed class InputBuilder
    {
        private struct Attributes
        {
            public readonly ArgumentAttribute Input;
            public readonly TlcModule.RangeAttribute Range;
            public readonly bool Optional;

            public Attributes(ArgumentAttribute input, TlcModule.RangeAttribute range, bool optional = false)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValueOrNull(range);
                Input = input;
                Range = range;
                Optional = optional;
            }
        }

        private readonly IExceptionContext _ectx;
        private readonly object _instance;
        private readonly Type _type;

        private readonly FieldInfo[] _fields;
        private readonly bool[] _wasSet;
        private readonly Attributes[] _attrs;
        private readonly ModuleCatalog _catalog;

        public InputBuilder(IExceptionContext ectx, Type inputType, ModuleCatalog catalog)
        {
            Contracts.CheckValue(ectx, nameof(ectx));
            _ectx = ectx;
            _ectx.CheckValue(inputType, nameof(inputType));
            _ectx.CheckValue(catalog, nameof(catalog));

            _type = inputType;
            _catalog = catalog;

            var fields = new List<FieldInfo>();
            var attrs = new List<Attributes>();

            foreach (var fieldInfo in _type.GetFields())
            {
                var attr = (ArgumentAttribute)fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault();
                if (attr == null || attr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;
                _ectx.Check(!fieldInfo.IsStatic && !fieldInfo.IsInitOnly && !fieldInfo.IsLiteral);

                var rangeAttr = fieldInfo.GetCustomAttributes(typeof(TlcModule.RangeAttribute), false).FirstOrDefault()
                    as TlcModule.RangeAttribute;
                Contracts.CheckValueOrNull(rangeAttr);

                var optional = fieldInfo.GetCustomAttributes(typeof(TlcModule.OptionalInputAttribute), false).Any();

                fields.Add(fieldInfo);
                attrs.Add(new Attributes(attr, rangeAttr, optional));
            }
            _ectx.Assert(fields.Count == attrs.Count);

            _instance = Activator.CreateInstance(inputType);
            _fields = fields.ToArray();
            _attrs = attrs.ToArray();
            _wasSet = new bool[_fields.Length];
        }

        private static bool AnyMatch(string name, string[] aliases)
        {
            if (aliases == null)
                return false;
            return aliases.Any(a => string.Equals(name, a, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Retreives the field index for a field with the given alias, or -1 if
        /// that field alias is not found.
        /// </summary>
        private int GetFieldIndex(string name)
        {
            _ectx.AssertNonEmpty(name);
            for (int i = 0; i < _attrs.Length; i++)
            {
                if (name == (_attrs[i].Input.Name ?? _fields[i].Name) || AnyMatch(name, _attrs[i].Input.Aliases))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Returns the Type of the given field, unwrapping any option
        /// types to be of their inner type. If the given alias doesn't exist
        /// this method returns null.
        /// </summary>
        public Type GetFieldTypeOrNull(string alias)
        {
            _ectx.CheckNonEmpty(alias, nameof(alias));

            var fi = GetFieldIndex(alias);
            if (fi < 0)
                return null;

            var type = _fields[fi].FieldType;
            if (type.IsGenericType &&
                (type.GetGenericTypeDefinition() == typeof(Optional<>) ||
                 type.GetGenericTypeDefinition() == typeof(Var<>) ||
                 type.GetGenericTypeDefinition() == typeof(Nullable<>)))
            {
                type = type.GetGenericArguments()[0];
            }

            return type;
        }

        public string GetFieldNameOrNull(string alias)
        {
            _ectx.CheckNonEmpty(alias, nameof(alias));

            var fi = GetFieldIndex(alias);
            return fi >= 0 ? _fields[fi].Name : null;
        }

        /// <summary>
        /// Returns the array of required values that were not specified using <see cref="TrySetValue"/>.
        /// </summary>
        public string[] GetMissingValues()
        {
            var missing = new List<string>();
            for (int i = 0; i < _fields.Length; i++)
            {
                var field = _fields[i];
                var attr = _attrs[i];
                if (attr.Input.IsRequired && !_wasSet[i])
                    missing.Add(attr.Input.Name ?? field.Name);
            }

            return missing.ToArray();
        }

        public bool IsInputOptional(string name)
        {
            var index = GetFieldIndex(name);
            if (index < 0)
                throw Contracts.Except($"Unknown input name {name}");
            return _attrs[index].Optional;
        }

        /// <summary>
        /// Set a value of a field specified by <paramref name="name"/> by parsing <paramref name="value"/>.
        /// </summary>
        public bool TrySetValueJson(string name, JToken value)
        {
            _ectx.CheckNonEmpty(name, nameof(name));
            _ectx.CheckValue(value, nameof(value));

            var index = GetFieldIndex(name);
            if (index < 0)
                return false;

            var field = _fields[index];
            // REVIEW: This method implies that it'll return a friendly bool for most
            // failure modes, but ParseJsonValue and GetFieldAssignableValue both throw if
            // types don't match up. Mixed failure modes are hostile to clients of this method.
            var csValue = ParseJsonValue(_ectx, field.FieldType, _attrs[index], value, _catalog);

            if (_attrs[index].Range != null && csValue != null && !_attrs[index].Range.IsValueWithinRange(csValue))
                return false;

            csValue = GetFieldAssignableValue(_ectx, field.FieldType, csValue);
            field.SetValue(_instance, csValue);
            _wasSet[index] = true;
            return true;
        }

        /// <summary>
        /// Set a value of a field specified by <paramref name="name"/> directly to <paramref name="value"/>.
        /// </summary>
        public bool TrySetValue(string name, object value)
        {
            _ectx.CheckNonEmpty(name, nameof(name));
            _ectx.CheckValueOrNull(value);
            var index = GetFieldIndex(name);
            if (index < 0)
                return false;

            var field = _fields[index];
            var csValue = GetFieldAssignableValue(_ectx, field.FieldType, value);
            field.SetValue(_instance, csValue);
            _wasSet[index] = true;
            return true;
        }

        public JObject GetJsonObject(object instance, Dictionary<string, List<ParameterBinding>> inputBindingMap, Dictionary<ParameterBinding, VariableBinding> inputMap)
        {
            Contracts.CheckValue(instance, nameof(instance));
            Contracts.Check(instance.GetType() == _type);

            var result = new JObject();
            var defaults = Activator.CreateInstance(_type);
            for (int i = 0; i < _fields.Length; i++)
            {
                var field = _fields[i];
                var attr = _attrs[i];
                var instanceVal = field.GetValue(instance);
                var defaultsVal = field.GetValue(defaults);

                if (inputBindingMap.TryGetValue(field.Name, out List<ParameterBinding> bindings))
                {
                    // Handle variables.
                    Contracts.Assert(bindings.Count > 0);
                    VariableBinding varBinding;
                    var paramBinding = bindings[0];
                    if (paramBinding is SimpleParameterBinding)
                    {
                        Contracts.Assert(bindings.Count == 1);
                        bool success = inputMap.TryGetValue(paramBinding, out varBinding);
                        Contracts.Assert(success);
                        Contracts.AssertValue(varBinding);

                        result.Add(field.Name, new JValue(varBinding.ToJson()));
                    }
                    else if (paramBinding is ArrayIndexParameterBinding)
                    {
                        // Array parameter bindings.
                        var array = new JArray();
                        foreach (var parameterBinding in bindings)
                        {
                            Contracts.Assert(parameterBinding is ArrayIndexParameterBinding);
                            bool success = inputMap.TryGetValue(parameterBinding, out varBinding);
                            Contracts.Assert(success);
                            Contracts.AssertValue(varBinding);
                            array.Add(new JValue(varBinding.ToJson()));
                        }

                        result.Add(field.Name, array);
                    }
                    else
                    {
                        // Dictionary parameter bindings. Not supported yet.
                        Contracts.Assert(paramBinding is DictionaryKeyParameterBinding);
                        throw Contracts.ExceptNotImpl("Dictionary of variables not yet implemented.");
                    }
                }
                else if (instanceVal == null && defaultsVal != null)
                {
                    // Handle null values.
                    result.Add(field.Name, new JValue(instanceVal));
                }
                else if (instanceVal != null && (attr.Input.IsRequired || !instanceVal.Equals(defaultsVal)))
                {
                    // A required field will be serialized regardless of whether or not its value is identical to the default.
                    var type = instanceVal.GetType();
                    if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Optional<>))
                    {
                        var isExplicit = ExtractOptional(ref instanceVal, ref type);
                        if (!isExplicit)
                            continue;
                    }

                    if (type == typeof(JArray))
                        result.Add(field.Name, (JArray)instanceVal);
                    else if (type.IsGenericType &&
                        ((type.GetGenericTypeDefinition() == typeof(Var<>)) ||
                        type.GetGenericTypeDefinition() == typeof(ArrayVar<>) ||
                        type.GetGenericTypeDefinition() == typeof(DictionaryVar<>)))
                    {
                        result.Add(field.Name, new JValue($"${((IVarSerializationHelper)instanceVal).VarName}"));
                    }
                    else if (type == typeof(bool) ||
                        type == typeof(string) ||
                        type == typeof(char) ||
                        type == typeof(double) ||
                        type == typeof(float) ||
                        type == typeof(int) ||
                        type == typeof(long) ||
                        type == typeof(uint) ||
                        type == typeof(ulong))
                    {
                        // Handle simple types.
                        result.Add(field.Name, new JValue(instanceVal));
                    }
                    else if (type.IsEnum)
                    {
                        // Handle enums.
                        result.Add(field.Name, new JValue(instanceVal.ToString()));
                    }
                    else if (type.IsArray)
                    {
                        // Handle arrays.
                        var array = (Array)instanceVal;
                        var jarray = new JArray();
                        var elementType = type.GetElementType();
                        if (elementType == typeof(bool) ||
                            elementType == typeof(string) ||
                            elementType == typeof(char) ||
                            elementType == typeof(double) ||
                            elementType == typeof(float) ||
                            elementType == typeof(int) ||
                            elementType == typeof(long) ||
                            elementType == typeof(uint) ||
                            elementType == typeof(ulong))
                        {
                            foreach (object item in array)
                                jarray.Add(new JValue(item));
                        }
                        else
                        {
                            var builder = new InputBuilder(_ectx, elementType, _catalog);
                            foreach (object item in array)
                                jarray.Add(builder.GetJsonObject(item, inputBindingMap, inputMap));
                        }
                        result.Add(field.Name, jarray);
                    }
                    else if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Dictionary<,>) &&
                             type.GetGenericArguments()[0] == typeof(string))
                    {
                        // Handle dictionaries.
                        // REVIEW: Needs to be implemented when we will have entry point arguments that contain dictionaries.
                    }
                    else if (typeof(IComponentFactory).IsAssignableFrom(type))
                    {
                        // Handle component factories.
                        bool success = _catalog.TryFindComponent(type, out ModuleCatalog.ComponentInfo instanceInfo);
                        Contracts.Assert(success);
                        var builder = new InputBuilder(_ectx, type, _catalog);
                        var instSettings = builder.GetJsonObject(instanceVal, inputBindingMap, inputMap);

                        ModuleCatalog.ComponentInfo defaultInfo = null;
                        JObject defSettings = new JObject();
                        if (defaultsVal != null)
                        {
                            var deftype = defaultsVal.GetType();
                            if (deftype.IsGenericType && deftype.GetGenericTypeDefinition() == typeof(Optional<>))
                                ExtractOptional(ref defaultsVal, ref deftype);
                            success = _catalog.TryFindComponent(deftype, out defaultInfo);
                            Contracts.Assert(success);
                            builder = new InputBuilder(_ectx, deftype, _catalog);
                            defSettings = builder.GetJsonObject(defaultsVal, inputBindingMap, inputMap);
                        }

                        if (instanceInfo.Name != defaultInfo?.Name || instSettings.ToString() != defSettings.ToString())
                        {
                            var jcomponent = new JObject
                            {
                                { FieldNames.Name, new JValue(instanceInfo.Name) }
                            };
                            if (instSettings.ToString() != defSettings.ToString())
                                jcomponent.Add(FieldNames.Settings, instSettings);
                            result.Add(field.Name, jcomponent);
                        }
                    }
                    else
                    {
                        // REVIEW: pass in the bindings once we support variables in inner fields.

                        // Handle structs.
                        var builder = new InputBuilder(_ectx, type, _catalog);
                        result.Add(field.Name, builder.GetJsonObject(instanceVal, new Dictionary<string, List<ParameterBinding>>(),
                            new Dictionary<ParameterBinding, VariableBinding>()));
                    }
                }
            }

            return result;
        }

        private static bool ExtractOptional(ref object value, ref Type type)
        {
            Contracts.Assert(type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Optional<>));
            type = type.GetGenericArguments()[0];
            var optObj = value as Optional;
            value = optObj.GetValue();
            return optObj.IsExplicit;
        }

        private static object ParseJsonValue(IExceptionContext ectx, Type type, Attributes attributes, JToken value, ModuleCatalog catalog)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(type);
            ectx.AssertValueOrNull(value);
            ectx.AssertValue(catalog);

            if (value == null)
                return null;

            if (value is JValue val && val.Value == null)
                return null;

            if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(Optional<>) || type.GetGenericTypeDefinition() == typeof(Nullable<>)))
            {
                if (type.GetGenericTypeDefinition() == typeof(Optional<>) && value.HasValues)
                    value = value.Values().FirstOrDefault();
                type = type.GetGenericArguments()[0];
            }

            if (type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(Var<>)))
            {
                string varName = value.Value<string>();
                ectx.Check(VariableBinding.IsBindingToken(value), "Variable name expected.");
                var variable = Activator.CreateInstance(type) as IVarSerializationHelper;
                var varBinding = VariableBinding.Create(ectx, varName);
                variable.VarName = varBinding.VariableName;
                return variable;
            }

            if (type == typeof(JArray) && value is JArray)
                return value;

            TlcModule.DataKind dt = TlcModule.GetDataType(type);

            try
            {
                switch (dt)
                {
                case TlcModule.DataKind.Bool:
                    return value.Value<bool>();
                case TlcModule.DataKind.String:
                    return value.Value<string>();
                case TlcModule.DataKind.Char:
                    return value.Value<char>();
                case TlcModule.DataKind.Enum:
                    if (!Enum.IsDefined(type, value.Value<string>()))
                        throw ectx.Except($"Requested value '{value.Value<string>()}' is not a member of the Enum type '{type.Name}'");
                    return Enum.Parse(type, value.Value<string>());
                case TlcModule.DataKind.Float:
                    if (type == typeof(double))
                        return value.Value<double>();
                    else if (type == typeof(float))
                        return value.Value<float>();
                    else
                    {
                        ectx.Assert(false);
                        throw ectx.ExceptNotSupp();
                    }
                case TlcModule.DataKind.Array:
                    var ja = value as JArray;
                    ectx.Check(ja != null, "Expected array value");
                    Func<IExceptionContext, JArray, Attributes, ModuleCatalog, object> makeArray = MakeArray<int>;
                    return Utils.MarshalInvoke(makeArray, type.GetElementType(), ectx, ja, attributes, catalog);
                case TlcModule.DataKind.Int:
                    if (type == typeof(long))
                        return value.Value<long>();
                    if (type == typeof(int))
                        return value.Value<int>();
                    ectx.Assert(false);
                    throw ectx.ExceptNotSupp();
                case TlcModule.DataKind.UInt:
                    if (type == typeof(ulong))
                        return value.Value<ulong>();
                    if (type == typeof(uint))
                        return value.Value<uint>();
                    ectx.Assert(false);
                    throw ectx.ExceptNotSupp();
                case TlcModule.DataKind.Dictionary:
                    ectx.Check(value is JObject, "Expected object value");
                    Func<IExceptionContext, JObject, Attributes, ModuleCatalog, object> makeDict = MakeDictionary<int>;
                    return Utils.MarshalInvoke(makeDict, type.GetGenericArguments()[1], ectx, (JObject)value, attributes, catalog);
                case TlcModule.DataKind.Component:
                    var jo = value as JObject;
                    ectx.Check(jo != null, "Expected object value");
                    // REVIEW: consider accepting strings alone.
                    var jName = jo[FieldNames.Name];
                    ectx.Check(jName != null, "Field '" + FieldNames.Name + "' is required for component.");
                    ectx.Check(jName is JValue, "Expected '" + FieldNames.Name + "' field to be a string.");
                    var name = jName.Value<string>();
                    ectx.Check(jo[FieldNames.Settings] == null || jo[FieldNames.Settings] is JObject,
                        "Expected '" + FieldNames.Settings + "' field to be an object");
                    return GetComponentJson(ectx, type, name, jo[FieldNames.Settings] as JObject, catalog);
                default:
                    var settings = value as JObject;
                    ectx.Check(settings != null, "Expected object value");
                    var inputBuilder = new InputBuilder(ectx, type, catalog);

                    if (inputBuilder._fields.Length == 0)
                        throw ectx.Except($"Unsupported input type: {dt}");

                    if (settings != null)
                    {
                        foreach (var pair in settings)
                        {
                            if (!inputBuilder.TrySetValueJson(pair.Key, pair.Value))
                                throw ectx.Except($"Unexpected value for component '{type}', field '{pair.Key}': '{pair.Value}'");
                        }
                    }

                    var missing = inputBuilder.GetMissingValues().ToArray();
                    if (missing.Length > 0)
                        throw ectx.Except($"The following required inputs were not provided for component '{type}': {string.Join(", ", missing)}");
                    return inputBuilder.GetInstance();
                }
            }
            catch (FormatException ex)
            {
                if (ex.IsMarked())
                    throw;
                throw ectx.Except(ex, $"Failed to parse JSON value '{value}' as {type}");
            }
        }

        /// <summary>
        /// Ensures that the given value can be assigned to an entry point field with 
        /// type <paramref name="type"/>. This method will wrap the value in the option
        /// type if needed and throw an exception if the value isn't assignable.
        /// </summary>
        /// <param name="ectx">The exception context.</param>
        /// <param name="type">Type type of the field this value is to be assigned to.</param>
        /// <param name="value">The value, typically originates from either ParseJsonValue, or is an external, user-provided object.</param>
        private static object GetFieldAssignableValue(IExceptionContext ectx, Type type, object value)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(type);
            ectx.AssertValueOrNull(value);
            // If 'type' is optional, make 'value' into an optional (this is the case of optional input).
            value = MakeOptionalIfNeeded(ectx, value, type);
            if (value != null && !type.IsInstanceOfType(value))
                throw ectx.Except($"Unexpected value type: {value.GetType()}");
            return value;
        }

        private static IComponentFactory GetComponentJson(IExceptionContext ectx, Type signatureType, string name, JObject settings, ModuleCatalog catalog)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(signatureType);
            ectx.AssertNonEmpty(name);
            ectx.AssertValueOrNull(settings);
            ectx.AssertValue(catalog);

            if (!catalog.TryGetComponentKind(signatureType, out string kind))
                throw ectx.Except($"Component type '{signatureType}' is not a valid signature type.");

            if (!catalog.TryFindComponent(kind, name, out ModuleCatalog.ComponentInfo component))
            {
                var available = catalog.GetAllComponents(kind).Select(x => $"'{x.Name}'");
                throw ectx.Except($"Component '{name}' of kind '{kind}' is not found. Available components are: {string.Join(", ", available)}");
            }

            var inputBuilder = new InputBuilder(ectx, component.ArgumentType, catalog);
            if (settings != null)
            {
                foreach (var pair in settings)
                {
                    if (!inputBuilder.TrySetValueJson(pair.Key, pair.Value))
                        throw ectx.Except($"Unexpected value for component '{name}', field '{pair.Key}': '{pair.Value}'");
                }
            }

            var missing = inputBuilder.GetMissingValues().ToArray();
            if (missing.Length > 0)
                throw ectx.Except($"The following required inputs were not provided for component '{name}': {string.Join(", ", missing)}");
            return inputBuilder.GetInstance() as IComponentFactory;
        }

        private static object MakeArray<T>(IExceptionContext ectx, JArray jArray, Attributes attributes, ModuleCatalog catalog)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(jArray);

            T[] array = new T[jArray.Count];
            for (int i = 0; i < array.Length; i++)
                array[i] = (T)GetFieldAssignableValue(ectx, typeof(T), ParseJsonValue(ectx, typeof(T), attributes, jArray[i], catalog));
            return array;
        }

        private static object MakeDictionary<T>(IExceptionContext ectx, JObject jDict, Attributes attributes, ModuleCatalog catalog)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertValue(jDict);

            var dict = new Dictionary<string, T>();
            foreach (var pair in jDict)
                dict[pair.Key] = (T)GetFieldAssignableValue(ectx, typeof(T), ParseJsonValue(ectx, typeof(T), attributes, pair.Value, catalog));
            return dict;
        }

        private static object MakeOptional<T>(object value)
        {
            return (Optional<T>)(T)value;
        }

        private static object MakeNullable<T>(object value)
            where T : struct
        {
            return (T?)value;
        }

        /// <summary>
        /// If outerType is an Optional{T}, the innerValue is wrapped in a constructed, explicit
        /// Optional instance, otherwise the value is directly returned.
        /// </summary>
        private static object MakeOptionalIfNeeded(IExceptionContext ectx, object innerValue, Type outerType)
        {
            Contracts.AssertValue(ectx);
            // You can make an Optional null value!
            ectx.AssertValueOrNull(innerValue);
            ectx.AssertValue(outerType);
            if (!outerType.IsGenericType)
                return innerValue;

            var genericType = outerType.GetGenericTypeDefinition();
            if (genericType != typeof(Optional<>) &&
                genericType != typeof(Nullable<>))
            {
                return innerValue;
            }

            bool isOptional = outerType.GetGenericTypeDefinition() == typeof(Optional<>);
            Func<object, object> creator;
            if (isOptional)
                creator = MakeOptional<int>;
            else
            {
                ectx.Assert(genericType == typeof(Nullable<>));
                creator = MakeNullable<int>;
            }

            return Utils.MarshalInvoke(creator, outerType.GetGenericArguments()[0], innerValue);
        }

        /// <summary>
        /// Returns the created instance.
        /// </summary>
        public object GetInstance()
        {
            return _instance;
        }
    }

    /// <summary>
    /// This class wraps around the output object type, does not create an instance, and provides utility methods for field type checking
    /// and extracting values.
    /// </summary>
    public sealed class OutputHelper
    {
        private readonly IExceptionContext _ectx;
        private readonly Type _type;

        private readonly FieldInfo[] _fields;
        private readonly TlcModule.OutputAttribute[] _attrs;

        public OutputHelper(IExceptionContext ectx, Type outputType)
        {
            Contracts.CheckValue(ectx, nameof(ectx));
            _ectx = ectx;
            _ectx.CheckValue(outputType, nameof(outputType));

            if (outputType.IsGenericType && outputType.GetGenericTypeDefinition() == typeof(CommonOutputs.MacroOutput<>))
                outputType = outputType.GetGenericArguments()[0];
            _type = outputType;

            var fields = new List<FieldInfo>();
            var attrs = new List<TlcModule.OutputAttribute>();
            foreach (var fieldInfo in _type.GetFields())
            {
                var attr = fieldInfo.GetCustomAttributes(typeof(TlcModule.OutputAttribute), false).FirstOrDefault()
                    as TlcModule.OutputAttribute;
                if (attr == null)
                    continue;
                fields.Add(fieldInfo);
                attrs.Add(attr);
            }
            _ectx.Assert(fields.Count == attrs.Count);

            _fields = fields.ToArray();
            _attrs = attrs.ToArray();
        }

        private FieldInfo GetField(string name)
        {
            _ectx.AssertNonEmpty(name);
            for (int i = 0; i < _attrs.Length; i++)
            {
                if (name == (_attrs[i].Name ?? _fields[i].Name))
                    return _fields[i];
            }
            return null;
        }

        public Type GetFieldType(string name)
        {
            _ectx.CheckNonEmpty(name, nameof(name));

            var fi = GetField(name);
            var type = fi?.FieldType;
            if (type != null && type.IsGenericType && (type.GetGenericTypeDefinition() == typeof(Var<>)))
                type = type.GetGenericArguments()[0];
            return type;
        }

        /// <summary>
        /// Extract all values of a specified output object.
        /// </summary>
        public IEnumerable<KeyValuePair<string, object>> ExtractValues(object output)
        {
            _ectx.CheckValue(output, nameof(output));
            _ectx.Check(output.GetType() == _type);

            for (int i = 0; i < _fields.Length; i++)
            {
                var fieldInfo = _fields[i];
                var attr = _attrs[i];
                yield return new KeyValuePair<string, object>(attr.Name ?? fieldInfo.Name, fieldInfo.GetValue(output));
            }
        }

        public JObject GetJsonObject(Dictionary<string, string> outputMap)
        {
            _ectx.CheckValue(outputMap, nameof(outputMap));
            var result = new JObject();
            foreach (var fieldInfo in _fields)
            {
                if (outputMap.TryGetValue(fieldInfo.Name, out string varname))
                    result.Add(fieldInfo.Name, new JValue($"${varname}"));
            }

            return result;
        }
    }

    /// <summary>
    /// These are the common field names used in the JSON objects for defining the manifest.
    /// </summary>
    public static class FieldNames
    {
        public const string Nodes = "Nodes";
        public const string Kind = "Kind";
        public const string Components = "Components";
        public const string ComponentKind = "ComponentKind";
        public const string Type = "Type";
        public const string ItemType = "ItemType";
        public const string Fields = "Fields";
        public const string Values = "Values";

        public const string Name = "Name";
        public const string Aliases = "Aliases";
        public const string FriendlyName = "FriendlyName";
        public const string ShortName = "ShortName";
        public const string Desc = "Desc";
        public const string Required = "Required";
        public const string Default = "Default";

        // Fields for scheduling.
        public const string Checkpoint = "Checkpoint";
        public const string StageId = "StageId";
        public const string Cost = "Cost";

        public const string Settings = "Settings";
        public const string Inputs = "Inputs";
        public const string Outputs = "Outputs";
        public const string InputKind = "InputKind";
        public const string OutputKind = "OutputKind";
        public const string SortOrder = "SortOrder";
        public const string IsNullable = "IsNullable";

        // Top level field names.
        public const string TopEntryPoints = "EntryPoints";
        public const string TopComponents = "Components";
        public const string TopEntryPointKinds = "EntryPointKinds";

        /// <summary>
        /// Range specific field names.
        /// </summary>
        public static class Range
        {
            public const string Type = "Range";

            public const string Sup = "Sup";
            public const string Inf = "Inf";
            public const string Max = "Max";
            public const string Min = "Min";
        }

        /// <summary>
        /// Obsolete Attribute specific field names.
        /// </summary>
        public static class Deprecated
        {
            public new static string ToString() => "Deprecated";
            public const string Message = "Message";
        }

        /// <summary>
        /// SweepableLongParam specific field names.
        /// </summary>
        public static class SweepableLongParam
        {
            public new static string ToString() => "SweepRange";
            public const string RangeType = "RangeType";
            public const string Max = "Max";
            public const string Min = "Min";
            public const string StepSize = "StepSize";
            public const string NumSteps = "NumSteps";
            public const string IsLogScale = "IsLogScale";
        }

        /// <summary>
        /// SweepableFloatParam specific field names.
        /// </summary>
        public static class SweepableFloatParam
        {
            public new static string ToString() => "SweepRange";
            public const string RangeType = "RangeType";
            public const string Max = "Max";
            public const string Min = "Min";
            public const string StepSize = "StepSize";
            public const string NumSteps = "NumSteps";
            public const string IsLogScale = "IsLogScale";
        }

        /// <summary>
        /// SweepableDiscreteParam specific field names.
        /// </summary>
        public static class SweepableDiscreteParam
        {
            public new static string ToString() => "SweepRange";
            public const string RangeType = "RangeType";
            public const string Options = "Values";
        }

        public static class PipelineSweeperSupportedMetrics
        {
            public new static string ToString() => "SupportedMetric";
            public const string Auc = BinaryClassifierEvaluator.Auc;
            public const string AccuracyMicro = Data.MultiClassClassifierEvaluator.AccuracyMicro;
            public const string AccuracyMacro = MultiClassClassifierEvaluator.AccuracyMacro;
            public const string F1 = BinaryClassifierEvaluator.F1;
            public const string AuPrc = BinaryClassifierEvaluator.AuPrc;
            public const string TopKAccuracy = MultiClassClassifierEvaluator.TopKAccuracy;
            public const string L1 = RegressionLossEvaluatorBase<MultiOutputRegressionEvaluator.Aggregator>.L1;
            public const string L2 = RegressionLossEvaluatorBase<MultiOutputRegressionEvaluator.Aggregator>.L2;
            public const string Rms = RegressionLossEvaluatorBase<MultiOutputRegressionEvaluator.Aggregator>.Rms;
            public const string LossFn = RegressionLossEvaluatorBase<MultiOutputRegressionEvaluator.Aggregator>.Loss;
            public const string RSquared = RegressionLossEvaluatorBase<MultiOutputRegressionEvaluator.Aggregator>.RSquared;
            public const string LogLoss = BinaryClassifierEvaluator.LogLoss;
            public const string LogLossReduction = BinaryClassifierEvaluator.LogLossReduction;
            public const string Ndcg = RankerEvaluator.Ndcg;
            public const string Dcg = RankerEvaluator.Dcg;
            public const string PositivePrecision = BinaryClassifierEvaluator.PosPrecName;
            public const string PositiveRecall = BinaryClassifierEvaluator.PosRecallName;
            public const string NegativePrecision = BinaryClassifierEvaluator.NegPrecName;
            public const string NegativeRecall = BinaryClassifierEvaluator.NegRecallName;
            public const string DrAtK = AnomalyDetectionEvaluator.OverallMetrics.DrAtK;
            public const string DrAtPFpr = AnomalyDetectionEvaluator.OverallMetrics.DrAtPFpr;
            public const string DrAtNumPos = AnomalyDetectionEvaluator.OverallMetrics.DrAtNumPos;
            public const string NumAnomalies = AnomalyDetectionEvaluator.OverallMetrics.NumAnomalies;
            public const string ThreshAtK = AnomalyDetectionEvaluator.OverallMetrics.ThreshAtK;
            public const string ThreshAtP = AnomalyDetectionEvaluator.OverallMetrics.ThreshAtP;
            public const string ThreshAtNumPos = AnomalyDetectionEvaluator.OverallMetrics.ThreshAtNumPos;
            public const string Nmi = ClusteringEvaluator.Nmi;
            public const string AvgMinScore = ClusteringEvaluator.AvgMinScore;
            public const string Dbi = ClusteringEvaluator.Dbi;
        }
    }
}
