// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Tools;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.EntryPoints.JsonUtils
{
    /// <summary>
    /// Utilities to generate JSON manifests for entry points and other components.
    /// </summary>
    public static class JsonManifestUtils
    {
        /// <summary>
        /// Builds a JSON representation of all entry points and components of the <paramref name="catalog"/>.
        /// </summary>
        /// <param name="ectx">The exception context to use</param>
        /// <param name="catalog">The module catalog</param>
        public static JObject BuildAllManifests(IExceptionContext ectx, ModuleCatalog catalog)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(catalog, nameof(catalog));

            var jEntryPoints = new JArray();
            var entryPointInfos = catalog.AllEntryPoints().ToArray();
            foreach (var entryPointInfo in entryPointInfos.OrderBy(x => x.Name))
                jEntryPoints.Add(BuildEntryPointManifest(ectx, entryPointInfo, catalog));

            var jKinds = new JArray();
            foreach (var kind in catalog.GetAllComponentKinds())
            {
                var jKind = new JObject();
                jKind[FieldNames.Kind] = kind;
                var jComponents = new JArray();
                foreach (var component in catalog.GetAllComponents(kind))
                    jComponents.Add(BuildComponentManifest(ectx, component, catalog));

                jKind[FieldNames.Components] = jComponents;

                jKinds.Add(jKind);
            }

            var jepKinds = new JArray();
            var kinds = new List<Type>();
            foreach (var entryPointInfo in entryPointInfos)
            {
                if (entryPointInfo.InputKinds != null)
                    kinds.AddRange(entryPointInfo.InputKinds);
                if (entryPointInfo.OutputKinds != null)
                    kinds.AddRange(entryPointInfo.OutputKinds);
            }

            foreach (var epKind in kinds.Distinct().OrderBy(k => k.Name))
            {
                var jepKind = new JObject();
                jepKind[FieldNames.Kind] = epKind.Name;
                var jepKindFields = new JArray();
                var propertyInfos = epKind.GetProperties().AsEnumerable();
                propertyInfos = epKind.GetInterfaces().Aggregate(propertyInfos, (current, face) => current.Union(face.GetProperties()));
                foreach (var fieldInfo in propertyInfos)
                {
                    var jField = new JObject();
                    jField[FieldNames.Name] = fieldInfo.Name;
                    var type = CSharpGeneratorUtils.ExtractOptionalOrNullableType(fieldInfo.PropertyType);
                    // Dive inside Var.
                    if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Var<>))
                        type = type.GetGenericArguments()[0];
                    var typeEnum = TlcModule.GetDataType(type);
                    jField[FieldNames.Type] = typeEnum.ToString();
                    jepKindFields.Add(jField);
                }
                jepKind[FieldNames.Settings] = jepKindFields;
                jepKinds.Add(jepKind);
            }

            var jResult = new JObject();
            jResult[FieldNames.TopEntryPoints] = jEntryPoints;
            jResult[FieldNames.TopComponents] = jKinds;
            jResult[FieldNames.TopEntryPointKinds] = jepKinds;
            return jResult;
        }

        private static JObject BuildComponentManifest(IExceptionContext ectx, ModuleCatalog.ComponentInfo componentInfo, ModuleCatalog catalog)
        {
            Contracts.AssertValueOrNull(ectx);
            ectx.AssertValue(componentInfo);
            ectx.AssertValue(catalog);
            var result = new JObject();
            result[FieldNames.Name] = componentInfo.Name;
            result[FieldNames.Desc] = componentInfo.Description;
            result[FieldNames.FriendlyName] = componentInfo.FriendlyName;
            if (Utils.Size(componentInfo.Aliases) > 0)
                result[FieldNames.Aliases] = new JArray(componentInfo.Aliases);

            result[FieldNames.Settings] = BuildInputManifest(ectx, componentInfo.ArgumentType, catalog);
            return result;
        }

        public static JObject BuildEntryPointManifest(IExceptionContext ectx, ModuleCatalog.EntryPointInfo entryPointInfo, ModuleCatalog catalog)
        {
            Contracts.CheckValueOrNull(ectx);
            ectx.CheckValue(entryPointInfo, nameof(entryPointInfo));
            ectx.CheckValue(catalog, nameof(catalog));

            var result = new JObject();
            result[FieldNames.Name] = entryPointInfo.Name;
            result[FieldNames.Desc] = entryPointInfo.Description;
            result[FieldNames.FriendlyName] = entryPointInfo.FriendlyName;
            result[FieldNames.ShortName] = entryPointInfo.ShortName;

            // There supposed to be 2 parameters, env and input.
            result[FieldNames.Inputs] = BuildInputManifest(ectx, entryPointInfo.InputType, catalog);
            result[FieldNames.Outputs] = BuildOutputManifest(ectx, entryPointInfo.OutputType, catalog);

            if (entryPointInfo.InputKinds != null)
            {
                var jInputKinds = new JArray();
                foreach (var kind in entryPointInfo.InputKinds)
                    jInputKinds.Add(kind.Name);
                result[FieldNames.InputKind] = jInputKinds;
            }

            if (entryPointInfo.OutputKinds != null)
            {
                var jOutputKinds = new JArray();
                foreach (var kind in entryPointInfo.OutputKinds)
                    jOutputKinds.Add(kind.Name);
                result[FieldNames.OutputKind] = jOutputKinds;
            }
            return result;
        }

        private static JArray BuildInputManifest(IExceptionContext ectx, Type inputType, ModuleCatalog catalog)
        {
            Contracts.AssertValueOrNull(ectx);
            ectx.AssertValue(inputType);
            ectx.AssertValue(catalog);

            // Instantiate a value of the input, to pull defaults out of.
            var defaults = Activator.CreateInstance(inputType);

            var inputs = new List<KeyValuePair<Double, JObject>>();
            foreach (var fieldInfo in inputType.GetFields())
            {
                var inputAttr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault() as ArgumentAttribute;
                if (inputAttr == null || inputAttr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;
                var jo = new JObject();
                jo[FieldNames.Name] = inputAttr.Name ?? fieldInfo.Name;
                jo[FieldNames.Type] = BuildTypeToken(ectx, fieldInfo, fieldInfo.FieldType, catalog);
                jo[FieldNames.Desc] = inputAttr.HelpText;
                if (inputAttr.Aliases != null)
                    jo[FieldNames.Aliases] = new JArray(inputAttr.Aliases);

                jo[FieldNames.Required] = inputAttr.IsRequired;
                jo[FieldNames.SortOrder] = inputAttr.SortOrder;
                jo[FieldNames.IsNullable] = fieldInfo.FieldType.IsGenericType && (fieldInfo.FieldType.GetGenericTypeDefinition() == typeof(Nullable<>));

                var defaultValue = fieldInfo.GetValue(defaults);
                var dataType = TlcModule.GetDataType(fieldInfo.FieldType);
                if (!inputAttr.IsRequired || (dataType != TlcModule.DataKind.Unknown && defaultValue != null))
                    jo[FieldNames.Default] = BuildValueToken(ectx, defaultValue, fieldInfo.FieldType, catalog);

                if (fieldInfo.FieldType.IsGenericType &&
                    fieldInfo.FieldType.GetGenericTypeDefinition() == typeof(Optional<>))
                {
                    var val = fieldInfo.GetValue(defaults) as Optional;
                    if (val == null && !inputAttr.IsRequired)
                        throw ectx.Except("Field '{0}' is an Optional<> type but is null by default, instead of set to a constructed implicit default.", fieldInfo.Name);
                    if (val != null && val.IsExplicit)
                        throw ectx.Except("Field '{0}' is an Optional<> type with a non-implicit default value.", fieldInfo.Name);
                }

                var rangeAttr = fieldInfo.GetCustomAttributes(typeof(TlcModule.RangeAttribute), false).FirstOrDefault() as TlcModule.RangeAttribute;
                if (rangeAttr != null)
                {
                    if (!TlcModule.IsNumericKind(TlcModule.GetDataType(fieldInfo.FieldType)))
                        throw ectx.Except("Field '{0}' has a range but is of a non-numeric type.", fieldInfo.Name);

                    if (!rangeAttr.Type.Equals(fieldInfo.FieldType))
                        throw ectx.Except("Field '{0}' has a range attribute that uses a type which is not equal to the field's FieldType.", fieldInfo.Name);

                    var jRange = new JObject();
                    if (rangeAttr.Sup != null)
                        jRange[FieldNames.Range.Sup] = JToken.FromObject(rangeAttr.Sup);
                    if (rangeAttr.Inf != null)
                        jRange[FieldNames.Range.Inf] = JToken.FromObject(rangeAttr.Inf);
                    if (rangeAttr.Max != null)
                        jRange[FieldNames.Range.Max] = JToken.FromObject(rangeAttr.Max);
                    if (rangeAttr.Min != null)
                        jRange[FieldNames.Range.Min] = JToken.FromObject(rangeAttr.Min);
                    jo[FieldNames.Range.Type] = jRange;
                }

                // Handle deprecated/obsolete attributes, passing along the message to the manifest.
                if (fieldInfo.GetCustomAttributes(typeof(ObsoleteAttribute), false).FirstOrDefault() is ObsoleteAttribute obsAttr)
                {
                    var jParam = new JObject
                    {
                        [FieldNames.Deprecated.Message] = JToken.FromObject(obsAttr.Message),
                    };
                    jo[FieldNames.Deprecated.ToString()] = jParam;
                }

                if (fieldInfo.GetCustomAttributes(typeof(TlcModule.SweepableLongParamAttribute), false).FirstOrDefault() is TlcModule.SweepableLongParamAttribute slpAttr)
                {
                    var jParam = new JObject
                    {
                        [FieldNames.SweepableLongParam.RangeType] = JToken.FromObject("Long"),
                        [FieldNames.SweepableLongParam.Min] = JToken.FromObject(slpAttr.Min),
                        [FieldNames.SweepableLongParam.Max] = JToken.FromObject(slpAttr.Max)
                    };
                    if (slpAttr.StepSize != null)
                        jParam[FieldNames.SweepableLongParam.StepSize] = JToken.FromObject(slpAttr.StepSize);
                    if (slpAttr.NumSteps != null)
                        jParam[FieldNames.SweepableLongParam.NumSteps] = JToken.FromObject(slpAttr.NumSteps);
                    if (slpAttr.IsLogScale)
                        jParam[FieldNames.SweepableLongParam.IsLogScale] = JToken.FromObject(true);
                    jo[FieldNames.SweepableLongParam.ToString()] = jParam;
                }

                if (fieldInfo.GetCustomAttributes(typeof(TlcModule.SweepableFloatParamAttribute), false).FirstOrDefault() is TlcModule.SweepableFloatParamAttribute sfpAttr)
                {
                    var jParam = new JObject
                    {
                        [FieldNames.SweepableFloatParam.RangeType] = JToken.FromObject("Float"),
                        [FieldNames.SweepableFloatParam.Min] = JToken.FromObject(sfpAttr.Min),
                        [FieldNames.SweepableFloatParam.Max] = JToken.FromObject(sfpAttr.Max)
                    };
                    if (sfpAttr.StepSize != null)
                        jParam[FieldNames.SweepableFloatParam.StepSize] = JToken.FromObject(sfpAttr.StepSize);
                    if (sfpAttr.NumSteps != null)
                        jParam[FieldNames.SweepableFloatParam.NumSteps] = JToken.FromObject(sfpAttr.NumSteps);
                    if (sfpAttr.IsLogScale)
                        jParam[FieldNames.SweepableFloatParam.IsLogScale] = JToken.FromObject(true);
                    jo[FieldNames.SweepableFloatParam.ToString()] = jParam;
                }

                if (fieldInfo.GetCustomAttributes(typeof(TlcModule.SweepableDiscreteParamAttribute), false).FirstOrDefault() is TlcModule.SweepableDiscreteParamAttribute sdpAttr)
                {
                    var jParam = new JObject
                    {
                        [FieldNames.SweepableDiscreteParam.RangeType] = JToken.FromObject("Discrete"),
                        [FieldNames.SweepableDiscreteParam.Options] = JToken.FromObject(sdpAttr.Options)
                    };
                    jo[FieldNames.SweepableDiscreteParam.ToString()] = jParam;
                }

                inputs.Add(new KeyValuePair<Double, JObject>(inputAttr.SortOrder, jo));
            }
            return new JArray(inputs.OrderBy(x => x.Key).Select(x => x.Value).ToArray());
        }

        private static JArray BuildOutputManifest(IExceptionContext ectx, Type outputType, ModuleCatalog catalog)
        {
            Contracts.AssertValueOrNull(ectx);
            ectx.AssertValue(outputType);
            ectx.AssertValue(catalog);

            var outputs = new List<KeyValuePair<Double, JObject>>();

            if (outputType.IsGenericType && outputType.GetGenericTypeDefinition() == typeof(CommonOutputs.MacroOutput<>))
                outputType = outputType.GetGenericArguments()[0];

            foreach (var fieldInfo in outputType.GetFields())
            {
                var outputAttr = fieldInfo.GetCustomAttributes(typeof(TlcModule.OutputAttribute), false)
                    .FirstOrDefault() as TlcModule.OutputAttribute;
                if (outputAttr == null)
                    continue;

                var jo = new JObject();
                jo[FieldNames.Name] = outputAttr.Name ?? fieldInfo.Name;
                jo[FieldNames.Type] = BuildTypeToken(ectx, fieldInfo, fieldInfo.FieldType, catalog);
                jo[FieldNames.Desc] = outputAttr.Desc;

                outputs.Add(new KeyValuePair<Double, JObject>(outputAttr.SortOrder, jo));
            }
            return new JArray(outputs.OrderBy(x => x.Key).Select(x => x.Value).ToArray());
        }

        private static JToken BuildTypeToken(IExceptionContext ectx, FieldInfo fieldInfo, Type type, ModuleCatalog catalog)
        {
            Contracts.AssertValueOrNull(ectx);
            ectx.AssertValue(type);
            ectx.AssertValue(catalog);

            // REVIEW: Allows newly introduced types to not break the manifest bulding process.
            // Where possible, these types should be replaced by component kinds.
            if (type == typeof(CommonInputs.IEvaluatorInput) ||
                type == typeof(CommonOutputs.IEvaluatorOutput))
            {
                var jo = new JObject();
                var typeString = $"{type}".Replace("Microsoft.ML.Runtime.EntryPoints.", "");
                jo[FieldNames.Kind] = "EntryPoint";
                jo[FieldNames.ItemType] = typeString;
                return jo;
            }
            type = CSharpGeneratorUtils.ExtractOptionalOrNullableType(type);

            // Dive inside Var.
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Var<>))
                type = type.GetGenericArguments()[0];

            var typeEnum = TlcModule.GetDataType(type);
            switch (typeEnum)
            {
                case TlcModule.DataKind.Unknown:
                    var jo = new JObject();
                    if (type == typeof(JArray))
                    {
                        jo[FieldNames.Kind] = TlcModule.DataKind.Array.ToString();
                        jo[FieldNames.ItemType] = "Node";
                        return jo;
                    }
                    if (type == typeof(JObject))
                    {
                        return "Bindings";
                    }
                    var fields = BuildInputManifest(ectx, type, catalog);
                    if (fields.Count == 0)
                        throw ectx.Except("Unexpected parameter type: {0}", type);
                    jo[FieldNames.Kind] = "Struct";
                    jo[FieldNames.Fields] = fields;
                    return jo;
                case TlcModule.DataKind.Float:
                case TlcModule.DataKind.Int:
                case TlcModule.DataKind.UInt:
                case TlcModule.DataKind.Char:
                case TlcModule.DataKind.String:
                case TlcModule.DataKind.Bool:
                case TlcModule.DataKind.DataView:
                case TlcModule.DataKind.TransformModel:
                case TlcModule.DataKind.PredictorModel:
                case TlcModule.DataKind.FileHandle:
                    return typeEnum.ToString();
                case TlcModule.DataKind.Enum:
                    jo = new JObject();
                    jo[FieldNames.Kind] = typeEnum.ToString();
                    var values = Enum.GetNames(type).Where(n => type.GetField(n).GetCustomAttribute<HideEnumValueAttribute>() == null);
                    jo[FieldNames.Values] = new JArray(values);
                    return jo;
                case TlcModule.DataKind.Array:
                    jo = new JObject();
                    jo[FieldNames.Kind] = typeEnum.ToString();
                    jo[FieldNames.ItemType] = BuildTypeToken(ectx, fieldInfo, type.GetElementType(), catalog);
                    return jo;
                case TlcModule.DataKind.Dictionary:
                    jo = new JObject();
                    jo[FieldNames.Kind] = typeEnum.ToString();
                    jo[FieldNames.ItemType] = BuildTypeToken(ectx, fieldInfo, type.GetGenericArguments()[1], catalog);
                    return jo;
                case TlcModule.DataKind.Component:
                    string kind;
                    if (!catalog.TryGetComponentKind(type, out kind))
                        throw ectx.Except("Field '{0}' is a component of unknown kind", fieldInfo.Name);

                    jo = new JObject();
                    jo[FieldNames.Kind] = typeEnum.ToString();
                    jo[FieldNames.ComponentKind] = kind;
                    return jo;
                case TlcModule.DataKind.State:
                    jo = new JObject();
                    var typeString = $"{type}".Replace("Microsoft.ML.Runtime.Interfaces.", "");
                    jo[FieldNames.Kind] = "C# Object";
                    jo[FieldNames.ItemType] = typeString;
                    return jo;
                default:
                    ectx.Assert(false);
                    throw ectx.ExceptNotSupp();
            }
        }

        private static JToken BuildValueToken(IExceptionContext ectx, object value, Type valueType, ModuleCatalog catalog)
        {
            Contracts.AssertValueOrNull(ectx);
            ectx.AssertValueOrNull(value);
            ectx.AssertValue(valueType);
            ectx.AssertValue(catalog);

            if (value == null)
                return null;

            // Dive inside Nullable.
            if (valueType.IsGenericType && valueType.GetGenericTypeDefinition() == typeof(Nullable<>))
                valueType = valueType.GetGenericArguments()[0];

            // Dive inside Optional.
            if (valueType.IsGenericType && valueType.GetGenericTypeDefinition() == typeof(Optional<>))
            {
                valueType = valueType.GetGenericArguments()[0];
                value = ((Optional)value).GetValue();
            }

            var dataType = TlcModule.GetDataType(valueType);
            switch (dataType)
            {
                case TlcModule.DataKind.Bool:
                case TlcModule.DataKind.Int:
                case TlcModule.DataKind.UInt:
                case TlcModule.DataKind.Float:
                case TlcModule.DataKind.String:
                    return new JValue(value);
                case TlcModule.DataKind.Char:
                    return new JValue(value.ToString());
                case TlcModule.DataKind.Array:
                    var valArray = value as Array;
                    var ja = new JArray();
                    foreach (var item in valArray)
                        ja.Add(BuildValueToken(ectx, item, item.GetType(), catalog));
                    return ja;
                case TlcModule.DataKind.Enum:
                    return value.ToString();
                case TlcModule.DataKind.Dictionary:
                    // REVIEW: need to figure out how to represent these.
                    throw ectx.ExceptNotSupp("Dictionary and component default values are not supported");
                case TlcModule.DataKind.Component:
                    var factory = value as IComponentFactory;
                    ectx.AssertValue(factory);
                    return BuildComponentToken(ectx, factory, catalog);
                default:
                    throw ectx.ExceptNotSupp("Encountered a default value for unsupported type {0}", dataType);
            }
        }

        /// <summary>
        /// Build a token for component default value. This will look up the component in the catalog, and if it finds an entry, it will
        /// build a JSON structure that would be parsed into the default value.
        ///
        /// This is an inherently fragile setup in case when the factory is not trivial, but it will work well for 'property bag' factories
        /// that we are currently using.
        /// </summary>
        private static JToken BuildComponentToken(IExceptionContext ectx, IComponentFactory value, ModuleCatalog catalog)
        {
            Contracts.AssertValueOrNull(ectx);
            ectx.AssertValue(value);
            ectx.AssertValue(catalog);

            var type = value.GetType();
            ModuleCatalog.ComponentInfo componentInfo;
            if (!catalog.TryFindComponent(type, out componentInfo))
            {
                // The default component is not in the catalog. This is, technically, allowed, but it means that there's no JSON representation
                // for the default value. We will emit the one the won't parse back.
                return new JValue("(custom component)");
            }

            ectx.Assert(componentInfo.ArgumentType == type);

            // Try to invoke default ctor for the factory to obtain defaults.
            object defaults;
            try
            {
                defaults = Activator.CreateInstance(type);
            }
            catch (MissingMemberException ex)
            {
                // There was no default constructor found.
                // This should never happen, since ModuleCatalog would error out if there is no default ctor.
                ectx.Assert(false);
                throw ectx.Except(ex, "Couldn't find default constructor");
            }

            var jResult = new JObject();
            var jSettings = new JObject();
            jResult[FieldNames.Name] = componentInfo.Name;

            // Iterate over all fields of the factory object, and compare the values with the defaults.
            // If the value differs, insert it into the settings object.
            bool anyValue = false;
            foreach (var fieldInfo in type.GetFields())
            {
                var attr = fieldInfo.GetCustomAttributes(typeof(ArgumentAttribute), false).FirstOrDefault()
                    as ArgumentAttribute;
                if (attr == null || attr.Visibility == ArgumentAttribute.VisibilityType.CmdLineOnly)
                    continue;
                ectx.Assert(!fieldInfo.IsStatic && !fieldInfo.IsInitOnly && !fieldInfo.IsLiteral);

                bool needValue = false;
                object actualValue = fieldInfo.GetValue(value);
                if (attr.IsRequired)
                    needValue = true;
                else
                {
                    object defaultValue = fieldInfo.GetValue(defaults);
                    needValue = !Equals(actualValue, defaultValue);
                }
                if (!needValue)
                    continue;
                jSettings[attr.Name ?? fieldInfo.Name] = BuildValueToken(ectx, actualValue, fieldInfo.FieldType, catalog);
                anyValue = true;
            }

            if (anyValue)
                jResult[FieldNames.Settings] = jSettings;
            return jResult;
        }
    }
}
