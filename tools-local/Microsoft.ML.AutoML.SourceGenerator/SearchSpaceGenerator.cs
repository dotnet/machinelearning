// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.CodeAnalysis;
using Microsoft.ML.AutoML.SourceGenerator.Template;

namespace Microsoft.ML.AutoML.SourceGenerator
{
    [Generator]
    public class SearchSpaceGenerator : ISourceGenerator
    {
        public void Execute(GeneratorExecutionContext context)
        {
            if (context.AdditionalFiles.Where(f => f.Path.Contains("code_gen_flag.json")).First() is AdditionalText text)
            {
                var json = text.GetText().ToString();
                var flags = JsonSerializer.Deserialize<Dictionary<string, bool>>(json);
                if (flags.TryGetValue(nameof(SearchSpaceGenerator), out var res) && res == false)
                {
                    return;
                }
            }

            var searchSpacesJson = context.AdditionalFiles.Where(f => f.Path.Contains("search_space.json"))
                                                          .Select(f => f.GetText().ToString())
                                                          .ToArray();
            var searchSpacesJNodes = searchSpacesJson.Select(x => JsonNode.Parse(x));

            foreach (var jNode in searchSpacesJNodes)
            {
                var className = Utils.ToTitleCase(jNode["name"].GetValue<string>());
                var searchSpaceJArray = jNode["search_space"].AsArray();
                var options = searchSpaceJArray.Select(t =>
                {
                    var optionName = Utils.ToTitleCase(t["name"].GetValue<string>());
                    string optionTypeName = t["type"].GetValue<string>() switch
                    {
                        "integer" => "int",
                        "float" => "float",
                        "double" => "double",
                        "string" => "string",
                        "boolean" => "bool",
                        "strings" => "string[]",
                        "resizingKind" => "ResizingKind",
                        "anchor" => "Anchor",
                        "colorBits" => "ColorBits",
                        "colorsOrder" => "ColorsOrder",
                        _ => throw new ArgumentException("unknown type"),
                    };

                    var defaultToken = t.AsObject().ContainsKey("default") ? t["default"] : null;
                    string optionDefaultValue = (defaultToken, optionTypeName) switch
                    {
                        (null, _) => string.Empty,
                        (_, "string") => $"\"{defaultToken.GetValue<string>()}\"",
                        (_, "int") => $"{defaultToken.GetValue<int>().ToString(CultureInfo.InvariantCulture)}",
                        (_, "double") => $"{defaultToken.GetValue<double>().ToString(CultureInfo.InvariantCulture)}",
                        (_, "float") => $"{defaultToken.GetValue<float>().ToString(CultureInfo.InvariantCulture)}F",
                        (_, "bool") => defaultToken.GetValue<bool>() ? "true" : "false",
                        (_, "Anchor") => defaultToken.GetValue<string>(),
                        (_, "ResizingKind") => defaultToken.GetValue<string>(),
                        (_, "ColorBits") => defaultToken.GetValue<string>(),
                        (_, "ColorsOrder") => defaultToken.GetValue<string>(),
                        (_, _) => throw new ArgumentException("unknown"),
                    };

                    var searchSpaceNode = t.AsObject().ContainsKey("search_space") ? t["search_space"] : null;
                    string optionAttribution = null;
                    if (searchSpaceNode is null)
                    {
                        // default option
                        optionAttribution = string.Empty;
                    }
                    else
                    {
                        var searchSpaceObject = searchSpaceNode.AsObject();
                        if (searchSpaceObject.ContainsKey("min"))
                        {
                            // range option
                            var minToken = searchSpaceNode["min"];
                            var minValue = searchSpaceNode["min"].GetValue<double>();
                            var maxValue = searchSpaceNode["max"].GetValue<double>();
                            var logBase = searchSpaceObject.ContainsKey("log_base") is false ? "false" : searchSpaceNode["log_base"].GetValue<bool>() ? "true" : "false";
                            optionAttribution = (optionTypeName, minValue, maxValue, logBase, optionDefaultValue) switch
                            {
                                ("int", _, _, _, null) => $"Range((int){Convert.ToInt32(minValue).ToString(CultureInfo.InvariantCulture)}, (int){Convert.ToInt32(maxValue).ToString(CultureInfo.InvariantCulture)}, logBase: {logBase.ToString(CultureInfo.InvariantCulture)})",
                                ("float", _, _, _, null) => $"Range((float){Convert.ToSingle(minValue).ToString(CultureInfo.InvariantCulture)}, (float){Convert.ToSingle(maxValue).ToString(CultureInfo.InvariantCulture)}, logBase: {logBase.ToString(CultureInfo.InvariantCulture)})",
                                ("double", _, _, _, null) => $"Range((double){minValue.ToString(CultureInfo.InvariantCulture)}, (double){maxValue.ToString(CultureInfo.InvariantCulture)}, logBase: {logBase.ToString(CultureInfo.InvariantCulture)})",
                                ("int", _, _, _, _) => $"Range((int){Convert.ToInt32(minValue).ToString(CultureInfo.InvariantCulture)}, (int){Convert.ToInt32(maxValue).ToString(CultureInfo.InvariantCulture)}, init: (int){optionDefaultValue.ToString(CultureInfo.InvariantCulture)}, logBase: {logBase.ToString(CultureInfo.InvariantCulture)})",
                                ("float", _, _, _, _) => $"Range((float){Convert.ToSingle(minValue).ToString(CultureInfo.InvariantCulture)}, (float){Convert.ToSingle(maxValue).ToString(CultureInfo.InvariantCulture)}, init: (float){optionDefaultValue.ToString(CultureInfo.InvariantCulture)}, logBase: {logBase.ToString(CultureInfo.InvariantCulture)})",
                                ("double", _, _, _, _) => $"Range((double){minValue.ToString(CultureInfo.InvariantCulture)}, (double){maxValue.ToString(CultureInfo.InvariantCulture)}, init: (double){optionDefaultValue.ToString(CultureInfo.InvariantCulture)}, logBase: {logBase.ToString(CultureInfo.InvariantCulture)})",
                                _ => throw new NotImplementedException(),
                            };
                            optionAttribution = $"[{optionAttribution}]";
                        }
                        else
                        {
                            // choice option
                            var values = searchSpaceNode["value"].GetValue<string[]>();
                            var valuesParam = optionTypeName switch
                            {
                                "int" => $"new object[]{{ {string.Join(",", values)} }}",
                                "boolean" => $"new object[]{{ {string.Join(",", values)} }}",
                                "string" => $"new object[]{{ {string.Join(",", values.Select(x => $"\"{x}\""))} }}",
                                _ => throw new NotImplementedException("only support int|boolean|string"),
                            };

                            optionAttribution = optionDefaultValue == null ? $"[Choice({valuesParam})]" : $"[Choice({valuesParam}, {optionDefaultValue})]";
                        }
                    }

                    return (optionTypeName, optionName, optionAttribution, optionDefaultValue);
                });

                var code = new SearchSpace()
                {
                    NameSpace = Constant.CodeGeneratorNameSpace,
                    ClassName = className,
                    Properties = options,
                }.TransformText();

                context.AddSource($"{className}.cs", code);
            }
        }

        public void Initialize(GeneratorInitializationContext context)
        {
        }
    }
}
