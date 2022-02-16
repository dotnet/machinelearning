// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.ML.AutoML.SourceGenerator.Template;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
                var flags = JsonConvert.DeserializeObject<Dictionary<string, bool>>(json);
                if (flags.TryGetValue(nameof(SearchSpaceGenerator), out var res) && res == false)
                {
                    return;
                }
            }

            var searchSpacesJson = context.AdditionalFiles.Where(f => f.Path.Contains("search_space.json"))
                                                          .Select(f => f.GetText().ToString())
                                                          .ToArray();
            var searchSpacesJObjects = searchSpacesJson.Select(x => JObject.Parse(x));

            //if (!Debugger.IsAttached)
            //    Debugger.Launch();

            foreach (var jObject in searchSpacesJObjects)
            {
                var className = Utils.ToTitleCase(jObject.Value<string>("name"));
                var searchSpaceJArray = jObject.Value<JArray>("search_space");
                var options = searchSpaceJArray.Select(t =>
                {
                    var optionName = Utils.ToTitleCase(t.Value<string>("name"));
                    string optionTypeName = t.Value<string>("type") switch
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

                    t.ToObject<JObject>().TryGetValue("default", out var defaultToken);
                    string optionDefaultValue = (defaultToken?.Type, optionTypeName) switch
                    {
                        (null, _) => string.Empty,
                        (_, "string") => $"\"{defaultToken.ToObject<string>()}\"",
                        (_, "int") => $"{defaultToken.ToObject<int>()}",
                        (_, "double") => $"{defaultToken.ToObject<double>()}",
                        (_, "float") => $"{defaultToken.ToObject<float>()}F",
                        (_, "bool") => defaultToken.ToObject<bool>() ? "true" : "false",
                        (_, "Anchor") => defaultToken.ToObject<string>(),
                        (_, "ResizingKind") => defaultToken.ToObject<string>(),
                        (_, "ColorBits") => defaultToken.ToObject<string>(),
                        (_, "ColorsOrder") => defaultToken.ToObject<string>(),
                        (_, _) => throw new ArgumentException("unknown"),
                    };

                    t.ToObject<JObject>().TryGetValue("search_space", out var searchSpaceToken);
                    string optionAttribution = null;
                    if (searchSpaceToken?.ToObject<object>() is null)
                    {
                        // default option
                        optionAttribution = string.Empty;
                    }
                    else
                    {
                        var searchSpaceObject = searchSpaceToken.ToObject<JObject>();
                        if (searchSpaceObject.TryGetValue("min", out var minToken))
                        {
                            // range option
                            var minValue = searchSpaceToken.Value<double>("min");
                            var maxValue = searchSpaceToken.Value<double>("max");
                            searchSpaceObject.TryGetValue("log_base", out var logBaseToken);
                            var logBase = logBaseToken is null ? "false" : logBaseToken.ToObject<bool>() ? "true" : "false";
                            optionAttribution = (optionTypeName, minValue, maxValue, logBase, optionDefaultValue) switch
                            {
                                ("int", _, _, _, null) => $"Range((int){Convert.ToInt32(minValue)}, (int){Convert.ToInt32(maxValue)}, logBase: {logBase})",
                                ("float", _, _, _, null) => $"Range((float){Convert.ToSingle(minValue)}, (float){Convert.ToSingle(maxValue)}, logBase: {logBase})",
                                ("double", _, _, _, null) => $"Range((double){minValue}, (double){maxValue}, logBase: {logBase})",
                                ("int", _, _, _, _) => $"Range((int){Convert.ToInt32(minValue)}, (int){Convert.ToInt32(maxValue)}, init: (int){optionDefaultValue}, logBase: {logBase})",
                                ("float", _, _, _, _) => $"Range((float){Convert.ToSingle(minValue)}, (float){Convert.ToSingle(maxValue)}, init: (float){optionDefaultValue}, logBase: {logBase})",
                                ("double", _, _, _, _) => $"Range((double){minValue}, (double){maxValue}, init: (double){optionDefaultValue}, logBase: {logBase})",
                                _ => throw new NotImplementedException(),
                            };
                            optionAttribution = $"[{optionAttribution}]";
                        }
                        else
                        {
                            // choice option
                            var values = searchSpaceToken.Value<string[]>("value");
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
