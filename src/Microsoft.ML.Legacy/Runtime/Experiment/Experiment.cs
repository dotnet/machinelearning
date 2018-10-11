// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// This class represents an entry point graph.
    /// The nodes in the graph represent entry point calls and
    /// the edges of the graph are variables that help connect the nodes.
    /// </summary>
    [JsonConverter(typeof(ExperimentSerializer))]
    public sealed partial class Experiment
    {
        private sealed class SerializationHelper
        {
            public string Name { get; set; }
            public object Inputs { get; set; }
            public object Outputs { get; set; }
        }

        private readonly Runtime.IHostEnvironment _env;
        private readonly ComponentCatalog _catalog;
        private readonly List<string> _jsonNodes;
        private readonly JsonSerializer _serializer;
        private readonly SerializationHelper _helper;
        private EntryPointGraph _graph;
        public Experiment(Runtime.IHostEnvironment env)
        {
            _env = env;
            AssemblyRegistration.RegisterAssemblies(_env);

            _catalog = _env.ComponentCatalog;
            _jsonNodes = new List<string>();
            _serializer = new JsonSerializer();
            _serializer.Converters.Add(new StringEnumConverter());
            _helper = new SerializationHelper();
        }

        /// <summary>
        /// Parses the nodes to determine the validity of the graph and
        /// to determine the inputs and outputs of the graph.
        /// </summary>
        public void Compile()
        {
            _env.Check(_graph == null, "Multiple calls to " + nameof(Compile) + "() detected.");
            var nodes = GetNodes();
            _graph = new EntryPointGraph(_env, nodes);
        }

        public JArray GetNodes()
        {
            JObject json;
            try
            {
                json = JObject.Parse($"{{'nodes': [{string.Join(",", _jsonNodes)}]}}");
            }
            catch (JsonReaderException ex)
            {
                throw _env.Except(ex, "Failed to parse experiment graph: {0}", ex.Message);
            }

            return json["nodes"] as JArray;
        }

        public void SetInput<TInput>(string varName, TInput input)
            where TInput : class
        {
            _env.CheckNonEmpty(varName, nameof(varName));
            _env.CheckValue(input, nameof(input));

            EntryPointVariable entryPointVariable = _graph.GetVariableOrNull(varName);

            if (entryPointVariable == null)
                throw _env.Except("Port '{0}' not found", varName);
            if (entryPointVariable.HasOutputs)
                throw _env.Except("Port '{0}' is not an input", varName);
            if (entryPointVariable.Value != null)
                throw _env.Except("Port '{0}' is already set", varName);
            if (!entryPointVariable.Type.IsAssignableFrom(typeof(TInput)))
                throw _env.Except("Port '{0}' is of incorrect type", varName);

            entryPointVariable.SetValue(input);
        }

        public void SetInput<TInput>(Var<TInput> variable, TInput input)
            where TInput : class
        {
            _env.CheckValue(variable, nameof(variable));
            var varName = variable.VarName;
            SetInput(varName, input);
        }

        public void SetInput<TInput, TInput2>(ArrayVar<TInput> variable, TInput2[] input)
            where TInput : class
        {
            _env.CheckValue(variable, nameof(variable));
            var varName = variable.VarName;
            _env.CheckNonEmpty(varName, nameof(variable.VarName));
            _env.CheckValue(input, nameof(input));
            if (!typeof(TInput).IsAssignableFrom(typeof(TInput2)))
                throw _env.ExceptUserArg(nameof(input), $"Type {typeof(TInput2)} not castable to type {typeof(TInput)}");

            EntryPointVariable entryPointVariable = _graph.GetVariableOrNull(varName);

            if (entryPointVariable == null)
                throw _env.Except("Port '{0}' not found", varName);
            if (entryPointVariable.HasOutputs)
                throw _env.Except("Port '{0}' is not an input", varName);
            if (entryPointVariable.Value != null)
                throw _env.Except("Port '{0}' is already set", varName);
            if (!entryPointVariable.Type.IsAssignableFrom(typeof(TInput[])))
                throw _env.Except("Port '{0}' is of incorrect type", varName);

            entryPointVariable.SetValue(input);
        }

        public void Run()
        {
            var graphRunner = new GraphRunner(_env, _graph);
            graphRunner.RunAll();
        }

        public TOutput GetOutput<TOutput>(Var<TOutput> var)
            where TOutput : class
        {
            _env.CheckValue(var, nameof(var));
            _env.CheckNonEmpty(var.VarName, nameof(var.VarName));
            var varBinding = VariableBinding.Create(_env, $"${var.VarName}");
            EntryPointVariable variable = _graph.GetVariableOrNull(varBinding.VariableName);

            if (variable == null)
                throw _env.Except("Port '{0}' not found", var.VarName);
            var value = varBinding.GetVariableValueOrNull(variable);
            if (value == null)
                return null;

            var result = value as TOutput;
            if (result == null)
                throw _env.Except("Incorrect type for output '{0}'", var.VarName);
            return result;
        }

        public void Reset()
        {
            _graph = null;
            _jsonNodes.Clear();
        }

        private string Serialize(string name, object input, object output)
        {
            _helper.Name = name;
            _helper.Inputs = input;
            _helper.Outputs = output;
            using (var sw = new StringWriter())
            {
                using (var jw = new JsonTextWriter(sw))
                {
                    jw.Formatting = Newtonsoft.Json.Formatting.Indented;
                    _serializer.Serialize(jw, _helper);
                }
                return sw.ToString();
            }
        }

        private string GetEntryPointName(Type inputType)
        {
            if (inputType.FullName != null)
            {
                int dotCounts = 0;
                string fullName = inputType.FullName;
                for (int i = fullName.Length - 1; i >= 0; i--)
                {
                    if (fullName[i] == '.')
                        dotCounts++;
                    if(dotCounts == 2)
                    {
                        return fullName.Substring(i + 1);
                    }
                }

                Contracts.Assert(dotCounts == 1);

                return fullName;
            }

            return null;
        }

        public EntryPointTransformOutput Add(CommonInputs.ITransformInput input)
        {
            var output = new EntryPointTransformOutput();
            Add(input, output);
            return output;
        }

        internal void Add(CommonInputs.ITransformInput input, CommonOutputs.ITransformOutput output)
        {
            _jsonNodes.Add(Serialize(GetEntryPointName(input.GetType()), input, output));
        }

        public EntryPointTrainerOutput Add(CommonInputs.ITrainerInput input)
        {
            var output = new EntryPointTrainerOutput();
            Add(input, output);
            return output;
        }

        internal void Add(CommonInputs.ITrainerInput input, CommonOutputs.ITrainerOutput output)
        {
            _jsonNodes.Add(Serialize(GetEntryPointName(input.GetType()), input, output));
        }

        public CommonOutputs.IEvaluatorOutput Add(CommonInputs.IEvaluatorInput input, CommonOutputs.IEvaluatorOutput output)
        {
            _jsonNodes.Add(Serialize(GetEntryPointName(input.GetType()), input, output));
            return output;
        }

        public string ToJsonString() => String.Join(",", _jsonNodes);
    }

    public sealed class ComponentSerializer : JsonConverter
    {
        private class Helper
        {
            public string Name { get; set; }
            public object Settings { get; set; }
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            Contracts.Assert(value is ComponentKind);
            var componentKind = (ComponentKind)value;
            var helper = new Helper();
            helper.Name = componentKind.ComponentName;
            helper.Settings = componentKind;
            serializer.ReferenceLoopHandling = ReferenceLoopHandling.Serialize;
            serializer.Serialize(writer, helper);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            throw Contracts.ExceptNotImpl("Parsing JSON for Component not needed for the C# API.");
        }

        public override bool CanConvert(Type objectType) => typeof(ComponentKind).IsAssignableFrom(objectType);

        public override bool CanRead => false;
    }

    public sealed class ExperimentSerializer : JsonConverter
    {
        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            Contracts.Assert(value is Experiment);
            var subGraph = (Experiment)value;
            var nodes = subGraph.GetNodes();
            nodes.WriteTo(writer);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            throw Contracts.ExceptNotImpl("Parsing JSON for Experiment not needed for the C# API.");
        }

        public override bool CanConvert(Type objectType) => typeof(Experiment).IsAssignableFrom(objectType);

        public override bool CanRead => false;
    }

    public abstract class ComponentKind
    {
        internal ComponentKind() { }

        [JsonIgnore]
        internal abstract string ComponentName { get; }
    }

    public static class ExperimentUtils
    {
        public static Experiment CreateExperiment(this IHostEnvironment env)
        {
            return new Experiment(env);
        }

        public static string GenerateOverallMetricVarName(Guid id) => $"Var_OM_{id:N}";
    }

    public sealed class EntryPointTransformOutput : CommonOutputs.ITransformOutput
    {
        /// <summary>
        /// Transformed dataset
        /// </summary>
        public Var<Runtime.Data.IDataView> OutputData { get; set; }

        /// <summary>
        /// Transform model
        /// </summary>
        public Var<ITransformModel> Model { get; set; }

        public EntryPointTransformOutput()
        {
            OutputData = new Var<Runtime.Data.IDataView>();
            Model = new Var<ITransformModel>();
        }
    }

    public sealed class EntryPointTrainerOutput : CommonOutputs.ITrainerOutput
    {
        /// <summary>
        /// The trained model
        /// </summary>
        public Var<IPredictorModel> PredictorModel { get; set; }

        public EntryPointTrainerOutput()
        {
            PredictorModel = new Var<IPredictorModel>();
        }
    }
}
