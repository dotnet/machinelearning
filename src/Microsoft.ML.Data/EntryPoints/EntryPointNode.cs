// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints.JsonUtils;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.EntryPoints
{
    public class VarSerializer : JsonConverter
    {
        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var variable = value as IVarSerializationHelper;
            Contracts.AssertValue(variable);
            if (!variable.IsValue)
                serializer.Serialize(writer, $"${variable.VarName}");
            else
                serializer.Serialize(writer, variable.Values.Select(v => $"${v}"));
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            throw Contracts.ExceptNotImpl();
        }

        public override bool CanConvert(Type objectType)
        {
            return typeof(IVarSerializationHelper).IsAssignableFrom(objectType);
        }

        public override bool CanRead => false;
    }

    internal interface IVarSerializationHelper
    {
        string VarName { get; set; }
        bool IsValue { get; }
        string[] Values { get; }
    }

    /// <summary>
    /// Marker class for the arguments that can be used as variables
    /// in an entry point graph.
    /// </summary>
    [JsonConverter(typeof(VarSerializer))]
    public sealed class Var<T> : IVarSerializationHelper
    {
        public string VarName { get; set; }
        bool IVarSerializationHelper.IsValue { get; }
        string[] IVarSerializationHelper.Values { get; }

        public Var()
        {
            Contracts.Assert(CheckType(typeof(T)));
            VarName = $"Var_{Guid.NewGuid().ToString("N")}";
        }

        public static bool CheckType(Type type)
        {
            if (type.IsArray)
                return CheckType(type.GetElementType());
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Dictionary<,>)
                && type.GetGenericTypeArgumentsEx()[0] == typeof(string))
            {
                return CheckType(type.GetGenericTypeArgumentsEx()[1]);
            }

            return
                type == typeof(IDataView) ||
                type == typeof(IFileHandle) ||
                type == typeof(IPredictorModel) ||
                type == typeof(ITransformModel) ||
                type == typeof(CommonInputs.IEvaluatorInput) ||
                type == typeof(CommonOutputs.IEvaluatorOutput) ||
                type == typeof(IMlState);
        }
    }

    /// <summary>
    /// Marker class for the arguments that can be used as array output variables
    /// in an entry point graph.
    /// </summary>
    [JsonConverter(typeof(VarSerializer))]
    public sealed class ArrayVar<T> : IVarSerializationHelper
    {
        public string VarName { get; set; }
        private readonly bool _isValue;
        bool IVarSerializationHelper.IsValue => _isValue;
        private readonly string[] _values;
        string[] IVarSerializationHelper.Values => _values;

        public ArrayVar()
        {
            Contracts.Assert(Var<T>.CheckType(typeof(T)));
            VarName = $"Var_{Guid.NewGuid().ToString("N")}";
        }

        public ArrayVar(params Var<T>[] variables)
        {
            Contracts.Assert(Var<T>.CheckType(typeof(T)));
            _values = variables.Select(v => v.VarName).ToArray();
            _isValue = true;
        }

        public Var<T> this[int i]
        {
            get
            {
                var item = new Var<T>();
                item.VarName = $"{VarName}[{i}]";
                return item;
            }
        }
    }

    /// <summary>
    /// Marker class for the arguments that can be used as dictionary output variables
    /// in an entry point graph.
    /// </summary>
    [JsonConverter(typeof(VarSerializer))]
    public sealed class DictionaryVar<T> : IVarSerializationHelper
    {
        public string VarName { get; set; }
        bool IVarSerializationHelper.IsValue { get; }
        string[] IVarSerializationHelper.Values { get; }

        public DictionaryVar()
        {
            Contracts.Assert(Var<T>.CheckType(typeof(T)));
            VarName = $"Var_{Guid.NewGuid().ToString("N")}";
        }

        public Var<T> this[string key]
        {
            get
            {
                var item = new Var<T>();
                item.VarName = $"{VarName}[\"{key}\"]";
                return item;
            }
        }
    }

    /// <summary>
    /// A descriptor of one 'variable' of the graph (input or output that is referenced as a $variable in the graph definition).
    /// </summary>
    public sealed class EntryPointVariable
    {
        private readonly IExceptionContext _ectx;
        public readonly string Name;
        public readonly Type Type;

        /// <summary>
        /// The value. It will originally start as null, and then assigned to the value,
        /// once it is available. The type is one of the valid types according to <see cref="IsValidType"/>.
        /// </summary>
        public object Value { get; private set; }

        public bool HasInputs { get; private set; }

        public bool HasOutputs { get; private set; }

        public bool IsValueSet { get; private set; }

        /// <summary>
        /// Whether the given type is a valid one to be a variable.
        /// </summary>
        public static bool IsValidType(Type variableType)
        {
            Contracts.CheckValue(variableType, nameof(variableType));

            // Option types should not be used to consturct graph.
            if (variableType.IsGenericType && variableType.GetGenericTypeDefinition() == typeof(Optional<>))
                return false;

            if (variableType == typeof(CommonInputs.IEvaluatorInput))
                return true;
            if (variableType == typeof(CommonOutputs.IEvaluatorOutput))
                return true;
            if (variableType == typeof(IMlState))
                return true;

            var kind = TlcModule.GetDataType(variableType);
            if (kind == TlcModule.DataKind.Array)
            {
                if (!variableType.IsArray)
                {
                    Contracts.Assert(false, "Unexpected type for array variable");
                    return false;
                }
                return IsValidType(variableType.GetElementType());
            }

            if (kind == TlcModule.DataKind.Dictionary)
            {
                Contracts.Assert(variableType.IsGenericType && variableType.GetGenericTypeDefinition() == typeof(Dictionary<,>)
                                 && variableType.GetGenericTypeArgumentsEx()[0] == typeof(string));
                return IsValidType(variableType.GetGenericTypeArgumentsEx()[1]);
            }

            return kind == TlcModule.DataKind.DataView
                   || kind == TlcModule.DataKind.FileHandle
                   || kind == TlcModule.DataKind.PredictorModel
                   || kind == TlcModule.DataKind.TransformModel;
        }

        public EntryPointVariable(IExceptionContext ectx, string name, Type type)
        {
            Contracts.AssertValueOrNull(ectx);
            _ectx = ectx;
            _ectx.AssertNonEmpty(name);

            Name = name;
            ectx.Assert(IsValidType(type));
            Type = type;
        }

        /// <summary>
        /// Set the value. It is only allowed once.
        /// </summary>
        public void SetValue(object value)
        {
            _ectx.AssertValueOrNull(value);
            _ectx.Assert(!IsValueSet);
            _ectx.Assert(value == null || Type.IsAssignableFrom(value.GetType()));
            Value = value;
            IsValueSet = true;
        }

        public void MarkUsage(bool isInput)
        {
            if (isInput)
                HasInputs = true;
            else
                HasOutputs = true;
        }

        public EntryPointVariable Clone(string newName)
        {
            var v = new EntryPointVariable(_ectx, newName, Type);
            v.MarkUsage(HasInputs);
            v.IsValueSet = IsValueSet;
            v.Value = Value;
            return v;
        }
    }

    /// <summary>
    /// A collection of all known variables, with an interface to add new variables, get values based on names etc.
    /// This is populated by individual nodes when they parse their respective JSON definitions, and then the values are updated
    /// during the node execution.
    /// </summary>
    public sealed class RunContext
    {
        private readonly Dictionary<string, EntryPointVariable> _vars;
        private readonly IExceptionContext _ectx;
        private int _idCount;

        public RunContext(IExceptionContext ectx)
        {
            Contracts.AssertValueOrNull(ectx);
            _ectx = ectx;
            _vars = new Dictionary<string, EntryPointVariable>();
        }

        public bool TryGetVariable(string name, out EntryPointVariable v)
        {
            return _vars.TryGetValue(name, out v);
        }

        public object GetValueOrNull(VariableBinding binding)
        {
            _ectx.AssertValue(binding);
            EntryPointVariable v;
            if (!TryGetVariable(binding.VariableName, out v))
                return null;
            return binding.GetVariableValueOrNull(v);
        }

        public void AddInputVariable(VariableBinding binding, Type type)
        {
            _ectx.AssertValue(binding);
            _ectx.AssertValue(type);

            if (binding is ArrayIndexVariableBinding)
                type = Utils.MarshalInvoke(MakeArray<int>, type);
            else if (binding is DictionaryKeyVariableBinding)
                type = Utils.MarshalInvoke(MakeDictionary<int>, type);

            EntryPointVariable v;
            if (!_vars.TryGetValue(binding.VariableName, out v))
            {
                v = new EntryPointVariable(_ectx, binding.VariableName, type);
                _vars[binding.VariableName] = v;
            }
            else if (v.Type != type)
                throw _ectx.Except($"Variable '{v.Name}' is used as {v.Type} and as {type}");
            v.MarkUsage(true);
        }

        private Type MakeArray<T>()
        {
            return typeof(T[]);
        }

        private Type MakeDictionary<T>()
        {
            return typeof(Dictionary<string, T>);
        }

        public void RemoveVariable(EntryPointVariable variable)
        {
            _ectx.CheckValue(variable, nameof(variable));
            _vars.Remove(variable.Name);
        }

        /// <summary>
        /// Returns true if added new variable, false if variable already exists.
        /// </summary>
        public Boolean AddOutputVariable(string name, Type type)
        {
            _ectx.AssertNonEmpty(name);
            _ectx.AssertValue(type);

            EntryPointVariable v;
            if (!_vars.TryGetValue(name, out v))
            {
                v = new EntryPointVariable(_ectx, name, type);
                _vars[name] = v;
            }
            else
            {
                if (v.Type != type)
                    throw _ectx.Except($"Variable '{v.Name}' is used as {v.Type} and as {type}");
                return false;
            }
            v.MarkUsage(false);
            return true;
        }

        public string[] GetMissingInputs()
        {
            return _vars.Values.Where(x => x.HasInputs && !x.HasOutputs && !x.IsValueSet)
                .Select(x => x.Name)
                .ToArray();
        }

        public string GenerateId(string name)
        {
            return $"Node_{_idCount++:000}_{name.Replace(" ", "_")}";
        }

        public void AddContextVariables(RunContext subGraphRunContext)
        {
            foreach (var kvp in subGraphRunContext._vars)
            {
                EntryPointVariable v;
                if (!_vars.TryGetValue(kvp.Key, out v))
                    _vars.Add(kvp.Key, kvp.Value);
                else
                    throw _ectx.Except($"Duplicate variable '{kvp.Key}' in subgraph.");
            }
        }

        public void RenameContextVariable(string oldName, string newName)
        {
            if (_vars.ContainsKey(newName))
                throw _ectx.Except($"Variable with name '{newName}' already exists in subgraph.");
            if (!_vars.ContainsKey(oldName))
                throw _ectx.Except($"Variable with name '{oldName}' not found in subgraph.");
            var v = _vars[oldName].Clone(newName);
            _vars.Add(newName, v);
            _vars.Remove(oldName);
        }

        public EntryPointVariable CreateTempOutputVar<T>(string varPrefix)
        {
            _ectx.CheckValue(varPrefix, nameof(varPrefix));

            int id = 0;
            EntryPointVariable v;
            string name = $"{varPrefix}_{id}";
            while (_vars.TryGetValue(name, out v))
            {
                name = $"{varPrefix}_{id}";
                id++;
            }

            Type type = typeof(T);
            v = new EntryPointVariable(_ectx, name, type);
            _vars[name] = v;
            v.MarkUsage(false);
            return v;
        }
    }

    /// <summary>
    /// A representation of one graph node.
    /// </summary>
    public sealed class EntryPointNode
    {
        // The unique node ID, generated at compilation.
        public readonly string Id;

        private readonly IHost _host;
        private readonly ModuleCatalog _catalog;
        private readonly ModuleCatalog.EntryPointInfo _entryPoint;
        private readonly InputBuilder _inputBuilder;
        private readonly OutputHelper _outputHelper;

        // Reference to the global run context.
        private RunContext _context;

        // Mapping of input parameter names to a list of ParameterBindings. This list
        // will contain a single element when a variable is directly assigned to an input
        // parameter. When an input parameter is assigned an array or dictionary of variable
        // values this list will contain an entry for each needed assignment.
        private readonly Dictionary<string, List<ParameterBinding>> _inputBindingMap;

        private readonly Dictionary<ParameterBinding, VariableBinding> _inputMap;

        // Outputs are simple- we both can't bind index/keyed values to a variable and can't
        // bind a value to a variable index/key slot.
        private readonly Dictionary<string, string> _outputMap;

        public bool IsFinished { get; private set; }

        public TimeSpan RunTime { get; internal set; }

        private static Regex _stageIdRegex = new Regex(@"[a-zA-Z0-9]*", RegexOptions.Compiled);
        private string _stageId;
        /// <summary>
        /// An alphanumeric string indicating the stage of a node.
        /// The fact that the nodes share the same stage ID hints that they should be executed together whenever possible.
        /// </summary>
        public string StageId
        {
            get { return _stageId; }
            set
            {
                if (!IsStageIdValid(value))
                    throw _host.Except("Stage ID must be alphanumeric.");
                _stageId = value;
            }
        }

        /// <summary>
        /// Hints that the output of this node should be checkpointed.
        /// </summary>
        public bool Checkpoint { get; set; }

        private float _cost;
        /// <summary>
        /// The cost of running this node. NaN indicates unknown.
        /// </summary>
        public float Cost
        {
            get { return _cost; }
            set
            {
                if (value < 0)
                    throw _host.Except("Cost cannot be negative.");
                _cost = value;
            }
        }

        private EntryPointNode(IHostEnvironment env, IChannel ch, ModuleCatalog moduleCatalog, RunContext context,
            string id, string entryPointName, JObject inputs, JObject outputs, bool checkpoint = false,
            string stageId = "", float cost = float.NaN, string label = null, string group = null, string weight = null, string name = null)
        {
            Contracts.AssertValue(env);
            env.AssertNonEmpty(id);
            _host = env.Register(id);
            _host.AssertValue(context);
            _host.AssertNonEmpty(entryPointName);
            _host.AssertValue(moduleCatalog);
            _host.AssertValueOrNull(inputs);
            _host.AssertValueOrNull(outputs);

            _context = context;
            _catalog = moduleCatalog;

            Id = id;
            if (!moduleCatalog.TryFindEntryPoint(entryPointName, out _entryPoint))
                throw _host.Except($"Entry point '{entryPointName}' not found");

            // Validate inputs.
            _inputMap = new Dictionary<ParameterBinding, VariableBinding>();
            _inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            _inputBuilder = new InputBuilder(_host, _entryPoint.InputType, moduleCatalog);

            // REVIEW: This logic should move out of Node eventually and be delegated to
            // a class that can nest to handle Components with variables.
            if (inputs != null)
            {
                foreach (var pair in inputs)
                    CheckAndSetInputValue(pair);
            }
            var missing = _inputBuilder.GetMissingValues().Except(_inputBindingMap.Keys).ToArray();
            if (missing.Length > 0)
                throw _host.Except($"The following required inputs were not provided: {String.Join(", ", missing)}");

            var inputInstance = _inputBuilder.GetInstance();
            SetColumnArgument(ch, inputInstance, "LabelColumn", label, "label", typeof(CommonInputs.ITrainerInputWithLabel));
            SetColumnArgument(ch, inputInstance, "GroupIdColumn", group, "group Id", typeof(CommonInputs.ITrainerInputWithGroupId));
            SetColumnArgument(ch, inputInstance, "WeightColumn", weight, "weight", typeof(CommonInputs.ITrainerInputWithWeight), typeof(CommonInputs.IUnsupervisedTrainerWithWeight));
            SetColumnArgument(ch, inputInstance, "NameColumn", name, "name");

            // Validate outputs.
            _outputHelper = new OutputHelper(_host, _entryPoint.OutputType);
            _outputMap = new Dictionary<string, string>();
            if (outputs != null)
            {
                foreach (var pair in outputs)
                    CheckAndMarkOutputValue(pair);
            }

            Checkpoint = checkpoint;
            StageId = stageId;
            Cost = cost;
        }

        private void SetColumnArgument(IChannel ch, object inputInstance, string argName, string colName, string columnRole, params Type[] inputKinds)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(inputInstance);
            ch.AssertNonEmpty(argName);
            ch.AssertValueOrNull(colName);
            ch.AssertNonEmpty(columnRole);
            ch.AssertValueOrNull(inputKinds);

            var colField = _inputBuilder.GetFieldNameOrNull(argName);
            if (string.IsNullOrEmpty(colField))
                return;

            const string warning = "Different {0} column specified in trainer and in macro: '{1}', '{2}'." +
                " Using column '{2}'. To column use '{1}' instead, please specify this name in" +
                "the trainer node arguments.";
            if (!string.IsNullOrEmpty(colName) && Utils.Size(_entryPoint.InputKinds) > 0 &&
                (Utils.Size(inputKinds) == 0 || _entryPoint.InputKinds.Intersect(inputKinds).Any()))
            {
                ch.AssertNonEmpty(colField);
                var colFieldType = _inputBuilder.GetFieldTypeOrNull(colField);
                ch.Assert(colFieldType == typeof(string));
                var inputColName = inputInstance.GetType().GetField(colField).GetValue(inputInstance);
                ch.Assert(inputColName is string || inputColName is Optional<string>);
                var str = inputColName is string ? (string)inputColName : ((Optional<string>)inputColName).Value;
                if (colName != str)
                    ch.Warning(warning, columnRole, colName, inputColName);
                else
                    _inputBuilder.TrySetValue(colField, colName);
            }
        }

        public static EntryPointNode Create(
            IHostEnvironment env,
            string entryPointName,
            object arguments,
            ModuleCatalog catalog,
            RunContext context,
            Dictionary<string, List<ParameterBinding>> inputBindingMap,
            Dictionary<ParameterBinding, VariableBinding> inputMap,
            Dictionary<string, string> outputMap,
            bool checkpoint = false,
            string stageId = "",
            float cost = float.NaN)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckNonEmpty(entryPointName, nameof(entryPointName));
            env.CheckValue(arguments, nameof(arguments));
            env.CheckValue(catalog, nameof(catalog));
            env.CheckValue(context, nameof(context));
            env.CheckValue(inputBindingMap, nameof(inputBindingMap));
            env.CheckValue(inputMap, nameof(inputMap));
            env.CheckValue(outputMap, nameof(outputMap));
            ModuleCatalog.EntryPointInfo info;
            bool success = catalog.TryFindEntryPoint(entryPointName, out info);
            env.Assert(success);

            var inputBuilder = new InputBuilder(env, info.InputType, catalog);
            var outputHelper = new OutputHelper(env, info.OutputType);

            using (var ch = env.Start("Create EntryPointNode"))
            {
                var entryPointNode = new EntryPointNode(env, ch, catalog, context, context.GenerateId(entryPointName), entryPointName,
                    inputBuilder.GetJsonObject(arguments, inputBindingMap, inputMap),
                    outputHelper.GetJsonObject(outputMap), checkpoint, stageId, cost);

                ch.Done();
                return entryPointNode;
            }
        }

        public static EntryPointNode Create(
            IHostEnvironment env,
            string entryPointName,
            object arguments,
            ModuleCatalog catalog,
            RunContext context,
            Dictionary<string, string> inputMap,
            Dictionary<string, string> outputMap,
            bool checkpoint = false,
            string stageId = "",
            float cost = float.NaN)
        {
            ModuleCatalog.EntryPointInfo info;
            bool success = catalog.TryFindEntryPoint(entryPointName, out info);
            env.Assert(success);

            var inputBindingMap = new Dictionary<string, List<ParameterBinding>>();
            var inputParamBindingMap = new Dictionary<ParameterBinding, VariableBinding>();
            foreach (var kvp in inputMap)
            {
                var paramBinding = new SimpleParameterBinding(kvp.Key);
                inputBindingMap.Add(kvp.Key, new List<ParameterBinding>() { paramBinding });
                inputParamBindingMap.Add(paramBinding, new SimpleVariableBinding(kvp.Value));
            }

            return Create(env, entryPointName, arguments, catalog, context, inputBindingMap, inputParamBindingMap,
                outputMap, checkpoint, stageId, cost);
        }

        /// <summary>
        /// Checks the given JSON object key-value pair is a valid EntryPoint input and
        /// extracts out any variables that need to be populated. These variables will be
        /// added to the EntryPoint context. Input parameters that are not set to variables
        /// will be immediately set using the input builder instance.
        /// </summary>
        private void CheckAndSetInputValue(KeyValuePair<string, JToken> pair)
        {
            var inputName = _inputBuilder.GetFieldNameOrNull(pair.Key);
            if (VariableBinding.IsBindingToken(pair.Value))
            {
                Type valueType = _inputBuilder.GetFieldTypeOrNull(pair.Key);
                if (valueType == null)
                    throw _host.Except($"Unexpected input name: '{pair.Key}'");
                if (!EntryPointVariable.IsValidType(valueType))
                    throw _host.Except($"Unexpected input variable type: {valueType}");

                var varBinding = VariableBinding.Create(_host, pair.Value.Value<string>());
                _context.AddInputVariable(varBinding, valueType);
                if (!_inputBindingMap.ContainsKey(inputName))
                    _inputBindingMap[inputName] = new List<ParameterBinding>();
                var paramBinding = new SimpleParameterBinding(inputName);
                _inputBindingMap[inputName].Add(paramBinding);
                _inputMap[paramBinding] = varBinding;
            }
            else if (pair.Value is JArray &&
                     ((JArray)pair.Value).Any(tok => VariableBinding.IsBindingToken(tok)))
            {
                // REVIEW: EntryPoint arrays and dictionaries containing
                // variables must ONLY contain variables right now.
                if (!((JArray)pair.Value).All(tok => VariableBinding.IsBindingToken(tok)))
                    throw _host.Except($"Input {pair.Key} may ONLY contain variables.");

                Type valueType = _inputBuilder.GetFieldTypeOrNull(pair.Key);
                if (valueType == null || !valueType.HasElementType)
                    throw _host.Except($"Unexpected input name: '{pair.Key}'");
                valueType = valueType.GetElementType();

                int i = 0;
                foreach (var varName in (JArray)pair.Value)
                {
                    var varBinding = VariableBinding.Create(_host, varName.Value<string>());
                    _context.AddInputVariable(varBinding, valueType);
                    if (!_inputBindingMap.ContainsKey(inputName))
                        _inputBindingMap[inputName] = new List<ParameterBinding>();
                    var paramBinding = new ArrayIndexParameterBinding(inputName, i++);
                    _inputBindingMap[inputName].Add(paramBinding);
                    _inputMap[paramBinding] = varBinding;
                }
            }
            // REVIEW: Implement support for Dictionary of variable values. We need to differentiate
            // between a Dictionary and a Component here, and likely need to support nested components
            // all of which might have variables. Our current machinery only works at the 'Node' level.
            else
            {
                // This is not a variable.
                if (!_inputBuilder.TrySetValueJson(pair.Key, pair.Value))
                    throw _host.Except($"Unexpected input: '{pair.Key}'");
            }
        }

        /// <summary>
        /// Checks the given JSON object key-value pair is a valid EntryPoint output.
        /// Extracts out any variables that need to be populated and adds them to the
        /// EntryPoint context.
        /// </summary>
        private void CheckAndMarkOutputValue(KeyValuePair<string, JToken> pair)
        {
            if (!VariableBinding.IsBindingToken(pair.Value))
                throw _host.Except("Only variables allowed as outputs");

            // Output variable.
            var varBinding = VariableBinding.Create(_host, pair.Value.Value<string>());
            if (!(varBinding is SimpleVariableBinding))
                throw _host.Except($"Output '{pair.Key}' can only be bound to a variable");

            var valueType = _outputHelper.GetFieldType(pair.Key);
            if (valueType == null)
                throw _host.Except($"Unexpected output name: '{pair.Key}");

            if (!EntryPointVariable.IsValidType(valueType))
                throw _host.Except($"Output '{pair.Key}' has invalid type");

            _context.AddOutputVariable(varBinding.VariableName, valueType);
            _outputMap[pair.Key] = varBinding.VariableName;
        }

        public void RenameInputVariable(string oldName, VariableBinding newBinding)
        {
            var toModify = new List<ParameterBinding>();
            foreach (var kvp in _inputMap)
            {
                if (kvp.Value.VariableName == oldName)
                    toModify.Add(kvp.Key);
            }
            foreach (var parameterBinding in toModify)
                _inputMap[parameterBinding] = newBinding;
        }

        public void RenameOutputVariable(string oldName, string newName, bool cascadeChanges = false)
        {
            string key = null;
            foreach (var kvp in _outputMap)
            {
                if (kvp.Value == oldName)
                {
                    key = kvp.Key;
                    break;
                }
            }

            if (key != null)
            {
                _outputMap[key] = newName;
                if (cascadeChanges)
                    _context.RenameContextVariable(oldName, newName);
            }
        }

        public void RenameAllVariables(Dictionary<string, string> mapping)
        {
            string newName;
            foreach (var kvp in _inputMap)
            {
                if (!mapping.TryGetValue(kvp.Value.VariableName, out newName))
                {
                    newName = new Var<IDataView>().VarName;
                    mapping.Add(kvp.Value.VariableName, newName);
                }
                kvp.Value.Rename(newName);
            }

            var toModify = new Dictionary<string, string>();
            foreach (var kvp in _outputMap)
            {
                if (!mapping.TryGetValue(kvp.Value, out newName))
                {
                    newName = new Var<IDataView>().VarName;
                    mapping.Add(kvp.Value, newName);
                }
                toModify.Add(kvp.Key, newName);
            }
            foreach (var kvp in toModify)
                _outputMap[kvp.Key] = kvp.Value;
        }

        private static bool IsStageIdValid(string str)
        {
            return str != null && _stageIdRegex.Match(str).Success;
        }

        public JObject ToJson()
        {
            var result = new JObject();
            result[FieldNames.Name] = _entryPoint.Name;
            result[FieldNames.Inputs] = _inputBuilder.GetJsonObject(_inputBuilder.GetInstance(), _inputBindingMap, _inputMap);
            result[FieldNames.Outputs] = _outputHelper.GetJsonObject(_outputMap);
            if (!string.IsNullOrEmpty(StageId))
                result[FieldNames.StageId] = StageId;
            if (Checkpoint)
                result[FieldNames.Checkpoint] = Checkpoint;
            if (!float.IsNaN(Cost))
                result[FieldNames.Cost] = Cost;
            return result;
        }

        /// <summary>
        /// Whether the node can run right now.
        /// </summary>
        public bool CanStart()
        {
            if (IsFinished)
                return false;
            return _inputMap.Where(kv => !_inputBuilder.IsInputOptional(kv.Key.ParameterName)).Select(kv => kv.Value).Distinct()
                .All(varBinding => _context.TryGetVariable(varBinding.VariableName, out EntryPointVariable v) && v.IsValueSet);
        }

        public void Run()
        {
            _host.Assert(CanStart());
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            // Set all remaining inputs.
            foreach (var pair in _inputBindingMap)
            {
                bool success = _inputBuilder.TrySetValue(pair.Key, BuildParameterValue(pair.Value));
                _host.Assert(success);
            }

            _host.Assert(_inputBuilder.GetMissingValues().Length == 0);
            object output;

            if (IsMacro)
            {
                output = _entryPoint.Method.Invoke(null, new object[] { _host, _inputBuilder.GetInstance(), this });
                var macroResult = (CommonOutputs.MacroOutput)output;
                _host.AssertValue(macroResult);
                _macroNodes = macroResult.Nodes;
            }
            else
            {
                output = _entryPoint.Method.Invoke(null, new object[] { _host, _inputBuilder.GetInstance() });
                foreach (var pair in _outputHelper.ExtractValues(output))
                {
                    string tgt;
                    if (_outputMap.TryGetValue(pair.Key, out tgt))
                    {
                        EntryPointVariable v;
                        bool good = _context.TryGetVariable(tgt, out v);
                        _host.Assert(good);
                        v.SetValue(pair.Value);
                    }
                }
            }

            stopWatch.Stop();
            RunTime = stopWatch.Elapsed;
            IsFinished = true;
        }

        public bool IsMacro => _entryPoint.OutputType.IsSubclassOf(typeof(CommonOutputs.MacroOutput));

        private IEnumerable<EntryPointNode> _macroNodes;

        public IEnumerable<EntryPointNode> MacroNodes => _macroNodes;
        public ModuleCatalog Catalog => _catalog;
        public RunContext Context => _context;
        public Dictionary<string, List<ParameterBinding>> InputBindingMap => _inputBindingMap;
        public Dictionary<ParameterBinding, VariableBinding> InputMap => _inputMap;
        public Dictionary<string, string> OutputMap => _outputMap;
        public override string ToString() => Id;

        private object BuildParameterValue(List<ParameterBinding> bindings)
        {
            _host.AssertNonEmpty(bindings);

            var firstBinding = bindings.First();
            _host.Assert(bindings.Skip(1).All(binding => binding.GetType().Equals(firstBinding.GetType())));

            if (firstBinding is SimpleParameterBinding)
            {
                _host.Assert(bindings.Count == 1);
                return _context.GetValueOrNull(_inputMap[firstBinding]);
            }
            if (firstBinding is ArrayIndexParameterBinding)
            {
                var type = _inputBuilder.GetFieldTypeOrNull(firstBinding.ParameterName).GetElementType();
                _host.AssertValue(type);
                var arr = Array.CreateInstance(type, bindings.Count);
                int i = 0;
                foreach (var binding in bindings)
                    arr.SetValue(_context.GetValueOrNull(_inputMap[binding]), i++);
                return arr;
            }
            if (firstBinding is DictionaryKeyParameterBinding)
            {
                // REVIEW: Implement dictionary support when needed;
                throw _host.ExceptNotImpl("Dictionary variable binding is not currently supported");
            }

            _host.Assert(false);
            throw _host.ExceptNotImpl("Unsupported ParameterBinding");
        }

        public static List<EntryPointNode> ValidateNodes(IHostEnvironment env, RunContext context, JArray nodes,
            ModuleCatalog moduleCatalog, string label = null, string group = null, string weight = null, string name = null)
        {
            Contracts.AssertValue(env);
            env.AssertValue(context);
            env.AssertValue(nodes);
            env.AssertValue(moduleCatalog);

            var result = new List<EntryPointNode>(nodes.Count);
            using (var ch = env.Start("Validating graph nodes"))
            {
                for (int i = 0; i < nodes.Count; i++)
                {
                    var node = nodes[i] as JObject;
                    if (node == null)
                        throw env.Except("Unexpected node token: '{0}'", nodes[i]);

                    string nodeName = node[FieldNames.Name].Value<string>();
                    var inputs = node[FieldNames.Inputs] as JObject;
                    if (inputs == null && node[FieldNames.Inputs] != null)
                        throw env.Except("Unexpected {0} token: '{1}'", FieldNames.Inputs, node[FieldNames.Inputs]);

                    var outputs = node[FieldNames.Outputs] as JObject;
                    if (outputs == null && node[FieldNames.Outputs] != null)
                        throw env.Except("Unexpected {0} token: '{1}'", FieldNames.Outputs, node[FieldNames.Outputs]);

                    var id = context.GenerateId(nodeName);
                    var unexpectedFields = node.Properties().Where(
                        x => x.Name != FieldNames.Name && x.Name != FieldNames.Inputs && x.Name != FieldNames.Outputs
                        && x.Name != FieldNames.StageId && x.Name != FieldNames.Checkpoint && x.Name != FieldNames.Cost);

                    var stageId = node[FieldNames.StageId] == null ? "" : node[FieldNames.StageId].Value<string>();
                    var checkpoint = node[FieldNames.Checkpoint] == null ? false : node[FieldNames.Checkpoint].Value<bool>();
                    var cost = node[FieldNames.Cost] == null ? float.NaN : node[FieldNames.Cost].Value<float>();

                    if (unexpectedFields.Any())
                    {
                        // REVIEW: consider throwing an exception.
                        ch.Warning("Node '{0}' has unexpected fields that are ignored: {1}", id, string.Join(", ", unexpectedFields.Select(x => x.Name)));
                    }

                    result.Add(new EntryPointNode(env, ch, moduleCatalog, context, id, nodeName, inputs, outputs, checkpoint, stageId, cost, label, group, weight, name));
                }

                ch.Done();
            }
            return result;
        }

        public void SetContext(RunContext context)
        {
            _host.CheckValue(context, nameof(context));
            _context = context;
        }

        public VariableBinding GetInputVariable(string paramName)
        {
            List<ParameterBinding> parameterBindings;
            bool success = InputBindingMap.TryGetValue(paramName, out parameterBindings);
            if (!success)
                throw _host.Except($"Invalid parameter '{paramName}': parameter does not exist.");
            if (parameterBindings == null || parameterBindings.Count > 1)
                throw _host.Except($"Invalid parameter '{paramName}': only simple parameters are supported.");
            VariableBinding variableBinding;
            success = InputMap.TryGetValue(parameterBindings[0], out variableBinding);
            _host.Assert(success && variableBinding != null);
            return variableBinding;
        }

        public string GetOutputVariableName(string paramName)
        {
            string outputVarName;
            bool success = OutputMap.TryGetValue(paramName, out outputVarName);
            if (!success)
                throw _host.Except($"Invalid parameter '{paramName}': parameter does not exist.");
            return outputVarName;
        }

        public Tuple<Var<T>, VariableBinding> AddNewVariable<T>(string uniqueName, T value)
        {
            // Make sure name is really unique.
            if (InputBindingMap.ContainsKey(uniqueName))
                throw _host.Except($"Key {uniqueName} already exists in binding map.");

            // Add parameter bindings
            var paramBinding = new SimpleParameterBinding(uniqueName);
            InputBindingMap.Add(uniqueName, new List<ParameterBinding> { paramBinding });

            // Create new variables
            var varBinding = new SimpleVariableBinding(uniqueName);
            Context.AddInputVariable(varBinding, typeof(T));
            InputMap.Add(paramBinding, varBinding);

            // Set value
            if (value != null && Context.TryGetVariable(varBinding.VariableName, out var variable))
                variable.SetValue(value);

            // Return Var<> object and variable binding
            return new Tuple<Var<T>, VariableBinding>(new Var<T> { VarName = varBinding.VariableName }, varBinding);
        }
    }

    public sealed class EntryPointGraph
    {
        private const string RegistrationName = "EntryPointGraph";
        private readonly IHost _host;

        private readonly RunContext _context;
        private readonly List<EntryPointNode> _nodes;

        public EntryPointGraph(IHostEnvironment env, ModuleCatalog moduleCatalog, JArray nodes)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(moduleCatalog, nameof(moduleCatalog));
            _host.CheckValue(nodes, nameof(nodes));

            _context = new RunContext(_host);
            _nodes = EntryPointNode.ValidateNodes(_host, _context, nodes, moduleCatalog);
        }

        public bool HasRunnableNodes => _nodes.FirstOrDefault(x => x.CanStart()) != null;
        public IEnumerable<EntryPointNode> Macros => _nodes.Where(x => x.IsMacro);
        public IEnumerable<EntryPointNode> NonMacros => _nodes.Where(x => !x.IsMacro);
        public IEnumerable<EntryPointNode> AllNodes => _nodes;
        public RunContext Context => _context;

        public string[] GetMissingInputs()
        {
            return _context.GetMissingInputs();
        }

        public void RunNode(EntryPointNode node)
        {
            _host.CheckValue(node, nameof(node));
            _host.Assert(_nodes.Contains(node));

            node.Run();
            if (node.IsMacro)
                _nodes.AddRange(node.MacroNodes);
        }

        public bool TryGetVariable(string name, out EntryPointVariable v)
        {
            return _context.TryGetVariable(name, out v);
        }

        public EntryPointVariable GetVariableOrNull(string name)
        {
            EntryPointVariable var;
            if (TryGetVariable(name, out var))
                return var;
            return null;
        }

        public void AddNode(EntryPointNode node)
        {
            _host.CheckValue(node, nameof(node));
            node.SetContext(_context);
            _nodes.Add(node);
        }
    }

    /// <summary>
    /// Represents a delayed binding in a JSON graph to an <see cref="EntryPointVariable"/>.
    /// The subclasses allow us to express that we either desire the variable itself,
    /// or a array-indexed or dictionary-keyed value from the variable, assuming it is
    /// of an Array or Dictionary type.
    /// </summary>
    public abstract class VariableBinding
    {
        public string VariableName { get; private set; }

        protected VariableBinding(string varName)
        {
            Contracts.AssertNonWhiteSpace(varName);
            VariableName = varName;
        }

        // A regex to validate an EntryPoint variable value accessor string. Valid EntryPoint variable names
        // can be any sequence of alphanumeric characters and underscores. They must start with a letter or underscore.
        // An EntryPoint variable can be followed with an array or dictionary specifier, which begins
        // with '[', contains either an integer or alphanumeric string, optionally wrapped in single-quotes,
        // followed with ']'.
        private static Regex _variableRegex = new Regex(
            @"\$(?<Name>[a-zA-Z_][a-zA-Z0-9_]*)(\[(((?<NumericAccessor>[0-9]*))|(\'?(?<StringAccessor>[a-zA-Z0-9_]*)\'?))\])?",
            RegexOptions.Compiled);

        public abstract object GetVariableValueOrNull(EntryPointVariable variable);

        public static VariableBinding Create(IExceptionContext ectx, string jsonString)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertNonWhiteSpace(jsonString);
            var match = _variableRegex.Match(jsonString);

            if (!match.Success)
                throw ectx.Except($"Unable to parse variable string '{jsonString}'");

            if (match.Groups["NumericAccessor"].Success)
            {
                return new ArrayIndexVariableBinding(
                    match.Groups["Name"].Value,
                    int.Parse(match.Groups["NumericAccessor"].Value));
            }

            if (match.Groups["StringAccessor"].Success)
            {
                return new DictionaryKeyVariableBinding(
                    match.Groups["Name"].Value,
                    match.Groups["StringAccessor"].Value);
            }

            return new SimpleVariableBinding(match.Groups["Name"].Value);
        }

        public static bool IsBindingToken(JToken tok)
        {
            var token = tok as JValue;
            return token?.Value != null && _variableRegex.IsMatch(token.Value<string>());
        }

        /// <summary>
        /// Verifies that the name of the graph variable is a valid one
        /// </summary>
        public static bool IsValidVariableName(IExceptionContext ectx, string variableName)
        {
            Contracts.AssertValue(ectx);
            ectx.AssertNonWhiteSpace(variableName);

            return _variableRegex.Match(variableName).Success;
        }

        public void Rename(string newName)
        {
            Contracts.CheckNonWhiteSpace(newName, nameof(newName));
            VariableName = newName;
        }

        public abstract string ToJson();

        public override string ToString() => VariableName;
    }

    public sealed class SimpleVariableBinding
        : VariableBinding
    {
        public SimpleVariableBinding(string name)
            : base(name)
        { }

        public override object GetVariableValueOrNull(EntryPointVariable variable)
        {
            Contracts.AssertValue(variable);
            return variable.Value;
        }

        public override string ToJson()
        {
            return $"${VariableName}";
        }
    }

    public sealed class DictionaryKeyVariableBinding
        : VariableBinding
    {
        public readonly string Key;

        public DictionaryKeyVariableBinding(string name, string key)
            : base(name)
        {
            Contracts.AssertNonWhiteSpace(key);
            Key = key;
        }

        public override object GetVariableValueOrNull(EntryPointVariable variable)
        {
            Contracts.AssertValue(variable);
            // REVIEW: Implement dictionary-based value retrieval.
            throw Contracts.ExceptNotImpl("Diction-based value retrieval is not supported.");
        }

        public override string ToJson()
        {
            return $"${VariableName}['{Key}']";
        }
    }

    public sealed class ArrayIndexVariableBinding
        : VariableBinding
    {
        public readonly int Index;

        public ArrayIndexVariableBinding(string name, int index)
            : base(name)
        {
            Contracts.Assert(index >= 0);
            Index = index;
        }

        public override object GetVariableValueOrNull(EntryPointVariable variable)
        {
            Contracts.AssertValue(variable, nameof(variable));
            var arr = variable.Value as Array;
            return arr?.GetValue(Index);
        }

        public override string ToJson()
        {
            return $"${VariableName}[{Index}]";
        }
    }

    /// <summary>
    /// Represents the l-value assignable destination of a <see cref="VariableBinding"/>.
    /// Subclasses exist to express the needed bindinds for subslots
    /// of a yet-to-be-constructed array or dictionary EntryPoint input parameter
    /// (for example, "myVar": ["$var1", "$var2"] would yield two <see cref="ArrayIndexParameterBinding"/>: (myVar, 0), (myVar, 1))
    /// </summary>
    public abstract class ParameterBinding
    {
        public readonly string ParameterName;

        protected ParameterBinding(string name)
        {
            Contracts.AssertNonWhiteSpace(name);
            ParameterName = name;
        }

        public override string ToString() => ParameterName;
    }

    public sealed class SimpleParameterBinding
        : ParameterBinding
    {
        public SimpleParameterBinding(string name)
            : base(name)
        { }

        public override bool Equals(object obj)
        {
            var asSelf = obj as SimpleParameterBinding;
            if (asSelf == null)
                return false;
            return asSelf.ParameterName.Equals(ParameterName, StringComparison.Ordinal);
        }

        public override int GetHashCode()
        {
            return ParameterName.GetHashCode();
        }
    }

    public sealed class DictionaryKeyParameterBinding
        : ParameterBinding
    {
        public readonly string Key;

        public DictionaryKeyParameterBinding(string name, string key)
            : base(name)
        {
            Contracts.AssertNonWhiteSpace(key);
            Key = key;
        }

        public override bool Equals(object obj)
        {
            var asSelf = obj as DictionaryKeyParameterBinding;
            if (asSelf == null)
                return false;
            return
                asSelf.ParameterName.Equals(ParameterName, StringComparison.Ordinal) &&
                asSelf.Key.Equals(Key, StringComparison.Ordinal);
        }

        public override int GetHashCode()
        {
            return Tuple.Create(ParameterName, Key).GetHashCode();
        }
    }

    public sealed class ArrayIndexParameterBinding
        : ParameterBinding
    {
        public readonly int Index;
        public ArrayIndexParameterBinding(string name, int index)
            : base(name)
        {
            Contracts.Check(index >= 0);
            Index = index;
        }

        public override bool Equals(object obj)
        {
            var asSelf = obj as ArrayIndexParameterBinding;
            if (asSelf == null)
                return false;
            return
                asSelf.ParameterName.Equals(ParameterName, StringComparison.Ordinal) &&
                asSelf.Index == Index;
        }

        public override int GetHashCode()
        {
            return Tuple.Create(ParameterName, Index).GetHashCode();
        }
    }
}
