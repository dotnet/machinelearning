// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Runtime.Internal.Utilities;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.EntryPoints.JsonUtils
{
    /// <summary>
    /// This class runs a graph of entry points with the specified inputs, and produces the specified outputs.
    /// The entry point graph is provided as a <see cref="JArray"/> of graph nodes. The inputs need to be provided separately:
    /// the graph runner will only compile a list of required inputs, and the calling code is expected to set them prior
    /// to running the graph.
    /// 
    /// REVIEW: currently, the graph is executed synchronously, one node at a time. This is an implementation choice, we
    /// probably need to consider parallel asynchronous execution, once we agree on an acceptable syntax for it.
    /// </summary>
    public sealed class GraphRunner
    {
        private const string RegistrationName = "GraphRunner";
        private readonly IHost _host;
        private readonly EntryPointGraph _graph;

        public GraphRunner(IHostEnvironment env, ModuleCatalog moduleCatalog, JArray nodes)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(moduleCatalog, nameof(moduleCatalog));
            _host.CheckValue(nodes, nameof(nodes));

            _graph = new EntryPointGraph(_host, moduleCatalog, nodes);
        }

        public GraphRunner(IHostEnvironment env, EntryPointGraph graph)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _graph = graph;
        }

        /// <summary>
        /// Run all nodes in the graph.
        /// </summary>
        public void RunAll()
        {
            var missingInputs = _graph.GetMissingInputs();
            if (missingInputs.Any())
                throw _host.Except("The following inputs are missing: {0}", string.Join(", ", missingInputs));

            while (_graph.HasRunnableNodes)
            {
                ExpandAllMacros();
                OptimizeGraph();
                RunAllNonMacros();
            }

            var remainingNodes = _graph.Macros.Union(_graph.NonMacros).Where(x => !x.IsFinished).Select(x => x.Id).ToArray();
            if (remainingNodes.Any())
                throw _host.Except("The following nodes didn't run due to circular dependency: {0}", string.Join(", ", remainingNodes));
        }

        private void RunAllNonMacros()
        {
            EntryPointNode nextNode;
            while ((nextNode = _graph.NonMacros.FirstOrDefault(x => x.CanStart())) != null)
                _graph.RunNode(nextNode);
        }

        private void ExpandAllMacros()
        {
            EntryPointNode nextNode;
            while ((nextNode = _graph.Macros.FirstOrDefault(x => x.CanStart())) != null)
                _graph.RunNode(nextNode);
        }

        private void OptimizeGraph()
        {
            // REVIEW: Insert smart graph optimizer here.
            // For now, do nothing.
        }

        /// <summary>
        /// Retrieve an output of the experiment graph.
        /// </summary>
        public TOutput GetOutput<TOutput>(string name)
            where TOutput : class
        {
            _host.CheckNonEmpty(name, nameof(name));
            EntryPointVariable variable;

            if (!_graph.TryGetVariable(name, out variable))
                throw _host.Except("Port '{0}' not found", name);
            if (variable.Value == null)
                return null;

            var result = variable.Value as TOutput;
            if (result == null)
                throw _host.Except("Incorrect type for output '{0}'", name);
            return result;
        }

        /// <summary>
        /// Get the value of an EntryPointVariable present in the graph, or returns null.
        /// </summary>
        public TOutput GetOutputOrDefault<TOutput>(string name)
        {
            _host.CheckNonEmpty(name, nameof(name));

            if (_graph.TryGetVariable(name, out EntryPointVariable variable))
                if(variable.Value is TOutput)
                    return (TOutput)variable.Value;

            return default;
        }

        /// <summary>
        /// Set the input of the experiment graph.
        /// </summary>
        public void SetInput<TInput>(string name, TInput input)
            where TInput : class
        {
            _host.CheckNonEmpty(name, nameof(name));
            _host.CheckValue(input, nameof(input));

            EntryPointVariable variable;

            if (!_graph.TryGetVariable(name, out variable))
                throw _host.Except("Port '{0}' not found", name);
            if (variable.HasOutputs)
                throw _host.Except("Port '{0}' is not an input", name);
            if (variable.IsValueSet)
                throw _host.Except("Port '{0}' is already set", name);
            if (!variable.Type.IsAssignableFrom(typeof(TInput)))
                throw _host.Except("Port '{0}' is of incorrect type", name);

            variable.SetValue(input);
        }

        /// <summary>
        /// Get the data kind of a particular port.
        /// </summary>
        public TlcModule.DataKind GetPortDataKind(string name)
        {
            _host.CheckNonEmpty(name, nameof(name));
            EntryPointVariable variable;

            if (!_graph.TryGetVariable(name, out variable))
                throw _host.Except("Variable '{0}' not found", name);
            return TlcModule.GetDataType(variable.Type);
        }
    }
}
