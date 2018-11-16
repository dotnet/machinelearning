// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.Model.Pfa
{
    /// <summary>
    /// A context for defining a restricted sort of PFA output.
    /// </summary>
    [BestFriend]
    internal sealed class PfaContext
    {
        public JToken InputType { get; set; }
        public JToken OutputType { get; set; }
        public JToken Final { get; set; }

        private readonly HashSet<string> _variables;
        private readonly List<CellBlock> _cellBlocks;
        private readonly List<VariableBlock> _letSetBlocks;
        private readonly List<FuncBlock> _funcBlocks;
        private readonly HashSet<string> _types;
        private readonly IHost _host;

        private readonly struct VariableBlock
        {
            public readonly string Type;
            public readonly KeyValuePair<string, JToken>[] Locals;

            public VariableBlock(string type, KeyValuePair<string, JToken>[] locals)
            {
                Type = type;
                Locals = locals;
            }

            public JToken ToToken()
            {
                var vars = new JObject();
                foreach (var v in Locals)
                    vars[v.Key] = v.Value;
                var blockJson = new JObject();
                blockJson[Type] = vars;
                return blockJson;
            }
        }

        private readonly struct CellBlock
        {
            public readonly string Name;
            public readonly JToken Type;
            public readonly JToken Init;

            public CellBlock(string name, JToken type, JToken init)
            {
                Name = name;
                Type = type;
                Init = init;
            }

            public JObject ToToken()
            {
                var vars = new JObject();
                vars["type"] = Type;
                vars["init"] = Init;
                return vars;
            }
        }

        private readonly struct FuncBlock
        {
            public readonly string Name;
            public readonly JArray Params;
            public readonly JToken ReturnType;
            public readonly JToken Do;

            public FuncBlock(string name, JArray prms, JToken returnType, JToken doBlock)
            {
                Name = name;
                Params = prms;
                ReturnType = returnType;
                Do = doBlock;
            }

            public JObject ToToken()
            {
                var func = new JObject();
                func["params"] = Params;
                func["ret"] = ReturnType;
                func["do"] = Do;
                return func;
            }
        }

        public PfaContext(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(PfaContext));
            _variables = new HashSet<string>();
            _cellBlocks = new List<CellBlock>();
            _letSetBlocks = new List<VariableBlock>();
            _funcBlocks = new List<FuncBlock>();
            _types = new HashSet<string>();
        }

        public JObject Finalize()
        {
            var val = new JObject();
            val["input"] = InputType;
            val["output"] = OutputType;

            // Add the cells.
            if (_cellBlocks.Count > 0)
            {
                var cells = new JObject();
                foreach (var cell in _cellBlocks)
                    cells[cell.Name] = cell.ToToken();
                val["cells"] = cells;
            }

            // Add the actions.
            if (_letSetBlocks.Count > 0)
            {
                var actions = new JArray();
                foreach (var block in _letSetBlocks)
                    actions.Add(block.ToToken());
                actions.Add(Final);
                val["action"] = actions;
            }
            else
                val["action"] = Final;

            // Add the functions at the end.
            if (_funcBlocks.Count > 0)
            {
                var funcs = new JObject();
                foreach (var block in _funcBlocks)
                    funcs[block.Name] = block.ToToken();
                val["fcns"] = funcs;
            }

            return val;
        }

        public KeyValuePair<string, JToken> CreatePair(string varName, string token)
        {
            return new KeyValuePair<string, JToken>(varName, JToken.Parse(token));
        }

        public void AddVariables(params KeyValuePair<string, JToken>[] locals)
        {
            // Add as lets, then sets.
            if (locals.Length == 0)
                return;
            var sets = new List<KeyValuePair<string, JToken>>();
            foreach (var l in locals)
            {
                if (_variables.Contains(l.Key))
                    sets.Add(l);
            }
            // If either all or none of the inputs are sets, we can simplify the logic slightly.
            if (sets.Count == 0 || locals.Length == sets.Count)
            {
                _letSetBlocks.Add(new VariableBlock(sets.Count == 0 ? "let" : "set", locals));
                _variables.UnionWith(locals.Select(v => v.Key));
                return;
            }
            var lets = new List<KeyValuePair<string, JToken>>(locals.Length - sets.Count);

            foreach (var l in locals)
            {
                if (!_variables.Contains(l.Key))
                    lets.Add(l);
            }
            _variables.UnionWith(locals.Select(v => v.Key));
            // We must do the lets first.
            _letSetBlocks.Add(new VariableBlock("let", lets.ToArray()));
            _letSetBlocks.Add(new VariableBlock("set", sets.ToArray()));
        }

        public void AddCell(string name, JToken type, JToken init)
        {
            Contracts.CheckValue(name, nameof(name));
            if (ContainsCell(name))
                throw Contracts.ExceptParam(nameof(name), $"Cell {name} already exists");
            Contracts.CheckValue(type, nameof(type));
            Contracts.CheckValue(init, nameof(init));
            _cellBlocks.Add(new CellBlock(name, type, init));
        }

        public void AddFunc(string name, JArray prms, JToken returnType, JToken doBlock)
        {
            _funcBlocks.Add(new FuncBlock(name, prms, returnType, doBlock));
        }

        /// <summary>
        /// For creating an anonymous function block. This in itself will not modify the context.
        /// </summary>
        public static JObject CreateFuncBlock(JArray prms, JToken returnType, JToken doBlock)
        {
            return new FuncBlock("foo", prms, returnType, doBlock).ToToken();
        }

        public bool ContainsCell(string name) => _cellBlocks.Any(c => c.Name == name);

        public bool ContainsVar(string name) => _variables.Contains(name);

        public bool ContainsFunc(string name) => _funcBlocks.Any(b => b.Name == name);

        public bool ContainsType(string name) => _types.Contains(name);

        /// <summary>
        /// PFA is weird in that you do not declare types separately, you declare them as part of a variable
        /// declaration. So, if you use a record type three times, that means one of the three usages must be
        /// accompanied by a full type declaration, whereas the other two can just then identify it by name.
        /// This is extremely silly, but there you go.
        ///
        /// Anyway: this will attempt to add a type to the list of registered types. If it returns <c>true</c>
        /// then the caller is responsible, then, for ensuring that their PFA code they are generating contains
        /// not only a reference of the type, but a declaration of the type. If however this returns <c>false</c>
        /// then it can just refer to the type by name, since it has already been declared.
        /// </summary>
        /// <param name="name">The type to register</param>
        /// <returns>If this name was not already registered</returns>
        /// <seealso cref="ContainsType(string)"/>
        public bool RegisterType(string name)
        {
            return _types.Add(name);
        }
    }
}
