// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Newtonsoft.Json.Linq;

namespace Microsoft.ML.Runtime.Model.Pfa
{
    using T = PfaUtils.Type;

    /// <summary>
    /// This wraps a <see cref="PfaContext"/>, except with auxiliary information
    /// that enables its inclusion relative to the <see cref="IDataView"/> ecosystem.
    /// The idea is that one starts with a context built from some starting point,
    /// then subsequent transforms via <see cref="ITransformCanSavePfa"/> augment this context.
    /// Beyond what is offered in <see cref="PfaContext"/>, <see cref="BoundPfaContext"/>
    /// has facilities to remember what column name in <see cref="IDataView"/> maps to
    /// what token in the PFA being built up.
    /// </summary>
    public sealed class BoundPfaContext
    {
        /// <summary>
        /// The internal PFA context, for an escape hatch.
        /// </summary>
        public PfaContext Pfa { get; }

        /// <summary>
        /// This will map from the "current" name of a data view column, to a corresponding
        /// token string.
        /// </summary>
        private readonly Dictionary<string, string> _nameToVarName;
        /// <summary>
        /// This contains a map of those names in
        /// </summary>
        private readonly HashSet<string> _unavailable;

        private readonly bool _allowSet;
        private readonly IHost _host;

        public BoundPfaContext(IHostEnvironment env, ISchema inputSchema, HashSet<string> toDrop, bool allowSet)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(BoundPfaContext));
            _host.CheckValue(inputSchema, nameof(inputSchema));
            _host.CheckValue(toDrop, nameof(toDrop));

            Pfa = new PfaContext(_host);
            _nameToVarName = new Dictionary<string, string>();
            _unavailable = new HashSet<string>();
            _allowSet = allowSet;
            SetInput(inputSchema, toDrop);
        }

        private void SetInput(ISchema schema, HashSet<string> toDrop)
        {
            var recordType = new JObject();
            recordType["type"] = "record";
            recordType["name"] = "DataInput";
            var fields = new JArray();
            var fieldNames = new HashSet<string>();
            for (int c = 0; c < schema.ColumnCount; ++c)
            {
                if (schema.IsHidden(c))
                    continue;
                string name = schema.GetColumnName(c);
                if (toDrop.Contains(name))
                    continue;
                JToken pfaType = PfaTypeOrNullForColumn(schema, c);
                if (pfaType == null)
                    continue;
                string fieldName = ModelUtils.CreateNameCore(name, fieldNames.Contains);
                fieldNames.Add(fieldName);
                var fieldDeclaration = new JObject();
                fieldDeclaration["name"] = fieldName;
                fieldDeclaration["type"] = pfaType;
                fields.Add(fieldDeclaration);
                _nameToVarName.Add(name, "input." + fieldName);
            }
            _host.Assert(_nameToVarName.Count == fields.Count);
            _host.Assert(_nameToVarName.Count == fieldNames.Count);
            recordType["fields"] = fields;
            _host.Check(fields.Count >= 1, "Schema produced no inputs for the PFA conversion.");
            if (fields.Count == 1)
            {
                // If there's only one, don't bother forming a record.
                var field = (JObject)fields[0];
                Pfa.InputType = field["type"];
                _nameToVarName[_nameToVarName.Keys.First()] = "input";
            }
            else
                Pfa.InputType = recordType;
        }

        /// <summary>
        /// This call will set <see cref="PfaContext.OutputType"/> to an appropriate output type based
        /// on the columns requested.
        /// </summary>
        /// <param name="schema">The schema corresponding to what we are outputting</param>
        /// <param name="toOutput">The columns to output</param>
        /// <returns>Returns a complete PFA program, where the output will correspond to the subset
        /// of columns from <paramref name="schema"/>.</returns>
        public JObject Finalize(ISchema schema, params string[] toOutput)
        {
            _host.CheckValue(schema, nameof(schema));
            _host.CheckValue(toOutput, nameof(toOutput));
            JToken lastType = null;
            string lastToken = null;

            var recordType = new JObject();
            var newStatement = new JObject();
            recordType["type"] = "record";
            recordType["name"] = "DataOutput";
            var fields = new JArray();
            var fieldNames = new HashSet<string>();

            foreach (var name in toOutput)
            {
                _host.CheckParam(name != null, nameof(toOutput), "Null values in array");
                if (!schema.TryGetColumnIndex(name, out int col))
                    throw _host.ExceptParam(nameof(toOutput), $"Requested column '{name}' not in schema");
                JToken type = PfaTypeOrNullForColumn(schema, col);
                if (type == null)
                    continue;
                string token = TokenOrNullForName(name);
                if (token == null) // Not available.
                    continue;

                // We can write it out.
                lastType = type;
                lastToken = token;

                string fieldName = ModelUtils.CreateNameCore(name, fieldNames.Contains);
                fieldNames.Add(fieldName);
                var fieldDeclaration = new JObject();
                fieldDeclaration["name"] = fieldName;
                fieldDeclaration["type"] = type;
                fields.Add(fieldDeclaration);

                newStatement[fieldName] = token;
            }
            recordType["fields"] = fields;

            _host.Check(fields.Count >= 1, "Pipeline produced no outputs for the PFA conversion");
            if (fields.Count == 1)
            {
                Pfa.OutputType = lastType;
                Pfa.Final = lastToken;
                return Pfa.Finalize();
            }
            var expr = new JObject();
            expr["type"] = "DataOutput";
            expr["new"] = newStatement;

            Pfa.OutputType = recordType;
            Pfa.Final = expr;

            return Pfa.Finalize();
        }

        private JToken PfaTypeOrNullForColumn(ISchema schema, int col)
        {
            _host.AssertValue(schema);
            _host.Assert(0 <= col && col < schema.ColumnCount);

            ColumnType type = schema.GetColumnType(col);
            return T.PfaTypeOrNullForColumnType(type);
        }

        private string CreateNameVar(string name)
        {
            _host.CheckValueOrNull(name);
            if (name == null)
                return ModelUtils.CreateNameCore("temp", Pfa.ContainsVar);
            _host.CheckNonEmpty(name, nameof(name));
            if (!_allowSet)
                return ModelUtils.CreateNameCore(name, Pfa.ContainsVar);
            _nameToVarName.TryGetValue(name, out string exclude);
            // We allow "hiding" of prior names, similar to how the IDV does.
            // We assume that if a name is requested "twice" then IDV name
            // hiding is going on, in which case it's fine to re-use the name.
            return ModelUtils.CreateNameCore(name, n => n != exclude && Pfa.ContainsVar(n));
        }

        private string CreateNameCell(string name)
        {
            return ModelUtils.CreateNameCore(name, Pfa.ContainsCell);
        }

        /// <summary>
        /// Attempts to declare variables corresponding to a given column name. This
        /// will attempt to produce a PFA <c>let</c>/<c>set</c> declaration, and also
        /// do name mapping. The idea is that any transform implementing <see cref="ITransformCanSavePfa"/>
        /// will call this method to say, "hey, I produce this column, and this is the equivalent
        /// PFA for it."
        /// </summary>
        /// <param name="vars">The map from requested name, usually a dataview name,
        /// to the declaration</param>
        /// <returns>An array of assigned names in the PFA corresponding to the items in
        /// vars</returns>
        public string[] DeclareVar(params KeyValuePair<string, JToken>[] vars)
        {
            _host.CheckValue(vars, nameof(vars));
            var names = new string[vars.Length];
            for (int i = 0; i < vars.Length; ++i)
            {
                string colName = vars[i].Key;
                names[i] = CreateNameVar(colName);
                if (colName != null)
                    _nameToVarName[colName] = names[i];
                vars[i] = new KeyValuePair<string, JToken>(names[i], vars[i].Value);
            }
            Pfa.AddVariables(vars);
            return names;
        }

        public string DeclareVar(string name, JToken value)
        {
            _host.CheckValueOrNull(name);
            _host.CheckValue(value, nameof(value));
            return DeclareVar(new KeyValuePair<string, JToken>(name, value))[0];
        }

        public string GetFreeFunctionName(string baseName)
        {
            if (!Pfa.ContainsFunc(baseName))
                return baseName;
            int i = 0;
            while (Pfa.ContainsFunc(baseName + i))
                i++;
            return baseName + i;
        }

        public string DeclareCell(string name, JToken type, JToken init)
        {
            _host.CheckValue(name, nameof(name));
            _host.CheckValue(type, nameof(type));
            _host.CheckValue(init, nameof(init));

            var cellName = CreateNameCell(name);
            _host.Assert(!Pfa.ContainsCell(cellName));
            Pfa.AddCell(cellName, type, init);
            return cellName;
        }

        /// <summary>
        /// As a complimentary operation to <see cref="DeclareVar(KeyValuePair{string, JToken}[])"/>,
        /// this provides a mechanism for a transform to say, "hey, I am producing this column, but I
        /// am not writing any PFA for it, so if anyone asks for this column downstream don't say I
        /// have it."
        /// </summary>
        /// <param name="names">The names to hide</param>
        public void Hide(params string[] names)
        {
            _host.CheckValue(names, nameof(names));
            foreach (var name in names)
            {
                _host.CheckParam(name != null, nameof(names), "A value was null");
                if (_nameToVarName.ContainsKey(name))
                    _unavailable.Add(name);
            }
        }

        /// <summary>
        /// Given an <see cref="IDataView"/> column name, return the string for referencing the corresponding
        /// token in the PFA, or <c>null</c> if such a thing does not exist.
        /// </summary>
        public string TokenOrNullForName(string name)
        {
            _host.CheckValue(name, nameof(name));
            if (_unavailable.Contains(name))
                return null;
            _nameToVarName.TryGetValue(name, out name);
            return name;
        }

        /// <summary>
        /// Given an <see cref="IDataView"/> column name, return whether in the PFA being built up
        /// whether the corresponding PFA variable is still the input. This will return <c>false</c>
        /// also in the event that the column is hidden, or simply not present.
        /// </summary>
        public bool IsInput(string name)
        {
            _host.CheckValue(name, nameof(name));
            return !_unavailable.Contains(name) && _nameToVarName.TryGetValue(name, out name) && (name == "input" || name.StartsWith("input."));
        }
    }
}
