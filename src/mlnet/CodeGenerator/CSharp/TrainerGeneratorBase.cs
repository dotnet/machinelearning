// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI.CodeGenerator.CSharp
{
    /// <summary>
    /// Supports generation of code for trainers (Binary,Multi,Regression)
    /// Ova is an exception though. Need to figure out how to tackle that.
    /// </summary>
    internal abstract class TrainerGeneratorBase : ITrainerGenerator
    {
        private PipelineNode node;
        private Dictionary<string, object> arguments = new Dictionary<string, object>();
        private bool hasAdvancedSettings = false;
        private string seperator = null;

        //abstract properties
        internal abstract string OptionsName { get; }
        internal abstract string MethodName { get; }
        internal abstract IDictionary<string, string> NamedParameters { get; }
        internal abstract string[] Usings { get; }

        /// <summary>
        /// Generates an instance of TrainerGenerator
        /// </summary>
        /// <param name="node"></param>
        protected TrainerGeneratorBase(PipelineNode node)
        {
            Initialize(node);
        }

        private void Initialize(PipelineNode node)
        {
            this.node = node;
            if (NamedParameters != null)
            {
                hasAdvancedSettings = node.Properties.Keys.Any(t => !NamedParameters.ContainsKey(t));
            }
            seperator = hasAdvancedSettings ? "=" : ":";
            if (!node.Properties.ContainsKey("LabelColumn"))
            {
                node.Properties.Add("LabelColumn", "Label");
            }
            node.Properties.Add("FeatureColumn", "Features");

            foreach (var kv in node.Properties)
            {
                object value = null;

                //For Nullable values.
                if (kv.Value == null)
                    continue;
                Type type = kv.Value.GetType();
                if (type == typeof(bool))
                {
                    //True to true
                    value = ((bool)kv.Value).ToString().ToLowerInvariant();
                }
                if (type == typeof(float))
                {
                    //0.0 to 0.0f
                    value = ((float)kv.Value).ToString() + "f";
                }

                if (type == typeof(int) || type == typeof(double) || type == typeof(long))
                {
                    value = (kv.Value).ToString();
                }

                if (type == typeof(string))
                {
                    var val = kv.Value.ToString();
                    if (val == "<Auto>")
                        continue; // This is temporary fix and needs to be fixed in AutoML SDK

                    // string to "string"
                    value = "\"" + val + "\"";
                }

                if (type == typeof(CustomProperty))
                {
                    value = kv.Value;
                }
                //more special cases to handle

                if (NamedParameters != null)
                {
                    arguments.Add(hasAdvancedSettings ? kv.Key : NamedParameters[kv.Key], value);
                }
                else
                {
                    arguments.Add(kv.Key, value);
                }

            }
        }

        internal static string BuildComplexParameter(string paramName, IDictionary<string, object> arguments, string seperator)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("new ");
            sb.Append(paramName);
            sb.Append("(){");
            sb.Append(AppendArguments(arguments, seperator));
            sb.Append("}");
            return sb.ToString();
        }

        internal static string AppendArguments(IDictionary<string, object> arguments, string seperator)
        {
            if (arguments.Count == 0)
                return string.Empty;

            StringBuilder sb = new StringBuilder();
            foreach (var kv in arguments)
            {
                sb.Append(kv.Key);
                sb.Append(seperator);
                if (kv.Value.GetType() == typeof(CustomProperty))
                    sb.Append(BuildComplexParameter(((CustomProperty)kv.Value).Name, ((CustomProperty)kv.Value).Properties, "="));
                else
                    sb.Append(kv.Value.ToString());
                sb.Append(",");
            }
            sb.Remove(sb.Length - 1, 1); //remove the last ,
            return sb.ToString();
        }

        public virtual string GenerateTrainer()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(MethodName);
            sb.Append("(");
            if (hasAdvancedSettings)
            {
                var paramString = BuildComplexParameter(OptionsName, arguments, "=");
                sb.Append(paramString);
            }
            else
            {
                sb.Append(AppendArguments(arguments, ":"));
            }
            sb.Append(")");
            return sb.ToString();
        }

        public virtual string[] GenerateUsings()
        {
            if (hasAdvancedSettings)
                return Usings;

            return null;
        }
    }
}
