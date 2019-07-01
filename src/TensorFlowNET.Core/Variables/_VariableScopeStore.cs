using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class _VariableScopeStore
    {
        public VariableScope current_scope { get; set; }
        public Dictionary<string, int> variable_scopes_count;

        public _VariableScopeStore()
        {
            current_scope = new VariableScope(false);
            variable_scopes_count = new Dictionary<string, int>();
        }

        public void open_variable_scope(string scope_name)
        {
            if (variable_scopes_count.ContainsKey(scope_name))
                variable_scopes_count[scope_name] += 1;
            else
                variable_scopes_count[scope_name] = 1;
        }

        public void close_variable_subscopes(string scope_name)
        {
            foreach (var k in variable_scopes_count.Keys)
                if (scope_name == null || k.StartsWith(scope_name + "/"))
                    variable_scopes_count[k] = 0;
        }

        public int variable_scope_count(string scope_name)
        {
            if (variable_scopes_count.ContainsKey(scope_name))
                return variable_scopes_count[scope_name];
            else
                return 0;
        }
    }
}
