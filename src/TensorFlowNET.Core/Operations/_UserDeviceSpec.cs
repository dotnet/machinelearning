using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class StringOrFunction
    {
        private object variable;

        private StringOrFunction(object val)
        {
            variable = val;
        }

        public static implicit operator StringOrFunction(string val)
        {
            return new StringOrFunction(val);
        }

        public static implicit operator StringOrFunction(Function val)
        {
            return new StringOrFunction(val);
        }

        public bool IsFunction
        {
            get
            {
                return variable is FunctionDef;
            }
        }

        public override string ToString()
        {
            if (variable == null)
                return "";

            if(!IsFunction)
            {
                return variable.ToString();
            }

            return ((FunctionDef)variable).ToString();
        }
    }

    public class _UserDeviceSpec
    {
        private StringOrFunction _device_name_or_function;
        private string display_name;
        private FunctionDef function;
        private string raw_string;

        public _UserDeviceSpec(StringOrFunction device_name_or_function)
        {
            
            _device_name_or_function = device_name_or_function;
            display_name = device_name_or_function.ToString();
        }
    }
}
