using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.OpDef.Types;

namespace Tensorflow
{
    public class op_def_registry
    {
        private static Dictionary<string, OpDef> _registered_ops;

        public static Dictionary<string, OpDef> get_registered_ops()
        {
            if(_registered_ops == null)
            {
                _registered_ops = new Dictionary<string, OpDef>();
                var handle = c_api.TF_GetAllOpList();
                var buffer = new Buffer(handle);
                var op_list = OpList.Parser.ParseFrom(buffer);

                foreach (var op_def in op_list.Op)
                    _registered_ops[op_def.Name] = op_def;
            }

            return _registered_ops;
        }
    }
}
