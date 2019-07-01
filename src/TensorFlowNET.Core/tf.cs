using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Eager;
using static Tensorflow.ops;

namespace Tensorflow
{
    public static partial class tf
    {
        public static TF_DataType bytes = TF_DataType.TF_STRING;
        public static TF_DataType int16 = TF_DataType.TF_INT16;
        public static TF_DataType int32 = TF_DataType.TF_INT32;
        public static TF_DataType int64 = TF_DataType.TF_INT64;
        public static TF_DataType float16 = TF_DataType.TF_HALF;
        public static TF_DataType float32 = TF_DataType.TF_FLOAT;
        public static TF_DataType float64 = TF_DataType.TF_DOUBLE;
        public static TF_DataType @bool = TF_DataType.TF_BOOL;
        public static TF_DataType chars = TF_DataType.TF_STRING;

        public static Context context = new Context(new ContextOptions(), new Status());

        public static Session defaultSession;

        public static RefVariable Variable<T>(T data, 
            bool trainable = true,
            bool validate_shape = true,
            string name = null, 
            TF_DataType dtype = TF_DataType.DtInvalid)
        {
            return Tensorflow.variable_scope.default_variable_creator(data, 
                trainable: trainable,
                validate_shape: validate_shape,
                name: name,
                dtype: TF_DataType.DtInvalid);
        }

        public static unsafe Tensor placeholder(TF_DataType dtype, TensorShape shape = null, string name = null)
        {
            return gen_array_ops.placeholder(dtype, shape, name);
        }

        public static void enable_eager_execution()
        {
            // contex = new Context();
            context.default_execution_mode = Context.EAGER_MODE;
        }

        public static string VERSION => c_api.StringPiece(c_api.TF_Version());

        public static Session Session()
        {
            defaultSession = new Session();
            return defaultSession;
        }

        public static Session Session(Graph graph)
        {
            return new Session(graph);
        }
    }
}
