using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public partial class Tensor
    {
        public static Tensor operator +(Tensor x, Tensor y) => BinaryOpWrapper("add", x, y);
        public static Tensor operator +(Tensor x, int y) => BinaryOpWrapper("add", x, y);

        public static Tensor operator -(Tensor t1) => gen_math_ops.neg(t1);

        public static Tensor operator -(Tensor x, Tensor y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(Tensor x, int y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(Tensor x, double y) => BinaryOpWrapper("sub", x, y);
        public static Tensor operator -(float x, Tensor y) => BinaryOpWrapper("sub", x, y);

        public static Tensor operator *(float x, Tensor y) => BinaryOpWrapper("mul", x, y);
        public static Tensor operator *(double x, Tensor y) => BinaryOpWrapper("mul", x, y);
        public static Tensor operator *(Tensor x, Tensor y) => BinaryOpWrapper("mul", x, y);
        public static Tensor operator *(Tensor x, int y) => BinaryOpWrapper("mul", x, y);

        public static Tensor operator /(Tensor x, Tensor y) => BinaryOpWrapper("truediv", x, y);
        public static Tensor operator /(Tensor x, float y) => BinaryOpWrapper("truediv", x, y);
        public static Tensor operator /(float x, Tensor y) => BinaryOpWrapper("truediv", x, y);
        public static Tensor operator /(Tensor x, double y) => BinaryOpWrapper("truediv", x, y);

        public static Tensor operator %(Tensor x, Tensor y) => BinaryOpWrapper("mod", x, y);

        public static Tensor operator >(Tensor x, int y) => gen_math_ops.greater(x, y);
        public static Tensor operator >=(Tensor x, Tensor y) => gen_math_ops.greater_equal(x, y);
        public static Tensor operator >(Tensor x, float y) => gen_math_ops.greater(x, y);
        public static Tensor operator >(Tensor x, double y) => gen_math_ops.greater(x, y);
        public static Tensor operator <(Tensor x, int y) => gen_math_ops.less(x, y);
        public static Tensor operator <=(Tensor x, Tensor y) => gen_math_ops.less_equal(x, y);
        public static Tensor operator <(Tensor x, float y) => gen_math_ops.less(x, y);
        public static Tensor operator <(Tensor x, double y) => gen_math_ops.less(x, y);

        private static Tensor BinaryOpWrapper<Tx, Ty>(string name, Tx x, Ty y)
        {
            TF_DataType dtype = TF_DataType.DtInvalid;
            if (x is Tensor tl)
                dtype = tl.dtype.as_base_dtype();
            if( y is Tensor tr)
                dtype = tr.dtype.as_base_dtype();
            
            var namescope = ops.name_scope(null, name, new { x, y });
            return with(namescope, scope =>
            {
                Tensor result = null;
                var x1 = ops.convert_to_tensor(x, dtype: dtype, name: "x");
                var y1 = ops.convert_to_tensor(y, dtype: dtype, name: "y");

                switch (name.ToLower())
                {
                    case "add":
                        result = gen_math_ops.add(x1, y1, name: scope);
                        break;
                    case "truediv":
                        result = gen_math_ops.real_div(x1, y1, name: scope);
                        break;
                    case "mul":
                        result = gen_math_ops.mul(x1, y1, name: scope);
                        break;
                    case "sub":
                        result = gen_math_ops.sub(x1, y1, name: scope);
                        break;
                    case "mod":
                        result = gen_math_ops.floor_mod(x1, y1, name: scope);
                        break;
                    default:
                        throw new NotImplementedException($"BinaryOpWrapper: {name} - {typeof(Tx).Name}, {typeof(Ty)}");
                }

                return result;
            });
            
        }
    }
}
