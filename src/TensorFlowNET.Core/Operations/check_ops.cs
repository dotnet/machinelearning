using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Python;

namespace Tensorflow
{
    public class check_ops
    {
        /// <summary>
        /// Assert the condition `x == y` holds element-wise.
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="name"></param>
        public static Operation assert_equal(object t1, object t2, object[] data = null, string message = null, string name = null)
        {
            if (message == null)
                message = "";

            return with(ops.name_scope(name, "assert_equal", new { t1, t2, data }), delegate
            {
                var x = ops.convert_to_tensor(t1, name: "x");
                var y = ops.convert_to_tensor(t2, name: "y");

                if (data == null)
                {
                    data = new object[]
                    {
                        message,
                        "Condition x == y did not hold element-wise:",
                        $"x (%s) = {x.name}",
                        x,
                        $"y (%s) = {y.name}",
                        y
                    };
                }

                var eq = gen_math_ops.equal(x, y);
                var condition = math_ops.reduce_all(eq);
                var x_static = tensor_util.constant_value(x);
                var y_static = tensor_util.constant_value(y);
                return control_flow_ops.Assert(condition, data);
            });
        }

        public static Operation assert_positive(Tensor x, object[] data = null, string message = null, string name = null)
        {
            if (message == null)
                message = "";

            return with(ops.name_scope(name, "assert_positive", new { x, data }), delegate
            {
                x = ops.convert_to_tensor(x, name: "x");
                if (data == null)
                {
                    name = x.name;
                    data = new object[]
                    {
                        message,
                        "Condition x > 0 did not hold element-wise:",
                        $"x (%s) = {name}",
                        x
                    };
                }
                var zero = ops.convert_to_tensor(0, dtype: x.dtype);
                return assert_less(zero, x, data: data);
            });
        }

        public static Operation assert_less(Tensor x, Tensor y, object[] data = null, string message = null, string name = null)
        {
            if (message == null)
                message = "";

            return with(ops.name_scope(name, "assert_less", new { x, y, data }), delegate
            {
                x = ops.convert_to_tensor(x, name: "x");
                y = ops.convert_to_tensor(y, name: "y");
                string x_name = x.name;
                string y_name = y.name;
                if (data == null)
                {
                    data = new object[]
                    {
                        message,
                        "Condition x < y did not hold element-wise:",
                        $"x (%s) = {x_name}",
                        $"y (%s) = {y_name}",
                        y
                    };
                }
                var condition = math_ops.reduce_all(gen_math_ops.less(x, y));
                return control_flow_ops.Assert(condition, data);
            });
        }
    }
}
