using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Tensorflow.Gradients;

namespace Tensorflow
{
    public partial class ops
    {
        static Dictionary<string, Func<Operation, Tensor[], Tensor[]>> gradientFunctions = null;

        /// <summary>
        /// Regiter new gradient function
        /// </summary>
        /// <param name="name">operation type</param>
        /// <param name="func">function delegate</param>
        public static void RegisterGradientFunction(string name, Func<Operation, Tensor[], Tensor[]> func)
        {
            if(gradientFunctions == null)
                gradientFunctions = new Dictionary<string, Func<Operation, Tensor[], Tensor[]>>();

            gradientFunctions[name] = func;
        }

        public static Func<Operation, Tensor[], Tensor[]> get_gradient_function(Operation op)
        {
            if (op.inputs == null) return null;

            if (gradientFunctions == null)
            {
                gradientFunctions = new Dictionary<string, Func<Operation, Tensor[], Tensor[]>>();

                var gradGroups = Assembly.GetExecutingAssembly()
                    .GetTypes()
                    .Where(x => x.GetCustomAttribute<RegisterGradient>() != null)
                    .ToArray();

                foreach (var g in gradGroups)
                {
                    var methods = g.GetMethods().Where(x => x.GetCustomAttribute<RegisterGradient>() != null)
                        .ToArray();

                    foreach (var m in methods)
                    {
                        RegisterGradientFunction(m.GetCustomAttribute<RegisterGradient>().Name,
                            (oper, out_grads) =>
                                 g.InvokeMember(m.Name,
                                    BindingFlags.InvokeMethod,
                                    null,
                                    null,
                                    args: new object[] { oper, out_grads }) as Tensor[]
                        );
                    }
                }
            }

            if (!gradientFunctions.ContainsKey(op.type))
                throw new NotImplementedException($"can't get graident function through get_gradient_function {op.type}");

            return gradientFunctions[op.type];
        }
    }
}
