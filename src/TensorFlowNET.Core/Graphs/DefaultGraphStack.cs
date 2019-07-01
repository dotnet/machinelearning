using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow
{
    public class DefaultGraphStack
    {
        List<StackModel> stack = new List<StackModel>();

        public void set_controller(Graph @default)
        {
            if (!stack.Exists(x => x.Graph == @default))
                stack.Add(new StackModel { Graph = @default, IsDefault = true });

            foreach (var s in stack)
                s.IsDefault = s.Graph == @default;
        }

        public Graph get_controller()
        {
            if (stack.Count == 0)
                stack.Add(new StackModel { Graph = tf.Graph(), IsDefault = true });

            return stack.First(x => x.IsDefault).Graph;
        }

        public void reset()
        {
            stack.Clear();
        }
    }

    public class StackModel
    {
        public Graph Graph { get; set; }
        public bool IsDefault { get; set; }
    }
}
