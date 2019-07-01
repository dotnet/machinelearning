using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Operations;

namespace Tensorflow
{
    public class control_flow_util
    {
        /// <summary>
        /// Return true if `op` is an Exit.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsLoopExit(Operation op)
        {
            return op.type == "Exit" || op.type == "RefExit";
        }

        /// <summary>
        /// Return true if `op` is a Switch.
        /// </summary>
        /// <param name="op"></param>
        /// <returns></returns>
        public static bool IsSwitch(Operation op)
        {
            return op.type == "Switch" || op.type == "RefSwitch";
        }

        /// <summary>
        /// Return the control flow context for the output of an op.
        /// </summary>
        public static ControlFlowContext GetOutputContext(Operation op)
        {
            var ctxt = op._get_control_flow_context();
            // Exit nodes usually have a control flow context, except in the case where the
            // exit node was imported via import_graph_def (in which case no nodes have
            // control flow contexts).
            if (ctxt != null && IsLoopExit(op))
                ctxt = ctxt.outer_context;
            return ctxt;
        }
    }
}
