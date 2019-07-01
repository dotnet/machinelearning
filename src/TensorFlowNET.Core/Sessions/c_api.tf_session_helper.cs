using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public partial class c_api
    {
        public static string[] TF_OperationOutputConsumers_wrapper(TF_Output oper_out)
        {
            int num_consumers = TF_OperationOutputNumConsumers(oper_out);
            int size = Marshal.SizeOf<TF_Input>();
            var handle = Marshal.AllocHGlobal(size * num_consumers);
            int num = TF_OperationOutputConsumers(oper_out, handle, num_consumers);
            var consumers = new string[num_consumers];
            for (int i = 0; i < num; i++)
            {
                TF_Input input = Marshal.PtrToStructure<TF_Input>(handle + i * size);
                consumers[i] = Marshal.PtrToStringAnsi(TF_OperationName(input.oper));
            }

            return consumers;
        }
    }
}
