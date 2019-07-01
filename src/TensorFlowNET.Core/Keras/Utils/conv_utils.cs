using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Utils
{
    public class conv_utils
    {
        public static string convert_data_format(string data_format, int ndim)
        {
            if (data_format == "channels_last")
                if (ndim == 3)
                    return "NWC";
                else if (ndim == 4)
                    return "NHWC";
                else if (ndim == 5)
                    return "NDHWC";
                else
                    throw new ValueError($"Input rank not supported: {ndim}");
            else if (data_format == "channels_first")
                if (ndim == 3)
                    return "NCW";
                else if (ndim == 4)
                    return "NCHW";
                else if (ndim == 5)
                    return "NCDHW";
                else
                    throw new ValueError($"Input rank not supported: {ndim}");
            else
                throw new ValueError($"Invalid data_format: {data_format}");
        }

        public static int[] normalize_tuple(int[] value, int n, string name)
        {
            return value;
        }

        public static string normalize_padding(string value)
        {
            return value.ToLower();
        }

        public static string normalize_data_format(string value)
        {
            return value.ToLower();
        }
    }
}
