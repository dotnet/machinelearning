using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tensorflow.Keras.Utils
{
    public class generic_utils
    {
        public static string to_snake_case(string name)
        {
            return string.Concat(name.Select((x, i) =>
            {
                return i > 0 && char.IsUpper(x) && !Char.IsDigit(name[i - 1]) ?
                    "_" + x.ToString() :
                    x.ToString();
            })).ToLower();
        }
    }
}
