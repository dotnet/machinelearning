using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Summaries
{
    /// <summary>
    /// Abstractly implements the SummaryWriter API.
    /// </summary>
    public abstract class SummaryToEventTransformer
    {
        public void add_summary(string summary, int global_step = 0)
        {
            var bytes = UTF8Encoding.Unicode.GetBytes(summary);
            // var summ = Tensorflow.Summary.Parser.ParseFrom(bytes);
        }
    }
}
