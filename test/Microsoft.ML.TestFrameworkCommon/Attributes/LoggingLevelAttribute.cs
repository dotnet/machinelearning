using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.TestFrameworkCommon.Attributes
{
    public sealed class LogMessageKind : Attribute
    {
        public ChannelMessageKind  MessageKind { get; }
        public LogMessageKind(ChannelMessageKind messageKind)
        {
            MessageKind = messageKind;
        }
    }
}
