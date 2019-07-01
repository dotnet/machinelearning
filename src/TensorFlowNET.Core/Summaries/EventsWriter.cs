using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow.Summaries
{
    public class EventsWriter
    {
        string _file_prefix;

        public EventsWriter(string file_prefix)
        {
            _file_prefix = file_prefix;
        }

        public void _WriteSerializedEvent(byte[] event_str)
        {
            File.WriteAllBytes(_file_prefix, event_str);
        }
    }
}
