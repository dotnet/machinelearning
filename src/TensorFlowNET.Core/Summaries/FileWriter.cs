using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Summaries
{
    /// <summary>
    /// Writes `Summary` protocol buffers to event files.
    /// </summary>
    public class FileWriter : SummaryToEventTransformer
    {
        EventFileWriter event_writer;

        public FileWriter(string logdir, Graph graph, 
            int max_queue = 10, int flush_secs = 120, string filename_suffix = null, 
            Session session = null)
        {
            if(session == null)
            {
                event_writer = new EventFileWriter(logdir, max_queue, flush_secs, filename_suffix);
            }
        }
    }
}
