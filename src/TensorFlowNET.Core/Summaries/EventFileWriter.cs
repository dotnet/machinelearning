using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Tensorflow.Summaries
{
    /// <summary>
    /// Creates a `EventFileWriter` and an event file to write to.
    /// </summary>
    public class EventFileWriter
    {
        string _logdir;
        // Represents a first-in, first-out collection of objects.
        Queue<Event> _event_queue;
        EventsWriter _ev_writer;
        int _flush_secs;
        Event _sentinel_event;
        bool _closed;
        EventLoggerThread _worker;

        public EventFileWriter(string logdir, int max_queue = 10, int flush_secs= 120,
               string filename_suffix = null)
        {
            _logdir = logdir;
            Directory.CreateDirectory(_logdir);
            _event_queue = new Queue<Event>(max_queue);
            _ev_writer = new EventsWriter(Path.Combine(_logdir, "events"));
            _flush_secs = flush_secs;
            _sentinel_event = new Event();
            if (!string.IsNullOrEmpty(filename_suffix))
                // self._ev_writer.InitWithSuffix(compat.as_bytes(filename_suffix)))
                throw new NotImplementedException("EventFileWriter filename_suffix is not null");
            _closed = false;
            _worker = new EventLoggerThread(_event_queue, _ev_writer, _flush_secs, _sentinel_event);
            _worker.start();
        }
    }
}
