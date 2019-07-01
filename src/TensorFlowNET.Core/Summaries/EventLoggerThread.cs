using Google.Protobuf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Tensorflow.Summaries
{
    /// <summary>
    /// Thread that logs events.
    /// </summary>
    public class EventLoggerThread
    {
        Queue<Event> _queue;
        bool daemon;
        EventsWriter _ev_writer;
        int _flush_secs;
        Event _sentinel_event;

        public EventLoggerThread(Queue<Event> queue, EventsWriter ev_writer, int flush_secs, Event sentinel_event)
        {
            daemon = true;
            _queue = queue;
            _ev_writer = ev_writer;
            _flush_secs = flush_secs;
            _sentinel_event = sentinel_event;
        }

        public void start() => run();

        public void run()
        {
            Task.Run(delegate
            {
                while (true)
                {
                    if(_queue.Count == 0)
                    {
                        Thread.Sleep(_flush_secs * 1000);
                        continue;
                    }

                    var @event = _queue.Dequeue();
                    _ev_writer._WriteSerializedEvent(@event.ToByteArray());
                    Thread.Sleep(1000);
                }
            });
        }
    }
}
