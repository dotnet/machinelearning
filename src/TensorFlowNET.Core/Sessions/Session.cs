using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Session : BaseSession, IPython
    {
        public Session(string target = "", Graph g = null)
            : base(target, g, null)
        {

        }

        public Session(IntPtr handle)
            : base("", null, null)
        {
            _session = handle;
        }

        public Session(Graph g, SessionOptions opts = null, Status s = null)
            : base("", g, opts)
        {
            if (s == null)
                s = Status;
        }

        public Session as_default()
        {
            tf.defaultSession = this;
            return this;
        }

        public static Session LoadFromSavedModel(string path)
        {
            var graph = c_api.TF_NewGraph();
            var status = new Status();
            var opt = c_api.TF_NewSessionOptions();

            var buffer = new TF_Buffer();
            var sess = c_api.TF_LoadSessionFromSavedModel(opt, IntPtr.Zero, path, new string[0], 0, graph, ref buffer, status);

            //var bytes = new Buffer(buffer.data).Data;
            //var meta_graph = MetaGraphDef.Parser.ParseFrom(bytes);

            status.Check();

            new Graph(graph).as_default();

            return sess;
        }

        public static implicit operator IntPtr(Session session) => session._session;
        public static implicit operator Session(IntPtr handle) => new Session(handle);

        public void close()
        {
            Dispose();
        }

        public void Dispose()
        {
            c_api.TF_DeleteSession(_session, Status);
            Status.Dispose();
        }

        public void __enter__()
        {

        }

        public void __exit__()
        {

        }

        /// <summary>
        /// Gets a new runner, this provides a simpler API to prepare the inputs to run on a session
        /// </summary>
        /// <returns>The runner.</returns>
        /// <remarks>
        /// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
        /// to construct the parameters that will be passed to the TFSession.Run method.
        ///
        /// The Run method will return an array of TFTensor values, one for each invocation to the Fetch method.
        /// </remarks>
        public Runner GetRunner()
        {
            return new Runner(this._session);
        }
    }
}
