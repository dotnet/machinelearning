using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    public class BaseSession
    {
        protected Graph _graph;
        protected bool _opened;
        protected bool _closed;
        protected int _current_version;
        protected byte[] _target;
        protected IntPtr _session;
        public Status Status;
        public Graph graph => _graph;

        public BaseSession(string target = "", Graph g = null, SessionOptions opts = null)
        {
            _graph = g is null ? ops.get_default_graph() : g;

            _target = UTF8Encoding.UTF8.GetBytes(target);

            SessionOptions newOpts = null;
            if (opts == null)
                newOpts = c_api.TF_NewSessionOptions();

            Status = new Status();

            _session = c_api.TF_NewSession(_graph, opts ?? newOpts, Status);

            // dispose newOpts
            if (opts == null)
                c_api.TF_DeleteSessionOptions(newOpts);

            Status.Check(true);
        }

        public virtual NDArray run(object fetches, params FeedItem[] feed_dict)
        {
            return _run(fetches, feed_dict);
        }

        public virtual NDArray run(object fetches, Hashtable feed_dict = null)
        {
            var feed_items = feed_dict == null ? new FeedItem[0] :
                feed_dict.Keys.OfType<object>().Select(key => new FeedItem(key, feed_dict[key])).ToArray();
            return _run(fetches, feed_items);
        }

        private NDArray _run(object fetches, FeedItem[] feed_dict = null)
        {
            var feed_dict_tensor = new Dictionary<object, object>();
            var feed_map = new Dictionary<object, object>();

            Func<FeedItem, IEnumerable<(object, object)>> feed_fn = (item) =>
            {
                return new (object, object)[] { (item.Key, item.Value) };
            };

            // Validate and process feed_dict.
            if (feed_dict != null)
            {
                foreach (var feed in feed_dict)
                {
                    foreach (var (subfeed, subfeed_val) in feed_fn(feed))
                    {
                        var subfeed_t = _graph.as_graph_element(subfeed, allow_tensor: true, allow_operation: false);
                        var subfeed_dtype = subfeed_t.dtype.as_numpy_datatype();

                        switch (subfeed_val)
                        {
                            case IntPtr val:
                                feed_dict_tensor[subfeed_t] = val;
                                break;
                            case NDArray val:
                                feed_dict_tensor[subfeed_t] = val;
                                break;
                            case float val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case double val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case short val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case int val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case long val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case long[] val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case int[] val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case string val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case byte[] val:
                                feed_dict_tensor[subfeed_t] = np.array(val);
                                break;
                            case char[] val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case bool val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case bool[] val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            case float[] val:
                                feed_dict_tensor[subfeed_t] = (NDArray)val;
                                break;
                            default:
                                Console.WriteLine($"can't handle data type of subfeed_val");
                                throw new NotImplementedException("_run subfeed");
                        }

                        feed_map[subfeed_t.name] = (subfeed_t, subfeed_val);
                    }
                }
            }

            // Create a fetch handler to take care of the structure of fetches.
            var fetch_handler = new _FetchHandler(_graph, fetches, feed_dict_tensor);

            // Run request and get response.
            // We need to keep the returned movers alive for the following _do_run().
            // These movers are no longer needed when _do_run() completes, and
            // are deleted when `movers` goes out of scope when this _run() ends.
            var _ = _update_with_movers();
            var final_fetches = fetch_handler.fetches();
            var final_targets = fetch_handler.targets();

            // We only want to really perform the run if fetches or targets are provided,
            // or if the call is a partial run that specifies feeds.
            var results = _do_run(final_targets.Select(x => (Operation)x).ToList(), final_fetches, feed_dict_tensor);

            return fetch_handler.build_results(this, results);
        }

        /// <summary>
        /// Runs a step based on the given fetches and feeds.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="target_list">A list of operations to be run, but not fetched.</param>
        /// <param name="fetch_list"></param>
        /// <param name="feed_dict"></param>
        /// <returns>
        /// A list of numpy ndarrays, corresponding to the elements of
        /// `fetch_list`.  If the ith element of `fetch_list` contains the
        /// name of an operation, the first Tensor output of that operation
        /// will be returned for that element.
        /// </returns>
        private NDArray[] _do_run(List<Operation> target_list, List<Tensor> fetch_list, Dictionary<object, object> feed_dict)
        {
            var feeds = feed_dict.Select(x =>
            {
                if (x.Key is Tensor tensor)
                {
                    switch (x.Value)
                    {
                        case IntPtr pointer:
                            return new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), pointer);
                        case Tensor t1:
                            return new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), t1);
                        case NDArray nd:
                            return new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(nd, tensor.dtype));
                        case int intVal:
                            return new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(intVal));
                        case float floatVal:
                            return new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(floatVal));
                        case double doubleVal:
                            return new KeyValuePair<TF_Output, Tensor>(tensor._as_tf_output(), new Tensor(doubleVal));
                        default:
                            throw new NotImplementedException("feed_dict data type");
                    }
                }
                throw new NotImplementedException("_do_run.feed_dict");
            }).ToArray();
            var fetches = fetch_list.Select(x => x._as_tf_output()).ToArray();
            var targets = target_list;

            return _call_tf_sessionrun(feeds, fetches, target_list);
        }

        private unsafe NDArray[] _call_tf_sessionrun(KeyValuePair<TF_Output, Tensor>[] feed_dict, TF_Output[] fetch_list, List<Operation> target_list)
        {
            // Ensure any changes to the graph are reflected in the runtime.
            _extend_graph();

            var status = new Status();

            var output_values = fetch_list.Select(x => IntPtr.Zero).ToArray();

            c_api.TF_SessionRun(_session,
                run_options: null,
                inputs: feed_dict.Select(f => f.Key).ToArray(),
                input_values: feed_dict.Select(f => (IntPtr)f.Value).ToArray(),
                ninputs: feed_dict.Length,
                outputs: fetch_list,
                output_values: output_values,
                noutputs: fetch_list.Length,
                target_opers: target_list.Select(f => (IntPtr)f).ToArray(),
                ntargets: target_list.Count,
                run_metadata: IntPtr.Zero,
                status: status);

            status.Check(true);

            var result = new NDArray[fetch_list.Length];

            for (int i = 0; i < fetch_list.Length; i++)
                result[i] = fetchValue(output_values[i]);

            for (int i = 0; i < feed_dict.Length; i++)
                feed_dict[i].Value.Dispose();

            return result;
        }

        public unsafe Tensor[] runTFSession(TF_Output[] inputs, Tensor[] inputValues, TF_Output[] outputs, Operation[] targetOpers = null, Buffer runMetadata = null, Buffer runOptions = null, Status status = null)
        {
            if (_session == IntPtr.Zero)
                throw new ObjectDisposedException("The object was disposed");
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));
            if (inputValues == null)
                throw new ArgumentNullException(nameof(inputValues));
            if (outputs == null)
                throw new ArgumentNullException(nameof(outputs));
            int iLen = inputs.Length;
            if (iLen != inputValues.Length)
                throw new ArgumentException("inputs and inputValues have different lengths", "inputs");
            int oLen = outputs.Length;

            // runOptions and runMetadata might be null
            if (status == null)
            {
                status = new Status();
            }

            // Create arrays for the unmanaged versions
            var ivals = new IntPtr[iLen];
            for (int i = 0; i < iLen; i++)
                ivals[i] = (IntPtr)inputValues[i];

            // I believe this might not be necessary, the output values in TF_SessionRun looks like a write-only result
            var ovals = new IntPtr[outputs.Length];
            IntPtr[] topers = null;
            int tLen = 0;
            if (targetOpers != null)
            {
                tLen = targetOpers.Length;
                topers = new IntPtr[tLen];
                for (int i = 0; i < tLen; i++)
                    topers[i] = (IntPtr)targetOpers[i];
            }

            unsafe
            {
                c_api.TF_SessionRun(_session, runOptions == null ? null : (TF_Buffer*)(IntPtr)runOptions, inputs, ivals, iLen, outputs, ovals, oLen, topers, tLen, runMetadata == null ? IntPtr.Zero : (IntPtr)runMetadata, (IntPtr)status);
            }
            status.Check(true);

            // Ensure that the input tensors remain rooted, so that the GC won't collect & run finalizers between
            // when they are copied to ivals and TF_SessionRun is called.
            GC.KeepAlive(inputValues);

            var result = new Tensor[oLen];
            for (int i = 0; i < oLen; i++)
            {
                result[i] = new Tensor(ovals[i]);
            }
            return result;
        }

        private unsafe NDArray fetchValue(IntPtr output)
        {
            var tensor = new Tensor(output);
            NDArray nd = null;
            Type type = tensor.dtype.as_numpy_datatype();
            var ndims = tensor.shape.Select(x => (int)x).ToArray();
            var offset = c_api.TF_TensorData(output);

            switch (tensor.dtype)
            {
                case TF_DataType.TF_BOOL:
                    var bools = new bool[tensor.size];
                    for (ulong i = 0; i < tensor.size; i++)
                        bools[i] = *(bool*)(offset + (int)(tensor.itemsize * i));
                    nd = np.array(bools).reshape(ndims);
                    break;
                case TF_DataType.TF_STRING:
                    var bytes = tensor.Data();
                    // wired, don't know why we have to start from offset 9.
                    // length in the begin
                    var str = UTF8Encoding.Default.GetString(bytes, 9, bytes[8]);
                    nd = np.array(str).reshape();
                    break;
                case TF_DataType.TF_UINT8:
                    var _bytes = new byte[tensor.size];
                    for (ulong i = 0; i < tensor.size; i++)
                        _bytes[i] = *(byte*)(offset + (int)(tensor.itemsize * i));
                    nd = np.array(_bytes).reshape(ndims);
                    break;
                case TF_DataType.TF_INT16:
                    var shorts = new short[tensor.size];
                    for (ulong i = 0; i < tensor.size; i++)
                        shorts[i] = *(short*)(offset + (int)(tensor.itemsize * i));
                    nd = np.array(shorts).reshape(ndims);
                    break;
                case TF_DataType.TF_INT32:
                    var ints = new int[tensor.size];
                    for (ulong i = 0; i < tensor.size; i++)
                        ints[i] = *(int*)(offset + (int)(tensor.itemsize * i));
                    nd = np.array(ints).reshape(ndims);
                    break;
                case TF_DataType.TF_INT64:
                    var longs = new long[tensor.size];
                    for (ulong i = 0; i < tensor.size; i++)
                        longs[i] = *(long*)(offset + (int)(tensor.itemsize * i));
                    nd = np.array(longs).reshape(ndims);
                    break;
                case TF_DataType.TF_FLOAT:
                    var floats = new float[tensor.size];
                    for (ulong i = 0; i < tensor.size; i++)
                        floats[i] = *(float*)(offset + (int)(tensor.itemsize * i));
                    nd = np.array(floats).reshape(ndims);
                    break;
                case TF_DataType.TF_DOUBLE:
                    var doubles = new double[tensor.size];
                    for (ulong i = 0; i < tensor.size; i++)
                        doubles[i] = *(double*)(offset + (int)(tensor.itemsize * i));
                    nd = np.array(doubles).reshape(ndims);
                    break;
                default:
                    throw new NotImplementedException("can't fetch output");
            }

            tensor.Dispose();

            return nd;
        }

        /// <summary>
        /// If a tensor handle that is fed to a device incompatible placeholder, 
        /// we move the tensor to the right device, generate a new tensor handle, 
        /// and update feed_dict to use the new handle.
        /// </summary>
        private List<object> _update_with_movers()
        {
            return new List<object> { };
        }

        private void _extend_graph()
        {

        }
    }
}
