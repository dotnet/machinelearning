using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. 
    /// This leads to a low-level programming model in which you first define the dataflow graph, 
    /// then create a TensorFlow session to run parts of the graph across a set of local and remote devices.
    /// https://www.tensorflow.org/guide/graphs
    /// </summary>
    /*
        A TensorFlow computation, represented as a dataflow graph.

        A `Graph` contains a set of
        `tf.Operation` objects,
        which represent units of computation; and
        `tf.Tensor` objects, which represent
        the units of data that flow between operations.

        A default `Graph` is always registered, and accessible by calling
        `tf.get_default_graph`.
        To add an operation to the default graph, simply call one of the functions
        that defines a new `Operation`:

        ```python
        c = tf.constant(4.0)
        assert c.graph is tf.get_default_graph()
        ```

        Another typical usage involves the
        `tf.Graph.as_default`
        context manager, which overrides the current default graph for the
        lifetime of the context:

        ```python
        g = tf.Graph()
        with g.as_default():
        # Define operations and tensors in `g`.
        c = tf.constant(30.0)
        assert c.graph is g
        ```

        Important note: This class *is not* thread-safe for graph construction. All
        operations should be created from a single thread, or external
        synchronization must be provided. Unless otherwise specified, all methods
        are not thread-safe.

        A `Graph` instance supports an arbitrary number of "collections"
        that are identified by name. For convenience when building a large
        graph, collections can store groups of related objects: for
        example, the `tf.Variable` uses a collection (named
        `tf.GraphKeys.GLOBAL_VARIABLES`) for
        all variables that are created during the construction of a graph. The caller
        may define additional collections by specifying a new name.     
     */
    public partial class Graph : IPython, IDisposable
    {
        private IntPtr _handle;
        private Dictionary<int, ITensorOrOperation> _nodes_by_id;
        public Dictionary<string, ITensorOrOperation> _nodes_by_name;
        private Dictionary<string, int> _names_in_use;
        public int _version;
        private int _next_id_counter;
        private List<Operation> _unfetchable_ops = new List<Operation>();
        private List<Tensor> _unfeedable_tensors = new List<Tensor>();

        public string _name_stack = "";
        private string _graph_key;
        public string graph_key => _graph_key;
        public string _last_loss_reduction;
        public bool _is_loss_scaled_by_optimizer { get; set; }
        public Status Status { get; }

        /// <summary>
        /// True if the graph is considered "finalized".  In that case no
        /// new operations can be added.
        /// </summary>
        private bool _finalized = false;

        /// <summary>
        /// Arbitrary collections of objects.
        /// </summary>
        private Dictionary<string, object> _collections = new Dictionary<string, object>();

        public bool building_function;
        
        public Graph()
        {
            _handle = c_api.TF_NewGraph();
            Status = new Status();
            _nodes_by_id = new Dictionary<int, ITensorOrOperation>();
            _nodes_by_name = new Dictionary<string, ITensorOrOperation>();
            _names_in_use = new Dictionary<string, int>();
            _graph_key = $"grap-key-{ops.uid()}/";
        }

        public Graph(IntPtr handle)
        {
            _handle = handle;
            Status = new Status();
            _nodes_by_id = new Dictionary<int, ITensorOrOperation>();
            _nodes_by_name = new Dictionary<string, ITensorOrOperation>();
            _names_in_use = new Dictionary<string, int>();
            _graph_key = $"grap-key-{ops.uid()}/";
        }

        public ITensorOrOperation as_graph_element(object obj, bool allow_tensor = true, bool allow_operation = true)
        {
            return _as_graph_element_locked(obj, allow_tensor, allow_operation);
        }

        /// <summary>
        /// Returns a context manager that makes this `Graph` the default graph.
        /// </summary>
        /// <returns></returns>
        public Graph as_default()
        {
            return ops.set_default_graph(this);
        }

        private Tensor _as_graph_element(object obj)
        {
            if (obj is RefVariable var)
                return var._as_graph_element();

            return null;
        }

        private ITensorOrOperation _as_graph_element_locked(object obj, bool allow_tensor = true, bool allow_operation = true)
        {
            string types_str = "";

            if (allow_tensor && allow_operation)
            {
                types_str = "Tensor or Operation";
            }
            else if (allow_tensor)
            {
                types_str = "Tensor";
            }
            else if (allow_operation)
            {
                types_str = "Operation";
            }

            var temp_obj = _as_graph_element(obj);
            if (temp_obj != null)
                obj = temp_obj;

            // If obj appears to be a name...
            if (obj is string name)
            {
                if (name.Contains(":") && allow_tensor)
                {
                    string op_name = name.Split(':')[0];
                    int out_n = int.Parse(name.Split(':')[1]);

                    if (_nodes_by_name.ContainsKey(op_name))
                        return _nodes_by_name[op_name].outputs[out_n];
                }
                else if (!name.Contains(":") & allow_operation)
                {
                    if (!_nodes_by_name.ContainsKey(name))
                        throw new KeyError($"The name {name} refers to an Operation not in the graph.");
                    return _nodes_by_name[name];
                }
                else if (!name.Contains(":") & !allow_operation)
                {
                    // Looks like an Operation name but can't be an Operation.
                    if (_nodes_by_name.ContainsKey(name))
                        // Yep, it's an Operation name
                        throw new ValueError($"The name {name} refers to an Operation, not a {types_str}.");
                    else
                        throw new ValueError(
                            $"The name {name} looks like an (invalid) Operation name, not a {types_str}" +
                            " Tensor names must be of the form \"<op_name>:<output_index>\".");
                }
            }

            if (obj is Tensor tensor && allow_tensor)
            {
                if (tensor.graph.Equals(this))
                {
                    return tensor;
                }
                else
                {
                    throw new Exception($"Tensor {obj} is not an element of this graph.");
                }
            }
            else if (obj is Operation op && allow_operation)
            {
                if (op.graph.Equals(this))
                {
                    return op;
                }
                else
                {
                    throw new Exception($"Operation {obj} is not an element of this graph.");
                }
            }

            throw new Exception($"Can not convert a {obj.GetType().Name} into a {types_str}.");
        }

        public void add_to_collection<T>(string name, T value)
        {
            _check_not_finalized();
            if (_collections.ContainsKey(name))
                (_collections[name] as List<T>).Add(value);
            else
                _collections[name] = new List<T> { value };
        }

        public void add_to_collections<T>(List<string> names, T value)
        {
            foreach (string name in names)
                add_to_collection(name, value);
        }

        private void _check_not_finalized()
        {
            if (_finalized)
                throw new RuntimeError("Graph is finalized and cannot be modified.");
        }

        public unsafe Operation create_op(string op_type, Tensor[] inputs, TF_DataType[] dtypes,
            TF_DataType[] input_types = null, string name = null,
            Dictionary<string, AttrValue> attrs = null, OpDef op_def = null)
        {
            if (inputs == null)
                inputs = new Tensor[0];

            foreach ((int idx, Tensor a) in Python.enumerate(inputs))
            {

            }

            if (String.IsNullOrEmpty(name))
                name = op_type;
            // If a names ends with a '/' it is a "name scope" and we use it as-is,
            // after removing the trailing '/'.
            name = name.EndsWith("/") ? ops._name_from_scope_name(name) : unique_name(name);
            var node_def = ops._NodeDef(op_type, name, device: "", attrs: attrs);

            var input_ops = inputs.Select(x => x.op).ToArray();
            var control_inputs = _control_dependencies_for_inputs(input_ops);

            var op = new Operation(node_def,
                this,
                inputs: inputs,
                output_types: dtypes,
                control_inputs: control_inputs,
                input_types: input_types,
                original_op: null,
                op_def: op_def);

            _create_op_helper(op, true);

            /*Console.Write($"create_op: {op_type} '{node_def.Name}'");
            Console.Write($", inputs: {(inputs.Length == 0 ? "empty" : String.Join(", ", inputs.Select(x => x.name)))}");
            Console.Write($", control_inputs: {(control_inputs.Length == 0 ? "empty" : String.Join(", ", control_inputs.Select(x => x.name)))}");
            Console.Write($", outputs: {(op.outputs.Length == 0 ? "empty" : String.Join(", ", op.outputs.Select(x => x.name)))}");
            Console.WriteLine();*/

            return op;
        }

        private void _create_op_helper(Operation op, bool compute_device = true)
        {
            _record_op_seen_by_control_dependencies(op);
        }

        public void _add_op(Operation op)
        {
            op._id_value = _next_id();
            _nodes_by_id[op._id] = op;
            _nodes_by_name[op.name] = op;
            _version = Math.Max(_version, op._id);
        }

        public int _next_id()
        {
            return ++_next_id_counter;
        }

        public bool is_fetchable<T>(T tensor_or_op)
        {
            if (tensor_or_op is Tensor tensor)
            {
                return !_unfetchable_ops.Contains(tensor); ;
            }
            else if (tensor_or_op is Operation op)
            {
                return !_unfetchable_ops.Contains(op);
            }

            return false;
        }

        public string get_name_scope()
        {
            return _name_stack;
        }

        public string name_scope(string name)
        {
            string new_stack = "";

            if (string.IsNullOrEmpty(name))
                new_stack = "";
            else if (name.EndsWith("/"))
                new_stack = ops._name_from_scope_name(name);
            else
                new_stack = unique_name(name);

            _name_stack = new_stack;

            return String.IsNullOrEmpty(new_stack) ? "" : new_stack + "/";
        }

        /// <summary>
        /// Return a unique operation name for `name`.
        /// 
        /// Note: You rarely need to call `unique_name()` directly.Most of
        /// the time you just need to create `with g.name_scope()` blocks to
        /// generate structured names.
        /// 
        /// `unique_name` is used to generate structured names, separated by
        /// `"/"`, to help identify operations when debugging a graph.
        /// Operation names are displayed in error messages reported by the
        /// TensorFlow runtime, and in various visualization tools such as
        /// TensorBoard.
        /// 
        /// If `mark_as_used` is set to `True`, which is the default, a new
        /// unique name is created and marked as in use.If it's set to `False`,
        /// the unique name is returned without actually being marked as used.
        /// This is useful when the caller simply wants to know what the name
        /// to be created will be.
        /// </summary>
        /// <param name="name">The name for an operation.</param>
        /// <param name="mark_as_used"> Whether to mark this name as being used.</param>
        /// <returns>A string to be passed to `create_op()` that will be used
        /// to name the operation being created.</returns>
        public string unique_name(string name, bool mark_as_used = true)
        {
            if (!String.IsNullOrEmpty(_name_stack))
                name = _name_stack + "/" + name;
            // For the sake of checking for names in use, we treat names as case
            // insensitive (e.g. foo = Foo).
            var name_key = name.ToLower();
            int i = 0;
            if (_names_in_use.ContainsKey(name_key))
                i = _names_in_use[name_key];
            // Increment the number for "name_key".
            if (mark_as_used)
                _names_in_use[name_key] = i + 1;
            if (i > 0)
            {
                // Make sure the composed name key is not already used.
                var base_name_key = name_key;
                while (_names_in_use.ContainsKey(name_key))
                {
                    name_key = $"{base_name_key}_{i}";
                    i += 1;
                }
                // Mark the composed name_key as used in case someone wants
                // to call unique_name("name_1").
                if (mark_as_used)
                    _names_in_use[name_key] = 1;

                // Return the new name with the original capitalization of the given name.
                name = $"{name}_{i-1}";
            }
            return name;
        }

        public TF_Output[] ReturnOutputs(IntPtr results)
        {
            IntPtr return_output_handle = IntPtr.Zero;
            int num_return_outputs = 0;
            c_api.TF_ImportGraphDefResultsReturnOutputs(results, ref num_return_outputs, ref return_output_handle);
            TF_Output[] return_outputs = new TF_Output[num_return_outputs];
            for (int i = 0; i < num_return_outputs; i++)
            {
                var handle = return_output_handle + (Marshal.SizeOf<TF_Output>() * i);
                return_outputs[i] = Marshal.PtrToStructure<TF_Output>(handle);
            }

            return return_outputs;
        }

        public unsafe Operation[] ReturnOperations(IntPtr results)
        {
            TF_Operation return_oper_handle = new TF_Operation();
            int num_return_opers = 0;
            c_api.TF_ImportGraphDefResultsReturnOperations(results, ref num_return_opers, ref return_oper_handle);
            Operation[] return_opers = new Operation[num_return_opers];
            for (int i = 0; i < num_return_opers; i++)
            {
                var handle = return_oper_handle.node + Marshal.SizeOf<TF_Operation>() * i;
                return_opers[i] = new Operation(*(IntPtr*)handle);
            }
            
            return return_opers;
        }

        public Operation OperationByName(string operName)
        {
            return c_api.TF_GraphOperationByName(_handle, operName);
        }

        public ITensorOrOperation[] get_operations()
        {
            return _nodes_by_name.Values.Select(x => x).ToArray();
        }

        public string[] get_all_collection_keys()
        {
            return _collections.Keys.Where(x => !x.StartsWith("__")).ToArray();
        }

        public object get_collection(string name, string scope = null)
        {
            return _collections.ContainsKey(name) ? _collections[name] : null;
        }

        public List<T> get_collection<T>(string name, string scope = null)
        {
            return _collections.ContainsKey(name) ? _collections[name] as List<T> : new List<T>();
        }

        public object get_collection_ref(string name)
        {
            if (!_collections.ContainsKey(name))
                _collections[name] = new List<object>();
            return _collections[name];
        }

        public void prevent_feeding(Tensor tensor)
        {
            _unfeedable_tensors.Add(tensor);
        }

        public void prevent_fetching(Operation op)
        {
            _unfetchable_ops.Add(op);
        }

        public void Dispose()
        {
            // c_api.TF_DeleteGraph(_handle);
        }

        /// <summary>
        /// Returns the <see cref="Tensor"/> with the given <paramref name="name"/>.
        /// This method may be called concurrently from multiple threads.
        /// </summary>
        /// <param name="name">The name of the `Tensor` to return.</param>
        /// <exception cref="KeyError">If <paramref name="name"/> does not correspond to a tensor in this graph.</exception>
        /// <returns>The `Tensor` with the given <paramref name="name"/>.</returns>
        public Tensor get_tensor_by_name(string name)
        {
            return (Tensor)this.as_graph_element(name, allow_tensor: true, allow_operation: false);
        }

        public void __enter__()
        {
        }

        public void __exit__()
        {

        }

        public static implicit operator IntPtr(Graph graph)
        {
            return graph._handle;
        }
    }
}
