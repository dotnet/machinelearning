using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    public class Runner
    {
        private List<TF_Output> inputs;
        private List<TF_Output> outputs;
        private List<Tensor> inputValues;
        private List<Operation> targets;
        private Session session;

        internal Runner(Session session)
        {
            inputs = new List<TF_Output>();
            outputs = new List<TF_Output>();
            inputValues = new List<Tensor>();
            targets = new List<Operation>();
            this.session = session;
            RunMetadata = null;
            RunOptions = null;
        }

        /// <summary>
        /// Adds an input to the session
        /// </summary>
        /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
        /// <param name="input">Incoming port.</param>
        /// <param name="value">Value to assing to the incoming port.</param>
        public Runner AddInput(TF_Output input, Tensor value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            inputs.Add(input);
            inputValues.Add(value);
            return this;
        }

        /// <summary>
        /// Adds an input to the session specified by name, with an optional index in the operation (separated by a colon).
        /// </summary>
        /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
        /// <param name="input">Incoming port, with an optional index separated by a colon.</param>
        /// <param name="value">Value to assing to the incoming port.</param>
        public Runner AddInput(string input, Tensor value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            inputs.Add(ParseOutput(input));
            inputValues.Add(value);
            return this;
        }

        /// <summary>
        /// Adds the specified operations as the ones to be retrieved.
        /// </summary>
        /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
        /// <param name="targets">One or more targets.</param>
        public Runner AddTarget(params Operation[] targets)
        {
            foreach (var t in targets)
                this.targets.Add(t);
            return this;
        }

        // Parses user strings that contain both the operation name and an index.
        private TF_Output ParseOutput(string operation)
        {
            var p = operation.IndexOf(':');
            if (p != -1 && p != operation.Length - 1)
            {
                var op = operation.Substring(0, p);
                if (int.TryParse(operation.Substring(p + 1), out var idx))
                {
                    return new TF_Output(session.graph.OperationByName(op), idx);
                }
            }
            return new TF_Output(session.graph.OperationByName(operation), 0);
        }

        /// <summary>
        /// Adds the specified operation names as the ones to be retrieved.
        /// </summary>
        /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
        /// <param name="targetNames">One or more target names.</param>
        public Runner AddTarget(params string[] targetNames)
        {
            foreach (var tn in targetNames)
                targets.Add(session.graph.OperationByName(tn));
            return this;
        }

        /// <summary>
        /// Makes the Run method return the index-th output of the tensor referenced by operation.
        /// </summary>
        /// <returns>The instance of runner, to allow chaining operations.</returns>
        /// <param name="operation">The name of the operation in the graph.</param>
        /// <param name="index">The index of the output in the operation.</param>
        public Runner Fetch(string operation, int index)
        {
            Operation op = session.graph.OperationByName(operation);
            TF_Output operationOutput = new TF_Output(op, index);
            outputs.Add(operationOutput);
            return this;
        }

        /// <summary>
        /// Makes the Run method return the output of the tensor referenced by operation, the operation string can contain the output index.
        /// </summary>
        /// <returns>The instance of runner, to allow chaining operations.</returns>
        /// <param name="operation">The name of the operation in the graph, which might be a simple name, or it might be name:index,
        /// where the index is the .</param>
        public Runner Fetch(string operation)
        {
            var op = ParseOutput(operation);
            outputs.Add(op);
            return this;
        }

        /// <summary>
        /// Makes the Run method return the output of the tensor referenced by output
        /// </summary>
        /// <returns>The instance of runner, to allow chaining operations.</returns>
        /// <param name="output">The output referencing a specified tensor.</param>
        public Runner Fetch(TF_Output output)
        {
            outputs.Add(output);
            return this;
        }

        /// <summary>
        /// Makes the Run method return the output of all the tensor referenced by outputs.
        /// </summary>
        /// <returns>The instance of runner, to allow chaining operations.</returns>
        /// <param name="outputs">The outputs referencing a specified tensor.</param>
        public Runner Fetch(params TF_Output[] outputs)
        {
            foreach (var output in outputs)
                this.outputs.Add(output);
            return this;
        }

        /// <summary>
        /// Makes the Run method return the output of all the tensor referenced by outputs.
        /// </summary>
        /// <returns>The instance of runner, to allow chaining operations.</returns>
        /// <param name="outputs">The output sreferencing a specified tensor.</param>
        public Runner Fetch(params string[] outputs)
        {
            foreach (var output in outputs)
                this.outputs.Add(ParseOutput(output));
            return this;
        }

        /// <summary>
        /// Protocol buffer encoded block containing the metadata passed to the <see cref="M:TensorFlow.TFSession.Run"/> method.
        /// </summary>
        public Buffer RunMetadata;

        /// <summary>
        /// Protocol buffer encoded block containing the run options passed to the <see cref="M:TensorFlow.TFSession.Run"/> method.
        /// </summary>
        public Buffer RunOptions;

        /// <summary>
        ///  Execute the graph fragments necessary to compute all requested fetches.
        /// </summary>
        /// <returns>One TFTensor for each call to Fetch that you made, in the order that you made them.</returns>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        public Tensor[] Run(Status status = null)
        {
            return session.Run(inputs.ToArray(), inputValues.ToArray(), outputs.ToArray(), targets.ToArray(), RunMetadata, RunOptions, status);
        }

        /// <summary>
        /// Run the specified operation, by adding it implicity to the output, single return value
        /// </summary>
        /// <param name="operation">The output of the operation.</param>
        /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
        /// <remarks>
        /// This method is a convenience method, and when you call it, it will clear any
        /// calls that you might have done to Fetch() and use the specified operation to Fetch
        /// instead.
        /// </remarks>
        public Tensor Run(TF_Output operation, Status status = null)
        {
            outputs.Clear();
            Fetch(operation);
            return Run(status)[0];
        }

    }
}
