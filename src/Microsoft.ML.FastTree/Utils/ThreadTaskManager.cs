// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.FastTree.Internal
{
    internal static class ThreadTaskManager
    {
        private static object _lockObject = new object();

        // REVIEW: Should this bother with number of threads? What should it do?
        public static int NumThreads { get; private set; }

        public static void Initialize(int numThreads)
        {
            lock (_lockObject)
            {
                if (NumThreads == 0)
                {
                    Contracts.Assert(numThreads > 0);
                    Contracts.Assert(NumThreads == 0);
                    NumThreads = numThreads;
                    BlockingThreadPool.Initialize(numThreads);
                }
            }
        }

        private class ThreadTask : IThreadTask
        {
            private readonly int _num;
            private readonly Action[] _actions;

            public ThreadTask(int num, IEnumerable<Action> actions)
            {
                _actions = actions.ToArray();
                _num = num;
            }

            public void RunTask()
            {
                // Special case one thread. We treat all other values the same - let the Task
                // library determine the best plan.
                if (_num == 1)
                {
                    for (int i = 0; i < _actions.Length; i++)
                    {
                        // REVIEW: Should this simply invoke the action on this thread?
                        var task = Task.Run(_actions[i]);
                        task.Wait();
                    }
                }
                else
                    Parallel.Invoke(new ParallelOptions() { MaxDegreeOfParallelism = _num }, _actions);
            }
        }

        /// <summary>
        /// Makes a new task using the subtasks
        /// </summary>
        /// <param name="subTasks">subtasks composing the task</param>
        /// <returns>An IThreadTask to run the tasks</returns>
        public static IThreadTask MakeTask(IEnumerable<Action> subTasks)
        {
            return new ThreadTask(NumThreads, subTasks);
        }

        /// <summary>
        /// Makes a new task from the supplied action that takes an integer argument, from 0...max
        /// </summary>
        /// <param name="subTaskAction">Action to run</param>
        /// <param name="maxArgument">The max range of the argument</param>
        /// <returns>A task that runs the action using each value of the argument from 0...max</returns>
        public static IThreadTask MakeTask(Action<int> subTaskAction, int maxArgument)
        {
            IEnumerable<Action> subTasks =
                Enumerable.Range(0, maxArgument)
                .Select<int, Action>(arg => (() => subTaskAction(arg)));
            return MakeTask(subTasks);
        }
    }

    /// <summary>
    /// Interface for a decomposable task that runs on many threads
    /// </summary>
    public interface IThreadTask
    {
        void RunTask();
    }
}
