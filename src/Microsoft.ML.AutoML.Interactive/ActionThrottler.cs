// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.ML.AutoML
{
    internal class ActionThrottler
    {
        private readonly Action _action;
        private readonly TimeSpan _minDelay;

        private DateTime _nextUpdateTime = DateTime.MinValue;
        private int _updatePending = 0;

        /// <summary>
        /// This constructor initializes an ActionThrottler that ensures <paramref name="action"/> runs no more than once per <paramref name="minDelay"/>.
        /// </summary>
        /// <param name="action">The action to thorttle.</param>
        /// <param name="minDelay">Timespan to indicate the minimum delay between each time action is executed.</param>
        public ActionThrottler(Action action, TimeSpan minDelay)
        {
            _minDelay = minDelay;
            _action = action;
        }


        public async Task ExecuteAsync()
        {
            if (Interlocked.CompareExchange(ref _updatePending, 1, 0) == 0) // _updatePending is int initialized with 0
            {
                DateTime currentTime = DateTime.UtcNow;

                if (_nextUpdateTime > currentTime)
                {
                    await Task.Delay(_nextUpdateTime - currentTime);
                }
                _action();
                _nextUpdateTime = DateTime.UtcNow + _minDelay;
                _updatePending = 0;
            }
        }
    }
}
