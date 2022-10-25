// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.Runtime;

#nullable enable
namespace Microsoft.ML.AutoML
{
    internal interface IStopTrainingManager
    {
        bool IsStopTrainingRequested();

        void Update(TrialResult result);

        public event EventHandler OnStopTraining;
    }

    internal class CancellationTokenStopTrainingManager : IStopTrainingManager
    {
        private readonly CancellationToken _token;
        private readonly IChannel? _channel;
        public event EventHandler? OnStopTraining;

        public CancellationTokenStopTrainingManager(CancellationToken ct, IChannel? channel)
        {
            _token = ct;
            _channel = channel;
            ct.Register(() =>
            {
                _channel?.Info("cancel training because cancellation token is invoked...");
                OnStopTraining?.Invoke(this, EventArgs.Empty);
            });
        }

        public bool IsStopTrainingRequested()
        {
            if (_token.IsCancellationRequested)
            {
                return true;
            }

            return false;
        }

        public void Update(TrialResult result)
        {
            return;
        }
    }

    internal class TimeoutTrainingStopManager : IStopTrainingManager
    {
        private readonly CancellationTokenStopTrainingManager _cancellationTokenTrainingStopManager;
        private readonly CancellationTokenSource _cts;

        public event EventHandler? OnStopTraining;

        public TimeoutTrainingStopManager(TimeSpan timeoutInSeconds, IChannel? channel)
        {
            _cts = new CancellationTokenSource();
            _cts.CancelAfter(timeoutInSeconds);
            _cancellationTokenTrainingStopManager = new CancellationTokenStopTrainingManager(_cts.Token, channel);
            _cancellationTokenTrainingStopManager.OnStopTraining += (o, e) =>
            {
                OnStopTraining?.Invoke(this, e);
            };
        }

        public bool IsStopTrainingRequested()
        {
            return _cancellationTokenTrainingStopManager.IsStopTrainingRequested();
        }

        public void Update(TrialResult result)
        {
            return;
        }
    }

    internal class MaxModelStopManager : IStopTrainingManager
    {
        private readonly int _maxModel;
        private int _exploredModel = 0;
        public event EventHandler? OnStopTraining;

        public MaxModelStopManager(int maxModel, IChannel? channel)
        {
            _maxModel = maxModel;
        }

        public bool IsStopTrainingRequested()
        {
            return _exploredModel >= _maxModel;
        }

        public void Update(TrialResult result)
        {
            _exploredModel++;
            if (_exploredModel > _maxModel)
            {
                OnStopTraining?.Invoke(this, EventArgs.Empty);
            }
        }
    }

    /// <summary>
    /// stop training when any of child training stop manager is stopped.
    /// </summary>
    internal class AggregateTrainingStopManager : IStopTrainingManager
    {
        private readonly List<IStopTrainingManager> _managers;

        public event EventHandler? OnStopTraining;

        public AggregateTrainingStopManager(IChannel? channel, params IStopTrainingManager[] managers)
        {
            _managers = managers.ToList();
            foreach (var manager in _managers)
            {
                manager.OnStopTraining += (o, e) =>
                {
                    OnStopTraining?.Invoke(this, e);
                };
            }
        }

        public bool IsStopTrainingRequested()
        {
            return _managers.Any(m => m.IsStopTrainingRequested());
        }

        public void AddTrainingStopManager(IStopTrainingManager manager)
        {
            _managers.Add(manager);
            manager.OnStopTraining += (o, e) =>
            {
                OnStopTraining?.Invoke(this, e);
            };
        }

        public void Update(TrialResult result)
        {
            foreach (var manager in _managers)
            {
                manager.Update(result);
            }
        }
    }
}
