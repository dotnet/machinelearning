using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace Phi.Module;

public interface IDynamicLoadModule
{
    public Action<nn.Module>? LoadToDeviceFunc { get; set; }

    public Action<nn.Module>? UnloadFromDeviceFunc { get; set; }
}
