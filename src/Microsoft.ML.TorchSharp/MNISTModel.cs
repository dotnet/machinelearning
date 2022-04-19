// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using TorchSharp;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace Microsoft.ML.TorchSharp
{
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "<Pending>")]
    public class MNISTModel : Module
    {
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format. Need to match exactly with original model file.
        private readonly Module conv1 = Conv2d(1, 32, 3);
        private readonly Module conv2 = Conv2d(32, 64, 3);
        private readonly Module fc1 = Linear(9216, 128);
        private readonly Module fc2 = Linear(128, 10);

        // These don't have any parameters, so the only reason to instantiate
        // them is performance, since they will be used over and over.
        private readonly Module pool1 = MaxPool2d(kernelSize: new long[] { 2, 2 });

        private readonly Module relu1 = ReLU();
        private readonly Module relu2 = ReLU();
        private readonly Module relu3 = ReLU();

        private readonly Module dropout1 = Dropout(0.25);
        private readonly Module dropout2 = Dropout(0.5);

        private readonly Module flatten = Flatten();
        private readonly Module logsm = LogSoftmax(1);

#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format. Need to match exactly with original model file.


        public MNISTModel(string name, torch.Device device = null) : base(name)
        {
            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            var l11 = conv1.forward(input);
            var l12 = relu1.forward(l11);

            var l21 = conv2.forward(l12);
            var l22 = relu2.forward(l21);
            var l23 = pool1.forward(l22);
            var l24 = dropout1.forward(l23);

            var x = flatten.forward(l24);

            var l31 = fc1.forward(x);
            var l32 = relu3.forward(l31);
            var l33 = dropout2.forward(l32);

            var l41 = fc2.forward(l33);

            return logsm.forward(l41);
        }
    }
}
