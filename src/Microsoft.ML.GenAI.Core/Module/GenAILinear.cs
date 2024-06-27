using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI;
internal class GenAILinear : nn.Module<Tensor, Tensor>
{
#pragma warning disable MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly Tensor weight;
    private readonly Tensor? bias;
#pragma warning restore MSML_PrivateFieldName // Private field name not in: _camelCase format
    private readonly int _inFeatures;
    private readonly int _outFeatures;

    public GenAILinear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32, string? device = null)
        : base(nameof(GenAILinear))
    {
        this._inFeatures = inFeatures;
        this._outFeatures = outFeatures;
        device ??= torch.get_default_device().ToString();
        this.weight = torch.randn(outFeatures, inFeatures, dtype: dtype, device: device);

        if (hasBias)
        {
            this.bias = torch.randn(outFeatures, dtype: dtype, device: device);
        }

        this.RegisterComponents();
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        using var dispose = torch.NewDisposeScope();

        // use float32
        var input2 = input.to_type(ScalarType.Float32);
        var weight2 = this.weight.to_type(ScalarType.Float32);
        var result = torch.matmul(input2, weight2.t());

        if (this.bias is not null)
        {
            result = result + this.bias.to_type(ScalarType.Float32);
        }

        return result.to_type(input.dtype).MoveToOuterDisposeScope();
    }
}
