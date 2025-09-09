# Microsoft.ML.IsolationForest.Sample

This sample shows several realistic ways to use the Isolation Forest implementation in **Microsoft.ML.IsolationForest**—all with **synthetic data only** (no files needed):

1. **Operations** – detect web latency spikes (p95 latency + error rate)
2. **Finance** – detect unusual purchase amounts (amount + hour-of-day)
3. **IoT** – train once and perform rolling scoring on sensor batches
4. **Quality control** – direct model usage with SHAP-like score

## Build & Run

From the repository root:

```bash
dotnet run --project docs/samples/Microsoft.ML.IsolationForest.Sample/Microsoft.ML.IsolationForest.Sample.csproj --no-build
