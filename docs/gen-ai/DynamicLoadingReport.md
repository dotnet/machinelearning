## Conclusion

- The main bottleneck of auto inference(dynamic loading) is the overhead of CPU-GPU data transfer.
- The larger the layer size, the more acceleration we can get from GPU. So we should try to put larger layers on GPU.

## Hardware: i9-14900k, 64GB memory, rtx 4090
### Sequential Layer

| Device | Num of Layers | Layer Size | Model Size | Num of Layers on GPU | Num of Layers on CPU | Average Inference (ms) | Acceleration | % of Layer in GPU |
|--------|----------------|------------|------------|-----------------------|-----------------------|------------------------|--------------|-------------------|
| CPU    | 512            | 4MB        | 2GB        | -                     | -                     | 939.8                  | 1.0          | 0%                |
| Auto   | 512            | 4MB        | 2GB        | 0                     | 512                   | 490                    | 1.9          | 0%                |
| Auto   | 512            | 4MB        | 2GB        | 253                   | 259                   | 272                    | 3.5          | 49.4%             |
| Auto   | 512            | 4MB        | 2GB        | 512                   | 0                     | 32                     | 29.4         | 100%              |
| GPU    | 512            | 4MB        | 2GB        | -                     | -                     | 32.4                   | 29.0         | 100%              |

### Sequential Layer, Deeper Model

| Device | Num of Layers | Layer Size | Model Size | Num of Layers on GPU | Num of Layers on CPU | Average Inference (ms) | Acceleration | % of Layer in GPU |
|--------|----------------|------------|------------|-----------------------|-----------------------|------------------------|--------------|-------------------|
| CPU    | 1024           | 4MB        | 4GB        | -                     | -                     | 1839.8                 | 1.0          | 0%                |
| Auto   | 1024           | 4MB        | 4GB        | 0                     | 1024                  | 954                    | 1.9          | 0%                |
| Auto   | 1024           | 4MB        | 4GB        | 252                   | 772                   | 787                    | 2.3          | 24.6%             |
| Auto   | 1024           | 4MB        | 4GB        | 508                   | 516                   | 530                    | 3.5          | 49.6%             |
| Auto   | 1024           | 4MB        | 4GB        | 764                   | 260                   | 312.5                  | 5.9          | 74.6%             |
| Auto   | 1024           | 4MB        | 4GB        | 1020                  | 4                     | 69.7                   | 26.9         | 99.6%             |
| GPU    | 1024           | 4MB        | 4GB        | -                     | -                     | 65.9                   | 27.9         | 100%              |

### Sequential Layer, Larger Layer (16MB)

| Device | Num of Layers | Layer Size | Model Size | Num of Layers on GPU | Num of Layers on CPU | Average Inference (ms) | Acceleration | % of Layer in GPU |
|--------|----------------|------------|------------|-----------------------|-----------------------|------------------------|--------------|-------------------|
| CPU    | 256            | 16MB       | 4GB        | -                     | -                     | 864                    | 1.0          | 0%                |
| Auto   | 256            | 16MB       | 4GB        | 0                     | 256                   | 844.7                  | 1.02         | 0%                |
| Auto   | 256            | 16MB       | 4GB        | 60                    | 196                   | 669.9                  | 1.3          | 23.4%             |
| Auto   | 256            | 16MB       | 4GB        | 124                   | 132                   | 494.2                  | 1.7          | 48.4%             |
| Auto   | 256            | 16MB       | 4GB        | 188                   | 68                    | 372.7                  | 2.3          | 73.4%             |
| Auto   | 256            | 16MB       | 4GB        | 252                   | 4                     | 152.5                  | 5.7          | 98.4%             |
| GPU    | 256            | 16MB       | 4GB        | -                     | -                     | 119                    | 7.3          | 100%              |

### Sequential Layer, Even Larger Layer (64MB)

| Device | Num of Layers | Layer Size | Model Size | Num of Layers on GPU | Num of Layers on CPU | Average Inference (ms) | Acceleration | % of Layer in GPU |
|--------|----------------|------------|------------|-----------------------|-----------------------|------------------------|--------------|-------------------|
| CPU    | 64             | 64MB       | 4GB        | -                     | -                     | 8501                   | 1.0          | 0%                |
| Auto   | 64             | 64MB       | 4GB        | 0                     | 64                    | 898                    | 9.5          | 0%                |
| Auto   | 64             | 64MB       | 4GB        | 12                    | 52                    | 755.2                  | 11.3         | 18.8%             |
| Auto   | 64             | 64MB       | 4GB        | 28                    | 36                    | 598                    | 14.2         | 43.8%             |
| Auto   | 64             | 64MB       | 4GB        | 44                    | 20                    | 419.7                  | 20.2         | 68.8%             |
| Auto   | 64             | 64MB       | 4GB        | 60                    | 4                     | 263.7                  | 32.3         | 93.8%             |
| Auto   | 64             | 64MB       | 4GB        | 64                    | 0                     | 70.54                     | 121         | 100%              |
| GPU    | 64             | 64MB       | 4GB        | -                     | -                     | 69.8                   | 121.7        | 100%              |

## Hardware: Xeon W-2133, 32GB memory, gtx 1066
| Device | Num of Layers | Layer Size | Model Size | Num of Layers on GPU | Num of Layers on CPU | Average Inference (ms) | Acceleration | % of Layer in GPU |
|--------|----------------|------------|------------|-----------------------|-----------------------|------------------------|--------------|-------------------|
| CPU    | 64             | 64MB       | 4GB        | -                     | -                     | 17419                   | 1.0          | 0%                |
| Auto   | 64             | 64MB       | 4GB        | 0                     | 64                    | 3783.4                    | 4.6          | 0%                |
| Auto   | 64             | 64MB       | 4GB        | 12                    | 52                    | 3415                  | 5.1         | 18.8%             |
| Auto   | 64             | 64MB       | 4GB        | 28                    | 36                    | 3004                    | 5.79         | 43.8%             |
| Auto   | 64             | 64MB       | 4GB        | 44                    | 20                    | 2536                  | 6.86         | 68.8%             |
| Auto   | 64             | 64MB       | 4GB        | 60                    | 4                     | 2101                  | 8.29         | 93.8%             |
| Auto   | 64             | 64MB       | 4GB        | 64                    | 0                     | 1163                     | 14.97         | 100%              |
| GPU    | 64             | 64MB       | 4GB        | -                     | -                     | 1213                   | 14.3        | 100%              |
