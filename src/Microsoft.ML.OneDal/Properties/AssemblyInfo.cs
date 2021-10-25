using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Tests" + PublicKey.TestValue)]
[assembly: InternalsVisibleTo(assemblyName: "RunTests" + InternalPublicKey.Value)]

[assembly: WantsToBeBestFriends]
