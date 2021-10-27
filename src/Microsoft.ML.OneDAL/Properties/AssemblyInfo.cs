using System.Runtime.CompilerServices;
using Microsoft.ML;

[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Tests" + PublicKey.TestValue)]

[assembly: InternalsVisibleTo(assemblyName: "RunTests" + InternalPublicKey.Value)]
[assembly: InternalsVisibleTo(assemblyName: "Microsoft.ML.Internal.MetaLinearLearner" + InternalPublicKey.Value)]

[assembly: WantsToBeBestFriends]
