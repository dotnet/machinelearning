// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Runtime.RunTests
{
    // The Xunit test framework requires the per-test initialization be implemented
    // as the test class constructor, and per-test clean-up be implemented in Dispose()
    // while having the class implementing IDisposable. To minimize changes to existing 
    // tests and make things look cleaner, for all those test classes that need
    // initialization or clean-up, we make them partial and put the corresponding 
    // constructors or IDisposable.Dispose() implementations here.

    public abstract partial class BaseTestPredictors : TestDmCommandBase
    {
        protected BaseTestPredictors(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public partial class TestBaselineNormalize : BaseTestBaseline
    {
        public TestBaselineNormalize(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public abstract partial class TestCommandBase : TestDataViewBase
    {
        protected TestCommandBase(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestConcurrency : BaseTestPredictors
    {
        public TestConcurrency(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestDataPipe : TestDataPipeBase
    {
        public TestDataPipe(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public abstract partial class TestDataPipeBase : TestDataViewBase
    {
        protected TestDataPipeBase(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestParquet : TestDataPipeBase
    {
        public TestParquet(ITestOutputHelper helper)
                    : base(helper)
        {
        }
    }

    public sealed partial class TestExceptionPropagation : TestDataViewBase
    {
        public TestExceptionPropagation(ITestOutputHelper helper)
           : base(helper)
        {
        }
    }

    public partial class TestEntryPoints : TestDataViewBase
    {
        public TestEntryPoints(ITestOutputHelper helper)
           : base(helper)
        {
        }
    }

    public sealed partial class TestSummaryEntryPoints : TestDataViewBase
    {
        public TestSummaryEntryPoints(ITestOutputHelper helper)
           : base(helper)
        {
        }
    }

    public sealed partial class TestDataPipeSkipTakeFilter : TestDataPipeBase
    {
        public TestDataPipeSkipTakeFilter(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public abstract partial class TestDataViewBase : BaseTestBaseline
    {
        protected TestDataViewBase(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestDataPipeNoBaseline : TestDataViewBase
    {
        public TestDataPipeNoBaseline(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestDmCommand : TestSteppedDmCommandBase
    {
        public TestDmCommand(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public abstract partial class TestDmCommandBase : TestCommandBase
    {
        protected TestDmCommandBase(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class ExprLanguageTests : BaseTestBaseline
    {
        public ExprLanguageTests(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class NeuralNetworkSerializationTests : BaseTestBaseline
    {
        public NeuralNetworkSerializationTests(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestPipeline : TestCommandBase
    {
        public TestPipeline(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestStringEnumerableTextReader : TestCommandBase
    {
        public TestStringEnumerableTextReader(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestPredictors : BaseTestPredictors
    {
        public TestPredictors(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestPredictorsLightGbm : BaseTestPredictors
    {
        public TestPredictorsLightGbm(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestReapply : TestDataPipeBase
    {
        public TestReapply(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }



    public sealed partial class TestTransposer : TestDataPipeBase
    {
        public TestTransposer(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public sealed partial class TestImageAnalyticsTransforms : TestDataPipeBase
    {
        public TestImageAnalyticsTransforms(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public partial class TestRepositoryReader : BaseTestBaseline
    {
        public TestRepositoryReader(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public partial class TestSweeper : BaseTestBaseline
    {
        public TestSweeper(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public partial class TestResourceDownload : BaseTestBaseline
    {
        public TestResourceDownload(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }

    public partial class TestResultProcessor : BaseTestPredictors
    {
        public TestResultProcessor(ITestOutputHelper helper)
            : base(helper)
        {
        }
    }
}

namespace Microsoft.ML.Runtime.RunTests.RServerScoring
{

    public sealed partial class TestRServerScoringLibrary : TestDataViewBase
    {
        public TestRServerScoringLibrary(ITestOutputHelper helper)
           : base(helper)
        {
        }
    }
}
