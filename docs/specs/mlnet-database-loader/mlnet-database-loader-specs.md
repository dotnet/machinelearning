# Specs for ML.NET Relational Database Loader

This specs-doc focuses on the features needed for the base **ML.NET API**, most of all.
The scenarios related to ML.NET **AutoML API**, the **CLI** and **VS Model Builder** will also be considered and covered in this document by in a significantly less detail since there should be different spec docs for those additional tools and APIs.

# Problem to solve

Customers (.NET developers in the enterprise, ISVs, etc.) have tolds us through multiple channels that one of their main source of data, including data for datasets to be used in machine learning, comes directly from databases. This is a very important difference between .NET developers in the enterprise and data scientists who are used to work with datasets as text files.

ML.NET 1.0 and 1.1 only supports the [IDataView LoadFromEnumerable()](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.dataoperationscatalog.loadfromenumerable?view=ml-dotnet-1.1.0) method in order to load data into an IDataView from any source that produces an IEnumerable structure, typically a database. However, the code for loading that data from a database (i.e. using Entity Framework or symply System.Data) is responsability of the user/developer. When training with very large tables in databases so training time could be many hours, such code can be complicated in certain cases.

Within the 'databases scope' problem there are multiple areas.

The **scope** for this feature is initially limited to **relational databases** with higher priority on SQL Server and Azure SQL Database, but one of the goals is to make this loader/connector compatible with any relational database which is supported by .NET providers.

- Scope to support in this feature:
    - Relational Databases, such as:
        - SQL Server and Azure SQL Database
        - Other relational databases such as MySQL, Oracle, PostgreSQL, etc.

- Scope not supported in this feature:
    - No-SQL Databases, such as:
        - Azure Cosmos DB
        - MongoDB
        - Any other non-SQL database
    - Big Data, such as:
        - Big Data: Spark, Hadoop, etc.


## Evidence

Sample feedback and quotes:

- *"[Rachel Poulsen comment](https://www.quora.com/What-are-the-most-valuable-skills-to-learn-for-a-data-scientist-now), Data Scientist at Silicon Valley Data Science: I have been a Data Scientist in Silicon Valley for the last 4 years. There are A LOT of tools out there for data science as I'm sure you are aware. But very importantly, SQL: I'd argue this is 100% necessary for all data scientists, regardless of whether you are using structured or unstructured data. Too many people and companies still use it and are comfortable with it for it to be thrown out with NoSQL techniques. Even companies using NoSQL databases and unstructured data are still using SQL."*

- *"forki commented on May 17, 2018
I did not say EF ;-) - I heard some of the authors of this package worked on EF before, but please please don't bind it to EF, or sql server or any other specific tech. Or course you want to get data into the learning system without going over to csv. But the data is usually already flattened anyway, so no need for a complex OR mapping tool - especially since the target data structure is Immutable"*

- *Other customers - TBD*

# Who is the Customer

- .NET developers used to work with databases who are getting started with machine learning with ML.NET.


# Goals

The business goals are the following, depending on the possible scenarios:

- Ability for developers to load and automatically stream data from relational databases in order to train/evaluate ML.NET models.
- The code to load from a database should be extremely easy, a single line of code in most cases.
- Tooling (Model Builder in VS and the CLI) and AutoML API should also support this feature.

# Solution

The solution is to create an ML.NET database loader classes supporting the above scenarios.

The main supported features are:

- **Easily load data into IDataView from databases**: The main goal is to load data into IDataView from databases very easily (a single line of code as one of the approaches) and properly (with good performance when streaming data while training).

- **Train ML.NET models with data streaming coming from a database**: As the ultimate goal, the main purpose of loading data into an IDataView is to usually train an ML.NET model with data coming from a database table (or view or SQL sentence).

- **Evaluate ML.NET models with data streaming coming from a database**: As secondary goal is to evaluate the quality/accuracy of an ML.NET model with data coming from a database table (or view or SQL sentence) by comparing it with a set or predictions. This goal should support:

  - Train/test split scenario (one database source as training dataset and one database source as testing-dataset), such as a 80%-20% approach.

  - Train/evaluation/test split scenario (one database source as training dataset, one database source as evaluation-dataset and one database source as test-dataset), such as a 50%-30%-20% approach.

  - Cross-validation scenario. Single database source. Internally it'll be split in multiple folds (such as 5 folds) for multiple trains and tests. This should be transparent from a database connection point of view which only needs one database source.

- **Additional support for AutoML API, CLI and Model Builder:** Loading data from databases should be supported by AutoML API, Model Builder in VS and the ML.NET CLI.

--------------------------------------

TBD - Include diagram with blocks of AutoML, CLI and Model Buidler using this component/loader

--------------------------------------


## Design criteria

- **Supported frameworks in .NET**: The DatabaseLoader component should be supported for the [frameworks supported by ML.NET](https://github.com/dotnet/machinelearning#installation) which include applications running on:

    - .NET Core 2.1 and higher
    - .NET Framework 4.6.1 and higher

The way to support those frameworks would be by creating a **.NET Standard 2.0 library** as the base for this package since .NET Standard is supported by both frameworks.

- **Supported RDBMS**: The DatabaseLoader component should be supported for most of the [data providers supported by System.Data.Common in .NET Standard 2.0](https://docs.microsoft.com/en-us/dotnet/framework/data/adonet/data-providers) which special focus/support on:

    - **P1 RDBMS support/tested priorities:**

        The following data providers should be tested with higher priority:

        - Data Provider for SQL Server (Microsoft provider)
        - Data Provider for Oracle - Test on:
            - [Oracle Data Provider for .NET (ODP.NET)](https://www.oracle.com/technetwork/developer-tools/visual-studio/downloads/index-085163.html)
        - MySql provider - Test on:
        - [MySQL Connector/NET provider](https://dev.mysql.com/doc/connector-net/en/connector-net-versions.html) / [MySql.Data NuGet](https://www.nuget.org/packages/MySql.Data/)
        - PostgreSQL providers - Test on:
        - Npgsql open source ADO.NET Data Provider for PostgreSQL

    This ML.NET database loader won't probably need Entity Framework, but for a relationship, see [EF providers](https://docs.microsoft.com/en-us/ef/core/providers/) for a relationship to ADO.NET providers.

    - **P2 RDBMS support/tested priorities:**

        Other data providers to test with lower priority. It will also need to select specific RDBMS on OLE DB and ODBC:

        - Data Provider for OLE DB
        - Data Provider for ODBC
        - Data Provider for EntityClient Provider (Entity Data Model (EDM))

- **CRITICAL: Implement support for 'handle and continue' after transient errors happening in Azure SQL Database (or any DB):** When using Azure SQL Database as the source of your training database, because databases in Azure SQL DB can be moved to different servers across the internal Azure SQL Database cluster, transient failures (usually for just a few seconds) in the form of connectivity exceptions can happen. Even further, by design in Azure SQL Database, if a process is blocking too many resources in SQL, sometimes the database connection can be thrown away in favor of other customers/databases.
There are several strategies in order to handle database transient errors (see [Working with SQL Database connection issues and transient errors](https://docs.microsoft.com/en-us/azure/sql-database/sql-database-connectivity-issues)) like doing a 'Retry strategy' and start with a new connection again. But that strategy is only okay for short/fast queries. That simple strategy which throws away all the progress made and start the same query again wouldn't be good when training with a very large table because it could mean that the training operation "never finishes" if you have at least one transient error on every "training try".
We'll need to come up with a reasonably general pattern (probably something that reasons about primary keys), but this scenario is not simple.

    See [related issue](https://github.com/dotnet/machinelearning-samples/issues/507)

- **Support for remote/network connection and internal connection within the RDBMS server**: The Database loader should not only support remote/network connection to the RDBMS server but also support from C# running within the RDBMS server such as [SQL Server CLR integration](https://docs.microsoft.com/en-us/sql/relational-databases/clr-integration/clr-integration-overview?view=sql-server-2017), [Oracle Database Extensions for .NET](https://docs.oracle.com/cd/B19306_01/win.102/b14306/intro.htm), etc.  Scenarios:

    1. Training-machine-ML.NET-code <--network--> Remote database-in-RDBMS-server

    2. RDBMS-server running database and .NET code using ML.NET code

- **NuGet packages and libraries design**:
The implementation of this feature should be packaged following the following approach, which is aligned and consistent to the current approach used by the .NET Framework and .NET Core in the System.Data.Common and System.Data.SqlClient:

    - Implementation code with NO depedencies to specific database providers (such as SQL Server, Oracle, MySQL, etc.) will be packaged in the same NuGet package and library than the existing TextLoader-related classes which is in the Microsoft.ML.Data library. This code is basically the foundational API for the Database loader where the user has to provide any specific database connection (so dependencies are taken in user's code).
    - Implementation code WITH dependencies to data proviers (such as SQL Server, Oracle, MySQL, etc.) that might be created when creating additional convenient APIs where the user only needs to provide a connection string and table-name or SQL statement, will be placed in a segregated class library and NuGet package, so that ML.NET core packages don't depend on specific database providers.

- **Support for sparse data**: The database loader should support sparse data, at least up to the maximum number of columns in SQL Server (1,024 columns per nonwide table, 30,000 columns per wide table or 4,096 columns per SELECT statement).

    ML.NET supports sparse data such as in the following example using a [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) of thousands or even millions of columns even when in this example only 200 columns have real data (sparse data):

    - [ML.NET sample using millions of columns with sparse data](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/LargeDatasets)

    SQL Server supports [Sparse columns](https://docs.microsoft.com/en-us/sql/relational-databases/tables/use-sparse-columns?view=sql-server-2017), however, it is just a way to optimize storage for null values. It still needs to have a real column created in the table per each logical column (i.e. 1,000 columns defined in the SQL table) even when it might not have data.






# Gated conditions for first Preview release

- TBD

# Gated conditions for GA release

- TBD

# API design examples

The following are a few C# examples on how this API could be used by a developer when loading data from a database into an IDataView.

Whenever in the samples below it is using a `SqlConnection` type, it could also be using any other supported DbConnection in .NET providers, such as:

- `SqlConnection` (SQL Server)
- `NpgsqlConnection` (PostgreSQL)
- `OracleConnection` (Oracle)
- `MySqlConnection` (MySql)
- etc.

The specific type (`SqlConnection`, `NpgsqlConnection`, `OracleConnection`, etc.) is specified as a generic parameter so references to the needed NuGet packages will be provided by the user when trying to compile.

**1. (Convenient 'must-have' method) Data loading from a database by specifying a SQL query sentence:**

The signature of the `DataOperationsCatalog.LoadFromDbSqlQuery` method looks like:

```C#
public IDataView LoadFromDbSqlQuery<TRow, DbConnection>(string connString, string sqlQuerySentence) where T : class;
```

`TRow` is the model's input data class (Observation class) to be used by IDataView and ML.NET API.

Example code using it:

```C#
MLContext mlContext = new MLContext();

//Example loading from a SQL Server or SQL Azure database with a SQL query sentence
IDataView trainingDataView = mlContext.Data.LoadFromDbSqlQuery<ModelInputData, SqlConnection>(connString: myConnString, sqlQuerySentence: "Select * from InputMLModelDataset where InputMLModelDataset.CompanyName = 'MSFT'");
```

**2. (Foundational method) Data loading from a database with a System.Data.IDataReader object:**

This is the foundational or pillar method which will be used by the rest of the higher level or convenient methods:

The signature of the `DataOperationsCatalog.LoadFromDbDataReader` method looks like:

```C#
public IDataView LoadFromDbDataReader<TRow>(Func<IDataReader> executeReader) where TRow : class;
```

`TRow` is the model's input data class (Observation class) to be used by IDataView and ML.NET API.

The following code is how the mlContext.Data.LoadFromDbDataReader() method is used:

```C#
MLContext mlContext = new MLContext();

string conString = YOUR_CONNECTION_STRING;
using (SqlConnection connection = new SqlConnection(conString))
{
    connection.Open();
    using (SqlCommand command = new SqlCommand("Select * from InputMLModelDataset", connection))
    {
        mlContext.Data.LoadFromDbDataReader<ModelInputData>(() => command.ExecuteReader());
    }
}
```

**IMPORTANT - Lambda with command.ExecuteReader() as parameter instead of a IDataReader as paramater:** It is important to highlight that because we want to enforce that each query will use a different data reader object (in order to avoid multi-threading issues), the `mlContext.Data.LoadFromDbDataReader<TRow>()` method is accepting a lambda expression with the `command.ExecuteReader()` line of code. That means the method only accepts a `Func<IDataReader>` delegate because the lambda expression has no input parameters and returns an IDataReader object (such as `SqlDataReader`).


**3. ('Nice to have') Data loading from a database table (Simplest approach):**

This is mostly a 'sugar method', but can be a 'nice to have feature' for users.
The signature of the `DataOperationsCatalog.LoadFromDbTable` method looks like:

```C#
public IDataView LoadFromDbTable<TRow, DbConnection>(string connString, string tableName)  where T : class;
```

Example code using it:

```C#
MLContext mlContext = new MLContext();

//Example loading from a SQL Server or SQL Azure database table
IDataView trainingDataView = mlContext.Data.LoadFromDbTable<ModelInputData, SqlConnection>(connString: myConnString,
                                                                                                 tableName: "TrainingDataTable");
```

**4. ('Nice to have') Data loading from a database view:**

This is mostly a 'sugar method', but can be a 'nice to have feature' for users.
The signature of the `DataOperationsCatalog.LoadFromDbView` method looks like:

```C#
public IDataView LoadFromDbView<TRow, DbConnection>(string connString, string viewName) where T : class;
```

Example code using it:

```C#
MLContext mlContext = new MLContext();

//Example loading from a SQL Server or SQL Azure database view
IDataView trainingDataView = mlContext.Data.LoadFromDbView<ModelInputData, SqlConnection>(connString: myConnString,
                                                                                                viewName: "TrainingDatabaseView");
```

## Support connectivity from .NET assemblies embedded into the RDBMS server

As introduced, the database loader should not only support remote/network connection to the RDBMS server but also support connectivity from C# running within the RDBMS server such as [SQL Server CLR integration](https://docs.microsoft.com/en-us/sql/relational-databases/clr-integration/clr-integration-overview?view=sql-server-2017) (aka [SQL CLR](https://en.wikipedia.org/wiki/SQL_CLR)), [Oracle Database Extensions for .NET](https://docs.oracle.com/cd/B19306_01/win.102/b14306/intro.htm), etc.

The only difference is the way you define the connection string, which simply provides **'context' string** (instead of server name, user, etc. when using the network), such as:

- Code example running on [SQL Server CLR integration](https://docs.microsoft.com/en-us/sql/relational-databases/clr-integration/clr-integration-overview?view=sql-server-2017)

    ```
    //SQL Server
    SqlConnection conn   = new SqlConnection("context connection=true")
    ```

    - See here an [example of a CLR Scalar-Valued Function](https://docs.microsoft.com/en-us/sql/relational-databases/clr-integration-database-objects-user-defined-functions/clr-scalar-valued-functions?view=sql-server-2017#example-of-a-clr-scalar-valued-function)

    - [Introduction to SQL Server CLR Integration](https://docs.microsoft.com/en-us/dotnet/framework/data/adonet/sql/introduction-to-sql-server-clr-integration)

- Code example running on [Oracle Database Extensions for .NET](https://docs.oracle.com/cd/B19306_01/win.102/b14306/intro.htm)
    ```
    //Oracle
    OracleConnection con = new OracleConnection();
    con.ConnectionString = "context connection=true";
    ```

    - See here an [exampleof a C# stored procedure in Oracle ](https://www.oracle.com/technetwork/articles/dotnet/williams-sps-089817.html?printOnly=1)

The code should be similar on any other RDBMS supporting .NET assemblies running within the database engine. The only change should be provided within the connection string.

**Out of scope:**

ML.NET won't implement components creating concrete database objects such as **CLR Scalar-Valued Functions** or **CLR/C# stored procedures** (which have different implementation for SQL Server, Oracle, etc.). Those components should be created by the user/developer using C# then adding ML.NET API code in order to train a model.

Also, note that  the fact that ML.NET will be supported to be used within user components using CLR integration, that doesn't mean that the user can do it on any RDBMS. There are RDBMS such as Azure SQL Database with single databases and elastic pools and other RDBMS that don't support that feature. Other RDBMS suchas SQL Server on-premises or Azure SQL Database Managed Instances, Oracle, etc. do support it.

For instance:

- [Feature comparison: Azure SQL Database versus SQL Server](https://docs.microsoft.com/en-us/azure/sql-database/sql-database-features)

# Input data classes/types design

There can be two different approaches here:

- Use similar input data classes/types to ML.NET 1.x input data classes
- Use similar input data classes/types to Entity Framework POCO entity data model classes

## Approach A: Using ML.NET input data classes

A sample ML.NET input data class are the following:

**A.1 - Simple input data class:**

```C#
using Microsoft.ML.Data;
//...
private class ModelInputData
{
    [LoadColumn(0)]
    public bool SentimentLabel { get; set; }
    [LoadColumn(1)]
    public string SentimentText { get; set; }
}
```

**A.2 - Large number of columns loaded at the same time into a single vector column:**

When you have hundreds or even thousands of columns, those columns should be grouped in the data model class by using "vector valued columns" which are arrays of types (such as float[]) for the contiguous columns with the same type, as the following code example:

```C#
using Microsoft.ML.Data;
//...
private class ModelInputData
{
    [LoadColumn(0)]
    public float TargetLabel { get; set;}

    [LoadColumn(1, 2000), ColumnName("Features")]
    [VectorType(2000)]
    public float[] FeatureVector { get; set;}
}
```

Pros:
- It uses the same consistent model input classes already used in ML.NET when loading from dataset files
- It supports loading many columns at the same time into a single vector column

Cons:
- It needs special attributes such as `LoadColumnAttribute`, therefore, these input classes are not POCO classes since they have a dependency on ML.NET packages.

**A.3 - Support column-map 'by property name' convention:**

When loading from a database, it has to be able to use a property-name based convention, meaning that if the class is not providing the `LoadColumnAttribute` then it would try to map the column name in the database table to the property's name, such as the following case:

```C#
private class ModelInputData
{
    public bool SentimentLabel { get; set; }
    public string SentimentText { get; set; }
}
```

This last approach is similar to the Entity Framework POCO entity class approach, however, Entity Framework POCO entity classes support an additional large number of .NET types including complex-types and navigation/embedded types.

*In a different page and related to dataset files instead of databases, that feature would also be useful if using dataset files with a header with column names matching property names.*

## Approach B: Using Entity Framework POCO entity data model classes

When using Entity Framework, a POCO entity is a class that doesn't depend on any framework-specific base class. This is also why they are persistence-ignorant objects following the [persistence ignorance principle](https://deviq.com/persistence-ignorance/).

It is like any other normal .NET CLR class, which is why it is called POCO ("Plain Old CLR Object").

POCO entities are supported in both EF 6.x and EF Core.

```C#
public class ModelInputData
{
    public int StudentID { get; set; }
    public string StudentName { get; set; }
    public DateTime? DateOfBirth { get; set; }
    public byte[]  Photo { get; set; }
    public decimal Height { get; set; }
    public float Weight { get; set; }

    public StudentAddress StudentAddress { get; set; }
    public Grade Grade { get; set; }
}
```

Pros:
- It uses the same data model classes used in Entity Framework entity models, familiar to many .NET developers.
- It doesn't need special attributes with dependency on ML.NET packages since they are POCO classes.

Cons:
- EF does not support loading many columns at the same time into a single vector column.
- EF requires a mandatory ID property in the POCO class
- ML.NET might not support certain .NET types allowed by EF POCO classes (i.e. DateTime, etc.).
- ML.NET doesn't support embedded/navigation/relationship entity types such as `StudentAddress` in the sample above, neither complex-types in EF.
- Input data classes won't be consistent/similar to ML.NET input data classes when using dataset files.

### Selected approach for input data class when reading from a database

*TO BE DISCUSSED/CONFIRMED:*

Due to the mentioned pros/cons above and additional constraints in ML.NET supported types, the most feasible approach in the short term is **to use ML.NET input data classes with/without ML.NET attributes**, consistent with current support in ML.NET, as ML.NET input data class when reading from a database.

Supporting the same scope of POCO entities supported by entity Framework seems problematic due to many more additional types supported plus embeded navigation types/classes (complex types), etc.


# ML.NET CLI design samples

**1. CLI training from a database table (Simplest approach):**

Sample CLI command:

```
> mlnet auto-train --task regression --db-conn-string "YOUR-DATABASE-CONNECTION-STRING" --db-table "MyTrainingDbTable" --label-column-name Price
```

**2. CLI training from a database view:**

Sample CLI command:

```
> mlnet auto-train --task regression --db-conn-string "YOUR-DATABASE-CONNECTION-STRING" --db-view "MyTrainingDbView" --label-column-name Price
```

**3. CLI training from a database with a SQL query sentence:**

Sample CLI command:

```
> mlnet auto-train --task regression --db-conn-string "YOUR-DATABASE-CONNECTION-STRING" --sql-query "SELECT * FROM MyTrainingDbTable WHERE Company = 'MSFT'" --label-column-name Price
```


# ML.NET AutoML API design samples

For ML.NET AutoML the C# code to use is the same than for regular ML.NET code since the code to load data infor an IDataView should use the same API, such as the following example:

```C#
MLContext mlContext = new MLContext();

//Load train dataset from a database table
IDataView trainDataView = mlContext.Data.LoadFromDatabaseTable<ModelInputData, SqlConnection>(connString: myConnString, tableName: "MyTrainDataTable");

//Load train dataset from a database table
IDataView testDataView = mlContext.Data.LoadFromDatabaseTable<ModelInputData, SqlConnection>(connString: myConnString, tableName: "MyTestDataTable");

// Run AutoML experiment
var progressHandler = new BinaryExperimentProgressHandler();

ExperimentResult<BinaryClassificationMetrics> experimentResult = mlContext.Auto()
    .CreateBinaryClassificationExperiment(ExperimentTime)
    .Execute(trainingDataView, progressHandler: progressHandler);
```

Therefore, most of the code above is regular AutoML API code and the only pieces of code using the DatabaseLoader are using the same API than when using regular ML.NET code for loading data from a database.

# Model Builder for Visual Studio mock UI samples

TBD




# Open questions

- QUESTION 1 TBD:


# References

- [TBD](http://TBD)

