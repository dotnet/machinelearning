using System;
using System.Data.SqlClient;

namespace Microsoft.ML.TestFrameworkCommon
{
    public static class Centrallizedlogger
    {
        //should hide connect string details or use intergrated security but this is only test data and should be removed after test is more stable so keep it for now
        static string connetionString = "Server=tcp:mldotnettest.database.windows.net,1433;Initial Catalog=flakytests;Persist Security Info=False;User ID=mldotnet;Password=Password2019^^;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;";
        
        public static void LogFlakyTests(string displayName, string os, string architecture, string framework,
            string configuration, int failCount, string messages, DateTime failTime)
        {
            //schema of flakytests table is like below
            //create table flakytests(
            //    testid int IDENTITY(1, 1) PRIMARY KEY,
            //    displayname varchar(255),
            //    os varchar(255),
            //    architecture varchar(255),
            //    framework varchar(255),
            //    configuration varchar(255),
            //    failcount int,
            //    messages varchar(MAX),
            //    failtime datetime
            //)
            string insStmt = "insert into flakytests ([displayname], [os], [architecture], [framework], [configuration], [failcount], [messages], [failtime]) values " +
                " (@displayname, @os, @architecture, @framework, @configuration, @failcount, @messages, @failtime)";

            try
            {
                using (SqlConnection cnn = new SqlConnection(connetionString))
                {
                    cnn.Open();
                    SqlCommand insCmd = new SqlCommand(insStmt, cnn);

                    insCmd.Parameters.AddWithValue("@displayname", displayName);
                    insCmd.Parameters.AddWithValue("@os", os);
                    insCmd.Parameters.AddWithValue("@architecture", architecture);
                    insCmd.Parameters.AddWithValue("@framework", framework);
                    insCmd.Parameters.AddWithValue("@configuration", configuration);
                    insCmd.Parameters.AddWithValue("@failcount", failCount);
                    insCmd.Parameters.AddWithValue("@messages", messages);
                    insCmd.Parameters.AddWithValue("@failtime", failTime);
                    insCmd.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Fail to write message into db with exception {ex.Message}");
            }
        }
    }
}
