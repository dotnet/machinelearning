// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#nullable enable

namespace Microsoft.ML.IsolationForest.Sample
{
    /// <summary>Website performance sample output row.</summary>
    internal sealed class WebOut
    {
        public float P95Ms { get; set; } = float.NaN;
        public float ErrorRatePct { get; set; } = float.NaN;

        // Isolation Forest outputs
        public float IF_Score { get; set; } = float.NaN;   // scaled 0..100
        public bool IF_Label { get; set; } = false;        // anomaly flag
    }

    /// <summary>IoT / sensor sample output row.</summary>
    internal sealed class IoTOut
    {
        public float TemperatureC { get; set; } = float.NaN;
        public float HumidityPct { get; set; } = float.NaN;
        public float PressureKPa { get; set; } = float.NaN;
        public float VibrationMmS { get; set; } = float.NaN;

        // Isolation Forest outputs
        public float IF_Score { get; set; } = float.NaN;
        public bool IF_Label { get; set; } = false;
    }

    /// <summary>Manufacturing / process control sample output row.</summary>
    internal sealed class ProcessOut
    {
        public float SpeedRpm { get; set; } = float.NaN;
        public float TorqueNm { get; set; } = float.NaN;
        public float TempC { get; set; } = float.NaN;
        public float HumidityPct { get; set; } = float.NaN;

        // Isolation Forest outputs
        public float IF_Score { get; set; } = float.NaN;
        public bool IF_Label { get; set; } = false;
    }

    /// <summary>Retail / purchases sample output row.</summary>
    internal sealed class PurchaseOut
    {
        public float Amount { get; set; } = float.NaN;
        public float ItemCount { get; set; } = float.NaN;
        public float CustomerTenureDays { get; set; } = float.NaN;

        // Isolation Forest outputs
        public float IF_Score { get; set; } = float.NaN;
        public bool IF_Label { get; set; } = false;
    }

    /// <summary>Support tickets / operations sample output row.</summary>
    internal sealed class TicketOut
    {
        public float Words { get; set; } = float.NaN;
        public float Sentiment { get; set; } = float.NaN;
        public float HourOfDay { get; set; } = float.NaN;

        // Isolation Forest outputs
        public float IF_Score { get; set; } = float.NaN;
        public bool IF_Label { get; set; } = false;
    }

    /// <summary>Generic two-output DTO (score + label) for quick samples.</summary>
    internal sealed class IfOut
    {
        public float IF_Score { get; set; } = float.NaN;
        public bool IF_Label { get; set; } = false;
    }
}
