// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
using namespace std::chrono;
class CTimer
{

private:

    steady_clock::time_point startPerfomanceCount;
    steady_clock::time_point endPerfomanceCount;
    duration<float> totalElapsed;

public:
    CTimer()
    {
        Initialize();
    }

    CTimer(bool bStartOnCreate)
    {
        Initialize();

        if (bStartOnCreate)
        {
            Start();
        }
    }

    void Initialize()
    {
        totalElapsed = duration<float>();
    }

    void Start()
    {
        startPerfomanceCount = std::chrono::steady_clock::now();
    }

    // time unit: seconds
    void Tag(const std::string& message)
    {
        endPerfomanceCount = std::chrono::steady_clock::now();
        totalElapsed += duration_cast<duration<float>> (endPerfomanceCount - startPerfomanceCount);
        OutputStatistics(message);
        //start next round
        Start();
    }

    float GetTotalElaps()
    {
        return totalElapsed.count();
    }
    float GetTimeSpan()
    {
        endPerfomanceCount = std::chrono::steady_clock::now();
        totalElapsed += duration_cast<duration<float>> (endPerfomanceCount - startPerfomanceCount);
        float timespent = totalElapsed.count();

        //start next round
        Start();

        return timespent;
    }

    float GetTaggedTimeSpan()
    {
        return duration_cast<duration<float>> (endPerfomanceCount - startPerfomanceCount).count();
    }

    void OutputStatistics(const std::string& message)
    {
        std::cout << "Time Cost totally: " << std::fixed << std::setprecision(6) << GetTotalElaps()
            << ", last time span(" << message << "): " << GetTaggedTimeSpan()
            << std::defaultfloat << " seconds.\n";
    }

private:
    CTimer(const CTimer& obj);
};