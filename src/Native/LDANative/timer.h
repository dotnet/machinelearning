#pragma once
#include <chrono>
using namespace std::chrono;
class CTimer
{

private:

    steady_clock::time_point startPerfomanceCount;
    steady_clock::time_point endPerfomanceCount;
    duration<float> totalElapsed;
public:
    char m_szMessage[1024];

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
    void Tag(const char* pszMsg = NULL)
    {
        endPerfomanceCount = std::chrono::steady_clock::now();
        totalElapsed += duration_cast<duration<float>> (endPerfomanceCount - startPerfomanceCount);
        OutputStatistics(pszMsg);
        //start next round
        Start();
    }

    // time unit: seconds
    void InnerTag()
    {
        endPerfomanceCount = std::chrono::steady_clock::now();
        totalElapsed += duration_cast<duration<float>> (endPerfomanceCount - startPerfomanceCount);

        OutputStatistics(m_szMessage);

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

    void OutputStatistics(const char* pszMsg = NULL)
    {
        printf("Time Cost totally: %f, last time span(%s): %f seconds.\n", GetTotalElaps(), pszMsg, GetTaggedTimeSpan());
    }

private:
    CTimer(const CTimer& obj);
};