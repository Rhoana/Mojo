#pragma once

#define SPLIT_SEARCH_REGION_VALUE 1

#define BORDER_TARGET 2
#define SOURCE_TARGET 3

#define PATH_RESULT_VALUE 4

#define BONUS_REGION 5
#define BONUS_VALUE 65536

#define PENALTY_REGION 6
#define PENALTY_VALUE 65536


namespace Mojo
{
namespace Native
{

class SimpleSplitTools
{

private:
    const static int small_r;
    const static int small_wh;
    const static int small_nhood[];
    const static int large_r;
    const static int large_wh;
    const static int large_nhood[];

public:
    static void DijkstraSearch ( const int* searchArea, const int* searchMask, const int* searchBonus, const int fromIndex, const int width, const int height, const int targetMax, int* dist, int* prev, int* toIndex );
    static void ApplySmallMask ( const int fromIndex, const int width, const int height, const int targetVal, int* area );
    static void ApplyLargeMask ( const int fromIndex, const int width, const int height, const int targetVal, int* area );

    static void ApplyCircleMask ( const int fromIndex, const int width, const int height, const int targetVal, float radius, int* area );
    static void ApplyCircleMask ( const int fromIndex, const int width, const int height, const int targetVal, float radius, unsigned int* area );

};

}
}