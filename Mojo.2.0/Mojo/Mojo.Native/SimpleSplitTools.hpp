#pragma once

// Point split
#define SOURCE_TARGET 1
#define BORDER_TARGET 2

// One region split
#define REGION_SPLIT 3

// Two region split
#define REGION_A 4
#define REGION_B 5

// Result (all modes)
#define PATH_RESULT 6

// Search mask
#define MASK_VALUE 10
#define MASK_PENALTY_VALUE 65536

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