#pragma once

#define SOURCE_TARGET -2
#define BORDER_TARGET -1

#define BONUS_REGION -3
#define BONUS_VALUE 65536

#define PATH_RESULT_VALUE -2

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