#include <math.h>
#include "SimpleSplitTools.hpp"
#include "Mojo.Core/Stl.hpp"
#include "Mojo.Core/Comparator.hpp"

namespace Mojo
{
namespace Native
{

//const int SimpleSplitTools::small_r = 4;
//const int SimpleSplitTools::small_wh = 9;
//const int SimpleSplitTools::small_nhood[] = { 0, 0, 0, 0, 1, 0, 0, 0, 0,
//												0, 0, 1, 1, 1, 1, 1, 0, 0,
//												0, 1, 1, 1, 1, 1, 1, 1, 0,
//												0, 1, 1, 1, 1, 1, 1, 1, 0,
//												1, 1, 1, 1, 1, 1, 1, 1, 1,
//												0, 1, 1, 1, 1, 1, 1, 1, 0,
//												0, 1, 1, 1, 1, 1, 1, 1, 0,
//												0, 0, 1, 1, 1, 1, 1, 0, 0,
//												0, 0, 0, 0, 1, 0, 0, 0, 0 };

const int SimpleSplitTools::small_r = 3;
const int SimpleSplitTools::small_wh = 7;
const int SimpleSplitTools::small_nhood[] = { 0, 0, 0, 1, 0, 0, 0,
												0, 1, 1, 1, 1, 1, 0,
												0, 1, 1, 1, 1, 1, 0,
												1, 1, 1, 1, 1, 1, 1,
												0, 1, 1, 1, 1, 1, 0,
												0, 1, 1, 1, 1, 1, 0,
												0, 0, 0, 1, 0, 0, 0 };

const int SimpleSplitTools::large_r = 6;
const int SimpleSplitTools::large_wh = 13;
const int SimpleSplitTools::large_nhood[] = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
												0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
												0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
												0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
												0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
												0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
												1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
												0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
												0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
												0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
												0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
												0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
												0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };

void SimpleSplitTools::ApplySmallMask ( const int fromIndex, const int width, const int height, const int targetVal, int* area )
{
    int nx, ny, ni, ax, ay;
    for ( nx = -small_r; nx <= small_r; ++nx )
    {
        for ( ny = -small_r; ny <= small_r; ++ny )
        {
            ni = nx + small_r + ( ny + small_r ) * small_wh;
            if ( small_nhood[ ni ] )
            {
                ax = fromIndex % width + nx;
                ay = fromIndex / width + ny;
                if ( ax >= 0 && ax < width &&
                    ay >= 0 && ay < height )
                {
                    area[ ax + ay * width ] = targetVal;
                }
            }
        }
    }
}

void SimpleSplitTools::ApplyLargeMask ( const int fromIndex, const int width, const int height, const int targetVal, int* area )
{
    int nx, ny, ni, ax, ay;
    for ( nx = -large_r; nx <= large_r; ++nx )
    {
        for ( ny = -large_r; ny <= large_r; ++ny )
        {
            ni = nx + large_r + ( ny + large_r ) * large_wh;

            if ( large_nhood[ ni ] )
            {
                ax = fromIndex % width + nx;
                ay = fromIndex / width + ny;
                if ( ax >= 0 && ax < width &&
                    ay >= 0 && ay < height )
                {
                    area[ ax + ay * width ] = targetVal;
                }
            }
        }
    }
}

void SimpleSplitTools::ApplyCircleMask ( const int fromIndex, const int width, const int height, const int targetVal, float radius, int* area )
{
    int mask_r = (int) ( radius + 0.5 );
    int mask_wh = mask_r * 2 + 1;
    float dist;

    int nx, ny, ax, ay;

    for ( nx = -mask_r; nx <= mask_r; ++nx )
    {
        for ( ny = -mask_r; ny <= mask_r; ++ny )
        {
            dist = sqrt( (float) ( nx * nx + ny * ny ) );

            if ( dist <= radius )
            {
                ax = fromIndex % width + nx;
                ay = fromIndex / width + ny;
                if ( ax >= 0 && ax < width &&
                    ay >= 0 && ay < height )
                {
                    area[ ax + ay * width ] = targetVal;
                }
            }
        }
    }
}

void SimpleSplitTools::ApplyCircleMask ( const int fromIndex, const int width, const int height, const int targetVal, float radius, unsigned int* area )
{
    int mask_r = (int) ( radius + 0.5 );
    int mask_wh = mask_r * 2 + 1;
    float dist;

    int nx, ny, ax, ay;

    for ( nx = -mask_r; nx <= mask_r; ++nx )
    {
        for ( ny = -mask_r; ny <= mask_r; ++ny )
        {
            dist = sqrt( (float) ( nx * nx + ny * ny ) );

            if ( dist <= radius )
            {
                ax = fromIndex % width + nx;
                ay = fromIndex / width + ny;
                if ( ax >= 0 && ax < width &&
                    ay >= 0 && ay < height )
                {
                    area[ ax + ay * width ] = targetVal;
                }
            }
        }
    }
}

void SimpleSplitTools::DijkstraSearch ( const int* searchArea, const int* searchMask, const int* searchBonus, const int fromIndex, const int width, const int height, const int targetMax, int* dist, int* prev, int* toIndex )
{
	for ( int i = 0; i < width * height; ++i )
	{
		dist[i] = std::numeric_limits<int>::max();
		prev[i] = -1;
	}

	dist[ fromIndex ] = 0;

	std::set< int2, Mojo::Core::Int2Comparator > distQueue;
	distQueue.insert( make_int2( fromIndex, 0 ) );

	bool found_target = false;
	int currentIndex = fromIndex;

	//Core::Printf( distQueue.empty() );
	//Core::Printf( distQueue.size() );

	//
	// Dijkstra until we find a target
	//
	int itCount = 0;
	while ( !distQueue.empty() )
	{
		std::set< int2, Mojo::Core::Int2Comparator >::iterator top = distQueue.begin();
		currentIndex = top->x;
		distQueue.erase( top );

		//Core::Printf( itCount, ": currentIndex=", currentIndex, "  dist=", dist[ currentIndex ], " searchArea=", searchArea[ currentIndex ], " searchMask=", searchMask[ currentIndex ] );

		if ( searchMask[ currentIndex ] <= targetMax )
		{
			found_target = true;
			break;
		}

		if ( dist[ currentIndex ] == -1 )
		{
			RELEASE_ASSERT( 0 );
			break;
		}

		int2 currentPix = make_int2( currentIndex % width, currentIndex / width );

		int nextIndex = -1;

		for ( int direction = 0; direction < 4; ++direction )
		{
			if ( direction == 0 && currentPix.x > 0 )
			{
				nextIndex = currentPix.x - 1 + currentPix.y * width;
			}
			else if ( direction == 1 && currentPix.x < width - 1 )
			{
				nextIndex = currentPix.x + 1 + currentPix.y * width;
			}
			else if ( direction == 2 && currentPix.y > 0 )
			{
				nextIndex = currentPix.x + ( currentPix.y - 1 ) * width;
			}
			else if ( direction == 3 && currentPix.y < height - 1 )
			{
				nextIndex = currentPix.x + ( currentPix.y + 1 ) * width;
			}
			else
			{
				continue;
			}

			if ( searchMask[ nextIndex ] == 1 )
			{
				continue;
			}

			//
			// Check this neighbour
			//
			//Core::Printf( "Checking neighbour ", nextIndex, ": dist=", dist[ nextIndex ], ".\n" );
			int stepDist = searchArea[ nextIndex ];

			if ( searchMask[ nextIndex ] == 1 )
			{
				//
				// Allow this step but make it very expensive
				//
				stepDist = stepDist + BONUS_VALUE;
				//continue;
			}

			if ( searchMask[ nextIndex ] == SOURCE_TARGET )
			{
				stepDist = 0;
			}
            else if ( searchBonus[ nextIndex ] == PENALTY_REGION )
            {
                stepDist = searchArea[ nextIndex ] + PENALTY_VALUE;
            }

			//Core::Printf( "stepDist=", stepDist, ".\n" );

			int altDist = dist[ currentIndex ] + stepDist;
			if ( altDist < dist[ nextIndex ] )
			{
				if ( prev[ nextIndex ] != -1 )
				{
					//
					// Remove previous node from the queue
					//
					distQueue.erase( distQueue.find( make_int2( nextIndex, dist[ nextIndex ] ) ) );
				}

				//
				// Insert new node into the queue
				//
				distQueue.insert( make_int2( nextIndex, altDist ) );

				dist[ nextIndex ] = altDist;
				prev[ nextIndex ] = currentIndex;

				//Core::Printf( "Updated distance to:", altDist, ".\n" );
			}

		}

		//++itCount;
		//if (1)//( itCount % 10 == 0 )
		//{
			//Core::Printf( "Finished Dijkstra iteration ", itCount, ": qsize=", distQueue.size(), "." );
		//}
	}

	if ( found_target )
	{
		//
		// Trace back from current Index
		//
		*toIndex = currentIndex;
        //currentIndex = prev[ currentIndex ];
		//Core::Printf( "Found target - tracing back." );
		int linePix = 0;
		int tempIndex;
		while ( currentIndex != fromIndex )
		{
			//Core::Printf( currentIndex, "dif=", currentIndex - prev[ currentIndex ]);
			tempIndex = currentIndex;
			currentIndex = prev[ currentIndex ];
			prev[ tempIndex ] = PATH_RESULT_VALUE;
			++linePix;
		}
		//Core::Printf( "Found ", linePix, " line pixels." );
	}
	else
	{
		Core::Printf( "Target not found." );
	}
}

}
}