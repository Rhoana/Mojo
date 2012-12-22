#include "FileSystemTileServer.hpp"

#include "Mojo.Core/Stl.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>

#include <time.h>

#include "Mojo.Core/OpenCV.hpp"
#include "Mojo.Core/ForEach.hpp"
#include "Mojo.Core/D3D11CudaTexture.hpp"
#include "Mojo.Core/ForEach.hpp"

#include "TileCacheEntry.hpp"
#include "SimpleSplitTools.hpp"

namespace Mojo
{
namespace Native
{

FileSystemTileServer::FileSystemTileServer( Core::PrimitiveMap constParameters ) :
	mConstParameters           ( constParameters ),
	mIsTiledDatasetLoaded      ( false ),
	mIsSegmentationLoaded      ( false )
{
    mSplitStepDist = 0;
    mSplitResultDist = 0;
    mSplitPrev = 0;
    mSplitSearchMask = 0;
    mSplitBonusArea = 0;
	mSplitBorderTargets = 0;
    mSplitResultArea = 0;
}

FileSystemTileServer::~FileSystemTileServer()
{
    if ( mSplitStepDist != 0 )
        delete[] mSplitStepDist;

    if ( mSplitResultDist != 0 )
        delete[] mSplitResultDist;

    if ( mSplitPrev != 0 )
        delete[] mSplitPrev;

    if ( mSplitSearchMask != 0 )
        delete[] mSplitSearchMask;

    if ( mSplitBonusArea != 0 )
        delete[] mSplitBonusArea;

	if ( mSplitBorderTargets != 0 )
		delete[] mSplitBorderTargets;

    if ( mSplitResultArea != 0 )
        delete[] mSplitResultArea;
}

//
// Dataset Methods
//

void FileSystemTileServer::LoadTiledDataset( TiledDatasetDescription& tiledDatasetDescription )
{
    switch ( tiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).dxgiFormat )
    {
    case DXGI_FORMAT_R8_UNORM:
        LoadTiledDatasetInternal< uchar1 >( tiledDatasetDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        break;
    }
}

void FileSystemTileServer::UnloadTiledDataset()
{
    UnloadTiledDatasetInternal();
}

bool FileSystemTileServer::IsTiledDatasetLoaded()
{
    return mIsTiledDatasetLoaded;
}

void FileSystemTileServer::LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription )
{
    switch ( tiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).dxgiFormat )
    {
    case DXGI_FORMAT_R8_UNORM:
        LoadSegmentationInternal< uchar1 >( tiledDatasetDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        break;
    }
}

void FileSystemTileServer::UnloadSegmentation()
{
    UnloadSegmentationInternal();
}

bool FileSystemTileServer::IsSegmentationLoaded()
{
    return mIsSegmentationLoaded;
}

//
// Edit Methods
//

void FileSystemTileServer::ReplaceSegmentationLabel( int oldId, int newId )
{
	if ( oldId != newId && mIsSegmentationLoaded )
    {
        Core::Printf( "\nReplacing segmentation label ", oldId, " with segmentation label ", newId, "...\n" );

		std::set< int4, Mojo::Core::Int4Comparator > tilesContainingOldId = mTiledDatasetDescription.idTileMap.Get( oldId );
		std::set< int4, Mojo::Core::Int4Comparator > tilesContainingNewId = mTiledDatasetDescription.idTileMap.Get( newId );

		PrepForNextUndoRedoChange();
		mUndoItem.newId = newId;
		mUndoItem.oldId = oldId;
		mUndoItem.idTileMapRemoveOldId.insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );
		mUndoItem.idTileMapAddNewId.insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );
		for ( std::set< int4, Mojo::Core::Int4Comparator >::iterator eraseIterator = tilesContainingNewId.begin(); eraseIterator != tilesContainingNewId.end(); ++eraseIterator )
		{
			mUndoItem.idTileMapAddNewId.erase( *eraseIterator );
		}

        for( std::set< int4, Mojo::Core::Int4Comparator >::iterator tileIndexi = tilesContainingOldId.begin(); tileIndexi != tilesContainingOldId.end(); ++tileIndexi )
        {
            int4 tileIndex = *tileIndexi;
            //
            // load tile
            //
            Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
            int* currentIdVolume    = (int*)volumeDescriptions.Get( "IdMap" ).data;

			//
			// Get or create the change bitset for this tile
			//
			std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > *changeBits = 
				&mUndoItem.changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];

            //
            // replace the old id and color with the new id and color...
            //
            int3 numVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
            for ( int zv = 0; zv < numVoxels.z; zv++ )
            {
                for ( int yv = 0; yv < numVoxels.y; yv++ )
                {
                    for ( int xv = 0; xv < numVoxels.x; xv++ )
                    {
                        int3 index3D = make_int3( xv, yv, zv );
                        int  index1D = Core::Index3DToIndex1D( index3D, numVoxels );
                        int  idValue = currentIdVolume[ index1D ];

                        if ( idValue == oldId )
                        {
                            currentIdVolume[ index1D ] = newId;
							changeBits->set( index1D );
                        }
                    }
                }
            }

            //
            // save tile to disk
            //
            //Core::Printf(
            //    "    Saving tile ",
            //    "w = ", Core::ToStringZeroPad( tileIndex.w, 8 ), ", ",
            //    "z = ", Core::ToStringZeroPad( tileIndex.z, 8 ), ", ",
            //    "y = ", Core::ToStringZeroPad( tileIndex.y, 8 ), ", ",
            //    "x = ", Core::ToStringZeroPad( tileIndex.x, 8 ) );

            SaveTile( tileIndex, volumeDescriptions );

            //
            // unload tile
            //
            UnloadTile( tileIndex );

        }

        //
        // add all the tiles containing old id to the list of tiles corresponding to the new id
        //
        mTiledDatasetDescription.idTileMap.Get( newId ).insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );

        //
        // completely remove old id from our id tile map, since the old id is no longer present in the segmentation 
        //
        mTiledDatasetDescription.idTileMap.GetHashMap().erase( oldId );

        Core::Printf( "\nFinished replacing segmentation label ", oldId, " with segmentation label ", newId, ".\n" );
    }
}

bool FileSystemTileServer::TileContainsId ( int3 numVoxelsPerTile, int3 currentIdNumVoxels, int* currentIdVolume, int segId )
{
    bool found = false;
    int maxIndex3D = Core::Index3DToIndex1D( make_int3( numVoxelsPerTile.x-1, numVoxelsPerTile.y-1, 0 ), currentIdNumVoxels );
    for (int i1D = 0; i1D < maxIndex3D; ++i1D )
    {
        if ( currentIdVolume[ i1D ] == segId )
        {
            found = true;
            break;
        }
    }
    return found;
}

void FileSystemTileServer::ReplaceSegmentationLabelCurrentSlice( int oldId, int newId, float3 pointTileSpace )
{
    if ( oldId != newId && mIsSegmentationLoaded )
    {
        int3 numVoxels = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels;
        int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
        int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;
        int3 pVoxelSpace = 
            make_int3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

        Core::Printf( "\nReplacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, "...\n" );

        std::set< int4, Mojo::Core::Int4Comparator > tilesContainingOldId = mTiledDatasetDescription.idTileMap.Get( oldId );
        std::set< int4, Mojo::Core::Int4Comparator > tilesContainingNewId = mTiledDatasetDescription.idTileMap.Get( newId );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		PrepForNextUndoRedoChange();
		mUndoItem.newId = newId;
		mUndoItem.oldId = oldId;
		std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > *changeBits;

		int4 previousTileIndex;
        bool tileLoaded = false;
        bool tileChanged = false;

        int currentW = 0;
        std::queue< int4 > tileQueue;
        std::multimap< int4, int4, Mojo::Core::Int4Comparator > sliceQueue;
        std::queue< int4 > wQueue;

        int* currentIdVolume;
        int3 currentIdNumVoxels;
        int4 thisVoxel;

        tileQueue.push( make_int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

        while ( currentW < numTiles.w )
        {
            while ( tileQueue.size() > 0 || sliceQueue.size() > 0 )
            {
                if ( tileQueue.size() > 0 )
                {
                    //
					// Keep to the same tile if possible
					//
                    thisVoxel = tileQueue.front();
                    tileQueue.pop();
                }
                else
                {
                    //
					// Start on the next tile
					//
                    thisVoxel = sliceQueue.begin()->second;
                    sliceQueue.erase( sliceQueue.begin() );
                }

                //
				// Find the tile for this pixel
				//
                int4 tileIndex = make_int4( thisVoxel.x / numVoxelsPerTile.x,
                    thisVoxel.y / numVoxelsPerTile.y,
                    thisVoxel.z / numVoxelsPerTile.z,
                    currentW);

                //
				// Load tile if necessary
				//
                if ( !tileLoaded || previousTileIndex.x != tileIndex.x ||
                    previousTileIndex.y != tileIndex.y ||
                    previousTileIndex.z != tileIndex.z ||
                    previousTileIndex.w != tileIndex.w )
                {
                    if ( tileLoaded )
                    {
                        //
						// Save and unload the previous tile
						//
                        if ( tileChanged )
                        {
                            SaveTile( previousTileIndex, volumeDescriptions );

                            //
							// Update the idTileMap
							//

							//
                            // Add this tile to the newId map
							//
							if ( tilesContainingNewId.find( previousTileIndex ) == tilesContainingNewId.end() )
							{
								mUndoItem.idTileMapAddNewId.insert ( previousTileIndex );
							}
                            tilesContainingNewId.insert( previousTileIndex );

                            //
							// Check if we can remove this tile from the oldId map
							//
                            if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, oldId ) )
                            {
                                tilesContainingOldId.erase( previousTileIndex );
								mUndoItem.idTileMapRemoveOldId.insert( previousTileIndex );
                            }

                        }
                        UnloadTile( previousTileIndex );
                    }

                    //
					// Load the current tile
					//
                    volumeDescriptions = LoadTile( tileIndex );
                    previousTileIndex = tileIndex;
                    currentIdVolume = (int*)volumeDescriptions.Get( "IdMap" ).data;
                    currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
                    tileLoaded = true;
                    tileChanged = false;

					//
					// Get or create the change bitset for this tile
					//
					changeBits = 
						&mUndoItem.changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
                }

                int tileX = thisVoxel.x % numVoxelsPerTile.x;
                int tileY = thisVoxel.y % numVoxelsPerTile.y;


                int3 index3D = make_int3( tileX, tileY, 0 );
                int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
                int  idValue = currentIdVolume[ index1D ];

                if ( idValue == oldId )
                {
                    currentIdVolume[ index1D ] = newId;
					changeBits->set( index1D );
                    tileChanged = true;

                    //
					// Only do flood fill at highest resolution - the rest is done with the wQueue
					//
                    if ( currentW == 0 )
                    {
                        //
						// Add neighbours to the appropriate queue
						//
                        if (thisVoxel.x > 0)
                        {
                            if (tileX > 0)
                            {
                                tileQueue.push( make_int4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< int4, int4 > (
                                    make_int4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    make_int4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.x < numVoxels.x - 1)
                        {
                            if (tileX < numVoxelsPerTile.x - 1)
                            {
                                tileQueue.push( make_int4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< int4, int4 > (
                                    make_int4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    make_int4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }
                        if (thisVoxel.y > 0)
                        {
                            if (tileY > 0)
                            {
                                tileQueue.push( make_int4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< int4, int4 > (
                                    make_int4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                                    make_int4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW ) ) );
                            }
                        }
                        if (thisVoxel.y < numVoxels.y - 1)
                        {
                            if (tileY < numVoxelsPerTile.y - 1)
                            {
                                tileQueue.push( make_int4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< int4, int4 > (
                                    make_int4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                                    make_int4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW ) ) );
                            }
                        }
                    }

                    //
					// Add a scaled-down w to the queue
					//
                    if (currentW < numTiles.w-1) wQueue.push( make_int4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
                }

            }
            std::swap(tileQueue, wQueue);
            ++currentW;
        }

        if ( tileLoaded )
        {
            //
			// Save and unload the previous tile
			//
            if ( tileChanged )
            {
                SaveTile( previousTileIndex, volumeDescriptions );

                //
				// Update the idTileMap
				//

                //
				// Add this tile to the newId map
				//
				if ( tilesContainingNewId.find( previousTileIndex ) == tilesContainingNewId.end() )
				{
					mUndoItem.idTileMapAddNewId.insert ( previousTileIndex );
				}
                tilesContainingNewId.insert( previousTileIndex );

                //
				// Check if we can remove this tile from the oldId map
				//
                if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, oldId ) )
                {
                    tilesContainingOldId.erase( previousTileIndex );
					mUndoItem.idTileMapRemoveOldId.insert( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
		// Update idTileMap
		//
        mTiledDatasetDescription.idTileMap.Set( oldId, tilesContainingOldId );
        mTiledDatasetDescription.idTileMap.Set( newId, tilesContainingNewId );

        Core::Printf( "\nFinished replacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, ".\n" );
    }
}

void FileSystemTileServer::ReplaceSegmentationLabelCurrentConnectedComponent( int oldId, int newId, float3 pointTileSpace )
{
    if ( oldId != newId && mIsSegmentationLoaded )
    {
        int3 numVoxels = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels;
        int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
        int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;
        int3 pVoxelSpace = 
            make_int3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

        Core::Printf( "\nReplacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, "...\n" );

        std::set< int4, Mojo::Core::Int4Comparator > tilesContainingOldId = mTiledDatasetDescription.idTileMap.Get( oldId );
        std::set< int4, Mojo::Core::Int4Comparator > tilesContainingNewId = mTiledDatasetDescription.idTileMap.Get( newId );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		PrepForNextUndoRedoChange();
		mUndoItem.newId = newId;
		mUndoItem.oldId = oldId;
		std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > *changeBits;

		int4 previousTileIndex;
        bool tileLoaded = false;
        bool tileChanged = false;

        int currentW = 0;
        std::queue< int4 > tileQueue;
        std::multimap< int4, int4, Mojo::Core::Int4Comparator > sliceQueue;
        std::queue< int4 > wQueue;

        int* currentIdVolume;
        int3 currentIdNumVoxels;
        int4 thisVoxel;

        tileQueue.push( make_int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

        while ( currentW < numTiles.w )
        {
            while ( tileQueue.size() > 0 || sliceQueue.size() )
            {
                if ( tileQueue.size() > 0 )
                {
                    //
					// Keep to the same tile if possible
					//
                    thisVoxel = tileQueue.front();
                    tileQueue.pop();
                }
                else if ( sliceQueue.size() > 0 )
                {
                    //
					// Start on the next tile
					//
                    thisVoxel = sliceQueue.begin()->second;
                    sliceQueue.erase( sliceQueue.begin() );
                }

                //
				// Find the tile for this pixel
				//
                int4 tileIndex = make_int4( thisVoxel.x / numVoxelsPerTile.x,
                    thisVoxel.y / numVoxelsPerTile.y,
                    thisVoxel.z / numVoxelsPerTile.z,
                    currentW);

                //
				// Load tile if necessary
				//
                if ( !tileLoaded || previousTileIndex.x != tileIndex.x ||
                    previousTileIndex.y != tileIndex.y ||
                    previousTileIndex.z != tileIndex.z ||
                    previousTileIndex.w != tileIndex.w )
                {
                    if ( tileLoaded )
                    {
                        //
						// Save and unload the previous tile
						//
                        if ( tileChanged )
                        {
                            SaveTile( previousTileIndex, volumeDescriptions );

                            //
							// Update the idTileMap
							//

                            //
							// Add this tile to the newId map
							//
							if ( tilesContainingNewId.find( previousTileIndex ) == tilesContainingNewId.end() )
							{
								mUndoItem.idTileMapAddNewId.insert ( previousTileIndex );
							}
                            tilesContainingNewId.insert( previousTileIndex );

                            //
							// Check if we can remove this tile from the oldId map
							//
                            if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, oldId ) )
                            {
                                tilesContainingOldId.erase( previousTileIndex );
								mUndoItem.idTileMapRemoveOldId.insert ( previousTileIndex );
                            }

                        }
                        UnloadTile( previousTileIndex );
                    }

                    //
					// Load the current tile
					//
                    volumeDescriptions = LoadTile( tileIndex );
                    previousTileIndex = tileIndex;
                    currentIdVolume = (int*)volumeDescriptions.Get( "IdMap" ).data;
                    currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
                    tileLoaded = true;
                    tileChanged = false;

					//
					// Get or create the change bitset for this tile
					//
					changeBits = 
						&mUndoItem.changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
                }

                int tileX = thisVoxel.x % numVoxelsPerTile.x;
                int tileY = thisVoxel.y % numVoxelsPerTile.y;

                int3 index3D = make_int3( tileX, tileY, 0 );
                int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
                int  idValue = currentIdVolume[ index1D ];

                if ( idValue == oldId )
                {
                    currentIdVolume[ index1D ] = newId;
					changeBits->set( index1D );
                    tileChanged = true;

                    //
					// Only do flood fill at highest resolution - the rest is done with the wQueue
					//
                    if ( currentW == 0 )
                    {
                        //
						// Add neighbours to the appropriate queue
						//
                        if (thisVoxel.x > 0)
                        {
                            if (tileX > 0)
                            {
                                tileQueue.push( make_int4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< int4, int4 > (
                                    make_int4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    make_int4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.x < numVoxels.x - 1)
                        {
                            if (tileX < numVoxelsPerTile.x - 1)
                            {
                                tileQueue.push( make_int4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< int4, int4 > (
                                    make_int4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    make_int4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.y > 0)
                        {
                            if (tileY > 0)
                            {
                                tileQueue.push( make_int4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< int4, int4 > (
                                    make_int4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                                    make_int4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.y < numVoxels.y - 1)
                        {
                            if (tileY < numVoxelsPerTile.y - 1)
                            {
                                tileQueue.push( make_int4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< int4, int4 > (
                                    make_int4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                                    make_int4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.z > 0)
                        {
                            sliceQueue.insert( std::pair< int4, int4 > (
                                make_int4( tileIndex.x, tileIndex.y, tileIndex.z - 1, tileIndex.w ),
                                make_int4( thisVoxel.x, thisVoxel.y, thisVoxel.z - 1, currentW ) ) );
                        }

                        if (thisVoxel.z < numVoxels.z - 1)
                        {
                            sliceQueue.insert( std::pair< int4, int4 > (
                                make_int4( tileIndex.x, tileIndex.y, tileIndex.z + 1, tileIndex.w ),
                                make_int4( thisVoxel.x, thisVoxel.y, thisVoxel.z + 1, currentW ) ) );
                        }
                    }

                    //
					// Add a scaled-down w to the queue
					//
                    if (currentW < numTiles.w-1) wQueue.push( make_int4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
                }

            }
            std::swap(tileQueue, wQueue);
            ++currentW;
        }

        if ( tileLoaded )
        {
            //
			// Save and unload the previous tile
			//
            if ( tileChanged )
            {
                SaveTile( previousTileIndex, volumeDescriptions );

                //
				// Update the idTileMap
				//

                //
				// Add this tile to the newId map
				//
				if ( tilesContainingNewId.find( previousTileIndex ) == tilesContainingNewId.end() )
				{
					mUndoItem.idTileMapAddNewId.insert ( previousTileIndex );
				}
                tilesContainingNewId.insert( previousTileIndex );

                //
				// Check if we can remove this tile from the oldId map
				//
                if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, oldId ) )
                {
                    tilesContainingOldId.erase( previousTileIndex );
					mUndoItem.idTileMapRemoveOldId.insert ( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
		// Update idTileMap
		//
        mTiledDatasetDescription.idTileMap.Set( oldId, tilesContainingOldId );
        mTiledDatasetDescription.idTileMap.Set( newId, tilesContainingNewId );

        Core::Printf( "\nFinished replacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, ".\n" );
    }
}

//
// 2D Split Methods
//


void FileSystemTileServer::DrawSplit( float3 pointTileSpace, float radius )
{
    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

    int3 pVoxelSpace = 
        make_int3(
        (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
        (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
        (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    int areaIndex = ( pVoxelSpace.x - mSplitWindowStart.x * numVoxelsPerTile.x ) +
        ( pVoxelSpace.y - mSplitWindowStart.y * numVoxelsPerTile.y ) * mSplitWindowWidth;

    SimpleSplitTools::ApplyCircleMask( areaIndex, mSplitWindowWidth, mSplitWindowHeight, BONUS_REGION, radius, mSplitBonusArea );

    UpdateSplitTiles();

    //Core::Printf( "\nDrew split circle voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, "), with radius ", radius, ".\n" );
}

void FileSystemTileServer::AddSplitSource( float3 pointTileSpace )
{
    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

    int3 pVoxelSpace = 
        make_int3(
        (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
        (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
        (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    //
    // Check for duplicates
    //
    bool duplicate = false;

    for ( int si = 0; si < mSplitSourcePoints.size(); ++si )
    {
        int3 existingPoint = mSplitSourcePoints[ si ];
        if ( existingPoint.x == pVoxelSpace.x && existingPoint.y == pVoxelSpace.y ) //ignore z
        {
            duplicate = true;
            break;
        }
    }

    if ( !duplicate )
    {
        mSplitSourcePoints.push_back( pVoxelSpace );
        Core::Printf( "\nAdded split point at voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ").\n" );
    }

}

void FileSystemTileServer::RemoveSplitSource()
{
    mSplitSourcePoints.pop_back();
    Core::Printf( "\nRemoved split point.\n" );
}

void FileSystemTileServer::UpdateSplitTilesHover()
{
	//
	// Export result to a layer
	//
    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

	Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

	unsigned int* splitData;

	for ( int xd = 0; xd < mSplitWindowTileSize.x; ++xd )
	{
		for (int yd = 0; yd < mSplitWindowTileSize.y; ++yd )
		{

			int4 tileIndex = make_int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );

			splitData = (unsigned int*) LoadTile( tileIndex ).Get( "OverlayMap" ).data;

			//
			// Copy result values into the tile
			//
			int xOffset = xd * numVoxelsPerTile.x;
			int yOffset = yd * numVoxelsPerTile.y;
			int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

			for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
			{
				int tileX = tileIndex1D % numVoxelsPerTile.x;
				int tileY = tileIndex1D / numVoxelsPerTile.x;
				int areaIndex1D = xOffset + tileX + ( yOffset + tileY ) * mSplitWindowWidth;

				splitData[ tileIndex1D ] = mSplitResultArea[ areaIndex1D ];

			}

			UnloadTile( tileIndex );

		}
	}
}

void FileSystemTileServer::UpdateSplitTiles()
{
		//
		// Export result to OverlayMap tiles
		//
        int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
		int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

		Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

        std::map< int4, int, Core::Int4Comparator > wQueue;

		int tileCount = 0;
		int sourceSplitCount = 0;
		unsigned int* splitData;

		for ( int xd = 0; xd < mSplitWindowTileSize.x; ++xd )
		{
			for (int yd = 0; yd < mSplitWindowTileSize.y; ++yd )
			{

				++tileCount;
				//Core::Printf( "Copying result tile ", tileCount, "." );

				int4 tileIndex = make_int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
				volumeDescriptions = LoadTile( tileIndex );
				Core::VolumeDescription splitVolumeDescription = volumeDescriptions.Get( "OverlayMap" );

				splitData = (unsigned int*) splitVolumeDescription.data;

				//
				// Copy result values into the tile
				//
				int xOffset = xd * numVoxelsPerTile.x;
				int yOffset = yd * numVoxelsPerTile.y;
				int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

				for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
				{
					int tileX = tileIndex1D % numVoxelsPerTile.x;
					int tileY = tileIndex1D / numVoxelsPerTile.x;
					int areaIndex1D = xOffset + tileX + ( yOffset + tileY ) * mSplitWindowWidth;

					RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );


                    if ( splitData[ tileIndex1D ] != mSplitResultArea[ areaIndex1D ] )
                    {
                        splitData[ tileIndex1D ] = mSplitResultArea[ areaIndex1D ];
                        wQueue[ make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitResultArea[ areaIndex1D ];
                    }
                    if ( mSplitBonusArea[ areaIndex1D ] && !mSplitResultArea[ areaIndex1D ] )
                    {
					    splitData[ tileIndex1D ] = 3;
						wQueue[ make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = 3;
                    }
				}

				UnloadTile( tileIndex );

			}
		}

		//
		// Fill up in w
		//

		bool tileLoaded = false;
        std::map< int4, int, Core::Int4Comparator >::iterator thisIt;
		int4 thisVoxel;
        int newValue;
		int4 tileIndex;
		int4 previousTileIndex;
		int3 currentTileNumVoxels;

		while ( wQueue.size() > 0 )
        {
            thisIt = wQueue.begin();
            thisVoxel = thisIt->first;
            newValue = thisIt->second;
            wQueue.erase( wQueue.begin() );

            tileIndex = make_int4( thisVoxel.x / numVoxelsPerTile.x,
                thisVoxel.y / numVoxelsPerTile.y,
                thisVoxel.z / numVoxelsPerTile.z,
                thisVoxel.w );

            //
			// Load tile if necessary
			//
            if ( !tileLoaded || previousTileIndex.x != tileIndex.x ||
                previousTileIndex.y != tileIndex.y ||
                previousTileIndex.z != tileIndex.z ||
                previousTileIndex.w != tileIndex.w )
            {
                if ( tileLoaded )
                {
                    UnloadTile( previousTileIndex );
                }

                //
				// Load the current tile
				//
                volumeDescriptions = LoadTile( tileIndex );
                previousTileIndex = tileIndex;
				splitData = (unsigned int*) volumeDescriptions.Get( "OverlayMap" ).data;
				currentTileNumVoxels = volumeDescriptions.Get( "OverlayMap" ).numVoxels;
                tileLoaded = true;

            }

            int tileX = thisVoxel.x % numVoxelsPerTile.x;
            int tileY = thisVoxel.y % numVoxelsPerTile.y;

            //if ( tileIndex.w == 1 )
            //{
            //    Core::Printf( "UpdateSplitTiles: updating voxel: ", thisVoxel.x, ",",  thisVoxel.y, ",",  thisVoxel.z, ",",  thisVoxel.w );
            //    Core::Printf( "=tile: ", tileIndex.x, ",",  tileIndex.y, ",",  tileIndex.z, ",",  tileIndex.w, " at ", tileX, ",", tileY, "." );
            //}

            int3 index3D = make_int3( tileX, tileY, 0 );
            int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentTileNumVoxels );

			//Core::Printf( "Writing to index1D=", index1D, " ( thisVoxel.x=", thisVoxel.x, " thisVoxel.y=", thisVoxel.y, " thisVoxel.z=", thisVoxel.z, " thisVoxel.w=", thisVoxel.w, " ).");

            if ( splitData[ index1D ] != newValue )
            {
                splitData[ index1D ] = newValue;

                //
			    // Add a scaled-down w to the queue
			    //
                if (thisVoxel.w < numTiles.w - 1)
                {
                    wQueue[ make_int4( thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, thisVoxel.w + 1 ) ] = newValue;
                }
            }

		}

		if ( tileLoaded )
		{
			UnloadTile( previousTileIndex );
		}

}

void FileSystemTileServer::ResetSplitTiles()
{
		//
		// Reset the overlay layer to zero
		//
        int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
		int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

		Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		std::queue< int4 > wQueue;

		int tileCount = 0;
		int sourceSplitCount = 0;
		unsigned int* splitData;

		for ( int xd = 0; xd < mSplitWindowTileSize.x; ++xd )
		{
			for (int yd = 0; yd < mSplitWindowTileSize.y; ++yd )
			{

				++tileCount;
				Core::Printf( "Copying result tile ", tileCount, "." );

				int4 tileIndex = make_int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
				volumeDescriptions = LoadTile( tileIndex );
				Core::VolumeDescription splitVolumeDescription = volumeDescriptions.Get( "OverlayMap" );

				splitData = (unsigned int*) splitVolumeDescription.data;

				//
				// Copy result values into the tile
				//
				int xOffset = xd * numVoxelsPerTile.x;
				int yOffset = yd * numVoxelsPerTile.y;
				int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

				for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
				{
					int tileX = tileIndex1D % numVoxelsPerTile.x;
					int tileY = tileIndex1D / numVoxelsPerTile.x;
					int areaIndex1D = xOffset + tileX + ( tileY ) * mSplitWindowWidth;

					RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

					if ( splitData[ tileIndex1D ] != 0 )
					{
						splitData[ tileIndex1D ] = 0;
						wQueue.push( make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) );
					}
				}

				UnloadTile( tileIndex );

			}
		}

		//
		// Fill up in w
		//

		bool tileLoaded = false;
		int4 thisVoxel;
		int4 tileIndex;
		int4 previousTileIndex;
		int3 currentTileNumVoxels;

		while ( wQueue.size() > 0 )
        {
            thisVoxel = wQueue.front();
            wQueue.pop();

            tileIndex = make_int4( thisVoxel.x / numVoxelsPerTile.x,
                thisVoxel.y / numVoxelsPerTile.y,
                thisVoxel.z / numVoxelsPerTile.z,
                thisVoxel.w );

            //
			// Load tile if necessary
			//
            if ( !tileLoaded || previousTileIndex.x != tileIndex.x ||
                previousTileIndex.y != tileIndex.y ||
                previousTileIndex.z != tileIndex.z ||
                previousTileIndex.w != tileIndex.w )
            {
                if ( tileLoaded )
                {
                    UnloadTile( previousTileIndex );
                }

                //
				// Load the current tile
				//
                volumeDescriptions = LoadTile( tileIndex );
                previousTileIndex = tileIndex;
				splitData = (unsigned int*) volumeDescriptions.Get( "OverlayMap" ).data;
				currentTileNumVoxels = volumeDescriptions.Get( "OverlayMap" ).numVoxels;
                tileLoaded = true;

            }

            int tileX = thisVoxel.x % numVoxelsPerTile.x;
            int tileY = thisVoxel.y % numVoxelsPerTile.y;

            //if ( tileIndex.w == 1 )
            //{
            //    Core::Printf( "ResetSplitTiles: updating voxel: ", thisVoxel.x, ",",  thisVoxel.y, ",",  thisVoxel.z, ",",  thisVoxel.w );
            //    Core::Printf( "=tile: ", tileIndex.x, ",",  tileIndex.y, ",",  tileIndex.z, ",",  tileIndex.w, " at ", tileX, ",", tileY, "." );
            //}

            int3 index3D = make_int3( tileX, tileY, 0 );
            int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentTileNumVoxels );

			//Core::Printf( "Writing to index1D=", index1D, " ( thisVoxel.x=", thisVoxel.x, " thisVoxel.y=", thisVoxel.y, " thisVoxel.z=", thisVoxel.z, " thisVoxel.w=", thisVoxel.w, " ).");

            if ( splitData[ index1D ] != 0 )
            {
                splitData[ index1D ] = 0;

                //
		        // Add a scaled-down w to the queue
		        //
                if (thisVoxel.w < numTiles.w - 1) wQueue.push( make_int4( thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, thisVoxel.w + 1 ) );
            }

		}

		if ( tileLoaded )
		{
			UnloadTile( previousTileIndex );
		}

}

void FileSystemTileServer::ResetSplitState()
{

    mSplitSourcePoints.clear();

    for ( int i = 0; i < mSplitWindowNPix; ++i )
    {
		//mSplitSearchMask[ i ] = 0;
        //mSplitResultArea[ i ] = 0;
		mSplitSearchMask[ i ] = mSplitBorderTargets[ i ];
        mSplitResultArea[ i ] = 0;
        mSplitBonusArea[ i ] = 0;
    }

	ResetSplitTiles();

    Core::Printf( "Reset Split State.\n");

}

void FileSystemTileServer::PrepForSplit( int segId, int zIndex )
{
    //
    // Find the size of this segment and load the bounding box of tiles
    //

    if ( mIsSegmentationLoaded )
    {

        Core::Printf( "\nPreparing for split of segment ", segId, " at z=", zIndex, ".\n" );

        int3 numVoxels = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels;
        int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
        int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

        int minTileX = numTiles.x;
        int maxTileX = 0;
        int minTileY = numTiles.y;
        int maxTileY = 0;

        std::set< int4, Mojo::Core::Int4Comparator > tilesContainingSegId = mTiledDatasetDescription.idTileMap.Get( segId );

        for ( std::set< int4, Mojo::Core::Int4Comparator >::iterator tileIterator = tilesContainingSegId.begin(); tileIterator != tilesContainingSegId.end(); ++tileIterator )
        {
            if ( tileIterator->z == zIndex && tileIterator->w == 0 )
            {
                //Include this tile
                minTileX = MIN( minTileX, tileIterator->x );
                maxTileX = MAX( maxTileX, tileIterator->x );
                minTileY = MIN( minTileY, tileIterator->y );
                maxTileY = MAX( maxTileY, tileIterator->y );
            }
        }

        if ( minTileX > maxTileX || minTileY > maxTileY )
		{
			//
			// No tile in this z section - ignore
			//
			return;
		}

        //
        // Calculate sizes
        //
        mSplitWindowStart = make_int3( minTileX, minTileY, zIndex );
        mSplitWindowTileSize = make_int3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

        Core::Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
        Core::Printf( "mSplitWindowSize=", mSplitWindowTileSize.x, ":", mSplitWindowTileSize.y, ":", mSplitWindowTileSize.z, ".\n" );

		mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowTileSize.x;
		mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowTileSize.y;
		mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

        Core::Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

		mSplitLabelCount = 0;

        //
        // Allocate working space (free if necessary)
        //

        if ( mSplitStepDist != 0 )
            delete[] mSplitStepDist;

        if ( mSplitResultDist != 0 )
            delete[] mSplitResultDist;

        if ( mSplitPrev != 0 )
            delete[] mSplitPrev;

        if ( mSplitSearchMask != 0 )
            delete[] mSplitSearchMask;

        if ( mSplitBonusArea != 0 )
            delete[] mSplitBonusArea;

		if ( mSplitBorderTargets != 0 )
			delete[] mSplitBorderTargets;

        if ( mSplitResultArea != 0 )
            delete[] mSplitResultArea;

		mSplitStepDist = new int[ mSplitWindowNPix ];
        mSplitResultDist = new int[ mSplitWindowNPix ];
		mSplitPrev = new int[ mSplitWindowNPix ];
		mSplitSearchMask = new int[ mSplitWindowNPix ];
		mSplitBonusArea = new int[ mSplitWindowNPix ];
		mSplitBorderTargets = new int[ mSplitWindowNPix ];
		mSplitResultArea = new unsigned int[ mSplitWindowNPix ];

        //
        // Load distances
        //
        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

        unsigned char* currentSrcVolume;
        int* currentIdVolume;

		int tileCount = 0;
		int areaCount = 0;
		for ( int xd = 0; xd < mSplitWindowTileSize.x; ++xd )
		{
			for (int yd = 0; yd < mSplitWindowTileSize.y; ++yd )
			{

				++tileCount;
				Core::Printf( "Loading distance tile ", tileCount, "." );

				int4 tileIndex = make_int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
				volumeDescriptions = LoadTile( tileIndex );
				currentSrcVolume = (unsigned char*)volumeDescriptions.Get( "SourceMap" ).data;
				currentIdVolume = (int*)volumeDescriptions.Get( "IdMap" ).data;

				//
				// Copy distance values into the buffer
				//
				int xOffset = xd * numVoxelsPerTile.x;
				int yOffset = yd * numVoxelsPerTile.y;
				int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

				for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
				{
                    int areaX = xOffset + tileIndex1D % numVoxelsPerTile.x;
                    int areaY = yOffset + tileIndex1D / numVoxelsPerTile.x;
					int areaIndex1D = areaX + areaY * mSplitWindowWidth;

					RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

					//
					// Distance calculation
					//
					int segVal = ((int) currentSrcVolume[ tileIndex1D ]) + 10;
					mSplitStepDist[ areaIndex1D ] = segVal * segVal + BONUS_VALUE;
					++areaCount;

                    //
                    // Mark border targets
                    //
					if ( currentIdVolume[ tileIndex1D ] == segId )
					{
						++mSplitLabelCount;
                        mSplitBorderTargets[ areaIndex1D ] = 0;
					}
					else if ( currentIdVolume[ tileIndex1D ] != 0 ||
                        areaX == 0 || areaX == mSplitWindowWidth - 1 || areaY == 0 || areaY == mSplitWindowHeight - 1 )
					{
                        mSplitBorderTargets[ areaIndex1D ] = BORDER_TARGET;
					}
                    else
                    {
                        mSplitBorderTargets[ areaIndex1D ] = 0;
                    }
				}

				UnloadTile( tileIndex );

			}
		}

        ResetSplitState();

		Core::Printf( "Loaded: areaCount=", areaCount );

        Core::Printf( "\nFinished preparing for split of segment ", segId, " at z=", zIndex, ".\n" );
    }

}

int FileSystemTileServer::CompleteSplit( int segId )
{
	int newId = 0;

	if ( mIsSegmentationLoaded && mSplitSourcePoints.size() > 0 )
    {

		int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

		//
		// Find a seed point close to the first split point
		//

        int areaIndex1D = ( mSplitSourcePoints[ 0 ].x - mSplitWindowStart.x * numVoxelsPerTile.x ) +
            ( mSplitSourcePoints[ 0 ].y - mSplitWindowStart.y * numVoxelsPerTile.y ) * mSplitWindowWidth;

		bool seedFound = false;
		int seedIndex1D;

		for ( int step = 1; !seedFound && step <= 10; ++step )
		{
			for ( int xd = -1; !seedFound && xd <= 1; ++xd )
			{
				for ( int yd = -1; !seedFound && yd <= 1; ++yd )
				{
					seedIndex1D = areaIndex1D + step * xd + step * yd * mSplitWindowWidth;
					if ( seedIndex1D >= 0 && seedIndex1D < mSplitWindowNPix && mSplitResultArea[ seedIndex1D ] == 0 )
					{
						seedFound = true;
						Core::Printf( "Seed found at:", seedIndex1D, "." );
					}
				}
			}
		}

		if ( !seedFound )
		{
			Core::Printf( "WARNING: Could not find seed point - aborting." );
			ResetSplitState();
			return 0;
		}

		//
		// Perform a 2D Flood-fill ( no changes yet, just record bits in the UndoItem )
		//

		int3 pVoxelSpace = 
			make_int3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
			seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
			mSplitWindowStart.z );

		int3 numVoxels = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels;
		int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

        Core::Printf( "\nSplitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ")...\n" );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		PrepForNextUndoRedoChange();

		std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > *changeBits;

		int4 previousTileIndex;
        bool tileLoaded = false;

        std::queue< int4 > tileQueue;
        std::multimap< int4, int4, Mojo::Core::Int4Comparator > sliceQueue;
		std::queue< int4 > wQueue;

        int* currentIdVolume;
        int3 currentIdNumVoxels;
        int4 thisVoxel;

		int nPixChanged = 0;
		bool invert = false;

        tileQueue.push( make_int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

        Core::Printf( "Filling at w=0." );

        while ( tileQueue.size() > 0 || sliceQueue.size() > 0 )
        {
            if ( tileQueue.size() > 0 )
            {
                //
				// Keep to the same tile if possible
				//
                thisVoxel = tileQueue.front();
                tileQueue.pop();
            }
            else
            {
                //
				// Start on the next tile
				//
                thisVoxel = sliceQueue.begin()->second;
                sliceQueue.erase( sliceQueue.begin() );
            }

            //
			// Find the tile for this pixel
			//
            int4 tileIndex = make_int4( thisVoxel.x / numVoxelsPerTile.x,
                thisVoxel.y / numVoxelsPerTile.y,
                thisVoxel.z / numVoxelsPerTile.z,
                0);

            //
			// Load tile if necessary
			//
            if ( !tileLoaded || previousTileIndex.x != tileIndex.x ||
                previousTileIndex.y != tileIndex.y ||
                previousTileIndex.z != tileIndex.z ||
                previousTileIndex.w != tileIndex.w )
            {
                if ( tileLoaded )
                {
                    //
					// Unload the previous tile
					//
                    UnloadTile( previousTileIndex );
                }

                //
				// Load the current tile
				//
                volumeDescriptions = LoadTile( tileIndex );
                previousTileIndex = tileIndex;
                currentIdVolume = (int*)volumeDescriptions.Get( "IdMap" ).data;
                currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
                tileLoaded = true;

				//
				// Get or create the change bitset for this tile
				//
				changeBits = &mUndoItem.changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
            }

            int tileX = thisVoxel.x % numVoxelsPerTile.x;
            int tileY = thisVoxel.y % numVoxelsPerTile.y;

            int3 index3D = make_int3( tileX, tileY, 0 );
            int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
            int  idValue = currentIdVolume[ index1D ];

			bool isSplitBorder = false;

			areaIndex1D = thisVoxel.x - mSplitWindowStart.x * numVoxelsPerTile.x +
				(thisVoxel.y - mSplitWindowStart.y * numVoxelsPerTile.y) * mSplitWindowWidth;
			if ( areaIndex1D >= 0 && areaIndex1D < mSplitWindowNPix )
			{
				isSplitBorder = mSplitResultArea[ areaIndex1D ] != 0;
			}

			if ( idValue == segId && !isSplitBorder && !changeBits->test( index1D ) )
            {
				changeBits->set( index1D );
				wQueue.push( thisVoxel );
				++nPixChanged;

                //
				// Add neighbours to the appropriate queue
				//
                if (thisVoxel.x > 0)
                {
                    if (tileX > 0)
                    {
                        tileQueue.push( make_int4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< int4, int4 > (
                            make_int4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                            make_int4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0 ) ) );
                    }
                }

                if (thisVoxel.x < numVoxels.x - 1)
                {
                    if (tileX < numVoxelsPerTile.x - 1)
                    {
                        tileQueue.push( make_int4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< int4, int4 > (
                            make_int4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                            make_int4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0 ) ) );
                    }
                }
                if (thisVoxel.y > 0)
                {
                    if (tileY > 0)
                    {
                        tileQueue.push( make_int4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< int4, int4 > (
                            make_int4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                            make_int4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0 ) ) );
                    }
                }
                if (thisVoxel.y < numVoxels.y - 1)
                {
                    if (tileY < numVoxelsPerTile.y - 1)
                    {
                        tileQueue.push( make_int4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< int4, int4 > (
                            make_int4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                            make_int4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0 ) ) );
                    }
                }
            }

			//
			// Check for inversion ( only re-label the smallest segment )
			//
			if ( tileQueue.size() == 0 && sliceQueue.size() == 0 && nPixChanged > 1 + mSplitLabelCount / 2 && !invert )
			{
				//
				// This fill is too big - find an alternative fill point
				//

				wQueue = std::queue< int4 >();

				if ( tileLoaded )
				{
					//
					// Unload the previous tile
					//
					UnloadTile( previousTileIndex );
					tileLoaded = false;
				}

				seedFound = false;

				//
				// Find a new seed point
				// (any point next to the border line that hasn't been filled)
				//

				for ( int xd = 0; !seedFound && xd < mSplitWindowTileSize.x; ++xd )
				{
					for (int yd = 0; !seedFound && yd < mSplitWindowTileSize.y; ++yd )
					{
						tileIndex = make_int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
						volumeDescriptions = LoadTile( tileIndex );
						currentIdVolume = (int*)volumeDescriptions.Get( "IdMap" ).data;

						int xOffset = xd * numVoxelsPerTile.x;
						int yOffset = yd * numVoxelsPerTile.y;
						int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

						for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
						{
							int areaX = xOffset + tileIndex1D % numVoxelsPerTile.x;
							int areaY = yOffset + tileIndex1D / numVoxelsPerTile.x;
							seedIndex1D = areaX + areaY * mSplitWindowWidth;

							if ( currentIdVolume[ tileIndex1D ] == segId && !mUndoItem.changePixels.GetHashMap()[ CreateTileString( tileIndex ) ].test( tileIndex1D ) && mSplitResultArea[ seedIndex1D ] == 0 )
							{
								//
								// Check neighbours
								//
								if ( ( areaX > 0 && mSplitResultArea[ seedIndex1D - 1 ] != 0 ) ||
									( areaX < mSplitWindowWidth - 1 && mSplitResultArea[ seedIndex1D + 1 ] != 0 ) ||
									( areaY > 0 && mSplitResultArea[ seedIndex1D - mSplitWindowWidth ] != 0 ) ||
									( areaY < mSplitWindowHeight - 1 && mSplitResultArea[ seedIndex1D + mSplitWindowWidth ] != 0 )
									)
								{
									seedFound = true;
									break;
								}
							}
						}

						UnloadTile( tileIndex );

					}
				}

				//
				// Reset the Undo Item
				//
				mUndoItem = FileSystemUndoRedoItem();

				if ( !seedFound )
				{
					Core::Printf( "WARNING: Could not find (inverted) seed point - aborting." );
					ResetSplitState();
					return 0;
				}
				else
				{
					Core::Printf( "Seed found at:", seedIndex1D, "." );
				}

				//
				// Use this new seed point
				//
				pVoxelSpace = 
					make_int3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
					seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
					mSplitWindowStart.z );

				nPixChanged = 0;
				invert = true;

				tileQueue.push( make_int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

				Core::Printf( "Filling (invert) at w=0." );

			}
        }

        if ( tileLoaded )
        {
            //
			// Unload the previous tile
			//
            UnloadTile( previousTileIndex );
			tileLoaded = false;
        }

		//
		// Perform the split at w=0 ( re-label the smallest segment )
		//

		newId = ++mTiledDatasetDescription.maxLabelId;
        std::set< int4, Mojo::Core::Int4Comparator > tilesContainingOldId = mTiledDatasetDescription.idTileMap.Get( segId );
        std::set< int4, Mojo::Core::Int4Comparator > tilesContainingNewId;

		mUndoItem.newId = newId;
		mUndoItem.oldId = segId;

		bool tileChanged;

        Core::Printf( "Splitting (invert=", invert, ")." );

		//
		// Do the split and fill up in w
		//

        std::swap(tileQueue, wQueue);
		int currentW = 0;

		tileLoaded = false;
		tileChanged = false;

		while ( currentW < numTiles.w )
		{
			Core::Printf( "Splitting at w=", currentW, "." );
			while ( tileQueue.size() > 0 )
			{
				thisVoxel = tileQueue.front();
				tileQueue.pop();

				//
				// Find the tile for this pixel
				//
				int4 tileIndex = make_int4( thisVoxel.x / numVoxelsPerTile.x,
					thisVoxel.y / numVoxelsPerTile.y,
					thisVoxel.z / numVoxelsPerTile.z,
					currentW);

				//
				// Load tile if necessary
				//
				if ( !tileLoaded || previousTileIndex.x != tileIndex.x ||
					previousTileIndex.y != tileIndex.y ||
					previousTileIndex.z != tileIndex.z ||
					previousTileIndex.w != tileIndex.w )
				{
					if ( tileLoaded )
					{
						//
						// Save and unload the previous tile
						//
						if ( tileChanged )
						{
							SaveTile( previousTileIndex, volumeDescriptions );

							//
							// Update the idTileMap
							//

							//
							// Add this tile to the newId map
							//
							if ( tilesContainingNewId.find( previousTileIndex ) == tilesContainingNewId.end() )
							{
								mUndoItem.idTileMapAddNewId.insert ( previousTileIndex );
							}
							tilesContainingNewId.insert( previousTileIndex );

							//
							// Check if we can remove this tile from the oldId map
							//
							if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, segId ) )
							{
								tilesContainingOldId.erase( previousTileIndex );
								mUndoItem.idTileMapRemoveOldId.insert( previousTileIndex );
							}

						}
						UnloadTile( previousTileIndex );
					}

					//
					// Load the current tile
					//
					volumeDescriptions = LoadTile( tileIndex );
					previousTileIndex = tileIndex;
					currentIdVolume = (int*)volumeDescriptions.Get( "IdMap" ).data;
					currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
					tileLoaded = true;
					tileChanged = false;

					//
					// Get or create the change bitset for this tile
					//
					changeBits = 
						&mUndoItem.changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
				}
				int tileX = thisVoxel.x % numVoxelsPerTile.x;
				int tileY = thisVoxel.y % numVoxelsPerTile.y;

				int3 index3D = make_int3( tileX, tileY, 0 );
				int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
				int  idValue = currentIdVolume[ index1D ];

				if ( idValue == segId )
				{
					currentIdVolume[ index1D ] = newId;
					changeBits->set( index1D );
					tileChanged = true;

					//
					// Add a scaled-down w to the queue
					//
					if (currentW < numTiles.w-1) wQueue.push( make_int4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
				}

			}
			std::swap(tileQueue, wQueue);
			++currentW;
		}

        if ( tileLoaded )
        {
            //
			// Save and unload the previous tile
			//
            if ( tileChanged )
            {
                SaveTile( previousTileIndex, volumeDescriptions );

                //
				// Update the idTileMap
				//

                //
				// Add this tile to the newId map
				//
				if ( tilesContainingNewId.find( previousTileIndex ) == tilesContainingNewId.end() )
				{
					mUndoItem.idTileMapAddNewId.insert ( previousTileIndex );
				}
                tilesContainingNewId.insert( previousTileIndex );

                //
				// Check if we can remove this tile from the oldId map
				//
                if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, segId ) )
                {
                    tilesContainingOldId.erase( previousTileIndex );
					mUndoItem.idTileMapRemoveOldId.insert( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
		// Update idTileMap
		//
        mTiledDatasetDescription.idTileMap.Set( segId, tilesContainingOldId );
        mTiledDatasetDescription.idTileMap.Set( newId, tilesContainingNewId );

        Core::Printf( "\nFinished Splitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") to new segmentation label ", newId, "...\n" );
    }

	//
	// Prep for more splitting
	//
	PrepForSplit( segId, mSplitWindowStart.z );

	return newId;

}

void FileSystemTileServer::FindSplitLine2DHover( int segId, float3 pointTileSpace )
{
    //
    // This was too slow - is there a faster way?
    //
}

void FileSystemTileServer::FindSplitLine2D( int segId )
{
	//
	// Find a splitting line that links all the given points
	// Save the result into a temp tile
	//
    if ( mIsSegmentationLoaded && mSplitSourcePoints.size() > 0 )
    {
		Core::Printf( "\nFinding Split line for segment ", segId, " with ", mSplitSourcePoints.size(), " split segments.\n" );

        int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

        int nSources = mSplitSourcePoints.size();
        int* sourceLinks = new int[ nSources ];
        int* sourceLocations = new int[ nSources ];

        for ( int si = 0; si < nSources; ++si )
        {
            sourceLinks[ si ] = 0;
            sourceLocations[ si ] = ( mSplitSourcePoints[ si ].x - mSplitWindowStart.x * numVoxelsPerTile.x ) +
                ( mSplitSourcePoints[ si ].y - mSplitWindowStart.y * numVoxelsPerTile.y ) * mSplitWindowWidth;

    		//Core::Printf( "Split point at:", mSplitSourcePoints[ si ].x, ":", mSplitSourcePoints[ si ].y, " index=", sourceLocations[ si ], ".\n" );
        }

        int nLinks = 0;
        int direction = 1;
        int targetMax = BORDER_TARGET;
        int currentSourceIx = 0;

		//
		// Reset mask area
		//
		for ( int i = 0; i < mSplitWindowNPix; ++i )
        {
			mSplitSearchMask[ i ] = mSplitBorderTargets[ i ];
            mSplitResultArea[ i ] = 0;
        }

		//ResetSplitTiles();

		while ( nLinks < nSources * 2 )
		{

            int sourceIndex = sourceLocations[ currentSourceIx ];

            // Set up the targets
            for ( int si = 0; si < nSources; ++si )
            {
                int areaIndex = sourceLocations[ si ];

                //
                // Do not target visited nodes or the current source
                //
                if ( si == currentSourceIx || sourceLinks[ si ] >= 1 )
                {
                    mSplitSearchMask[ areaIndex ] = 1;
					//Core::Printf( "Marked source (blocked) at ", areaIndex, ".\n" );
                }
                else
                {
                    mSplitSearchMask[ areaIndex ] = SOURCE_TARGET;
					//Core::Printf( "Marked source (ok) at ", areaIndex, ".\n" );
                }

            }

            if ( direction == 1 )
            {
                targetMax = BORDER_TARGET;
                //
                // Allow loops
                //
                if ( nSources > 2 && nLinks == nSources * 2 - 2 )
                {
                    mSplitSearchMask[ sourceLocations[ 0 ] ] = SOURCE_TARGET;
                }
            }
            else
            {
                //
                // link to border only on the last step
                //
                targetMax = nLinks == nSources * 2 - 1 ? BORDER_TARGET : SOURCE_TARGET;
            }

            int toIndex = -1;

            Mojo::Native::SimpleSplitTools::DijkstraSearch( mSplitStepDist, mSplitSearchMask, mSplitBonusArea, sourceIndex, mSplitWindowWidth, mSplitWindowHeight, targetMax, mSplitResultDist, mSplitPrev, &toIndex );

			if ( toIndex != -1 )
			{

			    //
			    // Get the result line and mask around it
			    //
				for ( int i = 0; i < mSplitWindowNPix; ++i )
				{
					if ( mSplitPrev[ i ] == PATH_RESULT_VALUE && mSplitResultArea[ i ] == 0 )
					{
						mSplitResultArea[ i ] = 1;
                        SimpleSplitTools::ApplySmallMask( i, mSplitWindowWidth, mSplitWindowHeight, 1, mSplitSearchMask );
					}
				}

				//
				// Unmask the sources
				//
                for ( int si = 0; si < nSources; ++si )
                {
                    int areaIndex = sourceLocations[ si ];
					mSplitResultArea[ areaIndex ] = 2;
                    SimpleSplitTools::ApplyLargeMask( areaIndex, mSplitWindowWidth, mSplitWindowHeight, 0, mSplitSearchMask );
					mSplitSearchMask [ areaIndex ] = SOURCE_TARGET;
                }

				++nLinks;
                ++sourceLinks[ currentSourceIx ];

                if ( mSplitSearchMask[ toIndex ] == SOURCE_TARGET )
                {
                    for ( int si = 0; si < nSources; ++si )
                    {
                        if ( sourceLocations[ si ] == toIndex )
                        {
                            ++nLinks;
                            ++sourceLinks[ si ];
                            currentSourceIx = si;
                            break;
                        }
                    }
                }
                else
                {
                    direction = 2;
                    currentSourceIx = 0;
                }

			}
			else
			{
				Core::Printf( "WARNING: Could not find shortest path in FindSplitLine2D." );
				break;
			}

		}

		UpdateSplitTiles();

        delete[] sourceLinks;
        delete[] sourceLocations;

		Core::Printf( "\nFinished splitting label ", segId, ".\n" );

	}
}


//
// Undo / Redo Methods
//

void FileSystemTileServer::UndoChange()
{
	//
	// If we are in the middle of the split reset the state (no redo)
	//
	if ( mSplitSourcePoints.size() > 0 )
	{
		ResetSplitState();
		mRedoItem = FileSystemUndoRedoItem();
		return;
	}

	int oldId = mUndoItem.oldId;
	int newId = mUndoItem.newId;

	if ( newId != 0 && oldId != 0 && newId != oldId && mIsSegmentationLoaded )
	{
        Core::Printf( "\nUndo operation: changing segmentation label ", newId, " back to segmentation label ", oldId, "...\n" );
		stdext::hash_map < std::string, std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > >::iterator changeIt;

		for ( changeIt = mUndoItem.changePixels.GetHashMap().begin(); changeIt != mUndoItem.changePixels.GetHashMap().end(); ++changeIt )
		{

			int4 tileIndex = CreateTileIndex( changeIt->first );

            //
            // load tile
            //
            Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
            int* currentIdVolume    = (int*)volumeDescriptions.Get( "IdMap" ).data;

			//
			// Get or create the change bitset for this tile
			//
			std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > *changeBits = &changeIt->second;

            //
            // replace the old id and color with the new id and color...
            //
            int3 numVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
            for ( int zv = 0; zv < numVoxels.z; zv++ )
            {
                for ( int yv = 0; yv < numVoxels.y; yv++ )
                {
                    for ( int xv = 0; xv < numVoxels.x; xv++ )
                    {
                        int3 index3D = make_int3( xv, yv, zv );
                        int  index1D = Core::Index3DToIndex1D( index3D, numVoxels );

						if ( changeBits->test( index1D ) )
						{
							currentIdVolume[ index1D ] = oldId;
						}
                    }
                }
            }

            //
            // save tile
            //
            SaveTile( tileIndex, volumeDescriptions );

            //
            // unload tile
            //
            UnloadTile( tileIndex );

        }

        //
        // remove newly added tiles from the "newId" idTileMap
        //
		for ( std::set< int4, Mojo::Core::Int4Comparator >::iterator eraseIterator = mUndoItem.idTileMapAddNewId.begin(); eraseIterator != mUndoItem.idTileMapAddNewId.end(); ++eraseIterator )
		{
			mTiledDatasetDescription.idTileMap.Get( newId ).erase( *eraseIterator );
		}
		//
		// put removed tiles back into the "oldId" idTileMap (create a new idTileMap if necessary)
		//
		mTiledDatasetDescription.idTileMap.GetHashMap()[ oldId ].insert( mUndoItem.idTileMapRemoveOldId.begin(), mUndoItem.idTileMapRemoveOldId.end() );

        Core::Printf( "\nUndo operation complete: changed segmentation label ", newId, " back to segmentation label ", oldId, ".\n" );

		//
		// Make this a redo item
		//
		mRedoItem = mUndoItem;
		mUndoItem = FileSystemUndoRedoItem();
	}
}

void FileSystemTileServer::RedoChange()
{
	int oldId = mRedoItem.oldId;
	int newId = mRedoItem.newId;

	if ( newId != 0 && oldId != 0 && newId != oldId && mIsSegmentationLoaded )
	{
        Core::Printf( "\nRedo operation: changing segmentation label ", oldId, " back to segmentation label ", newId, "...\n" );
		stdext::hash_map < std::string, std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > >::iterator changeIt;

		for ( changeIt = mRedoItem.changePixels.GetHashMap().begin(); changeIt != mRedoItem.changePixels.GetHashMap().end(); ++changeIt )
		{

			int4 tileIndex = CreateTileIndex( changeIt->first );

            //
            // load tile
            //
            Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
            int* currentIdVolume    = (int*)volumeDescriptions.Get( "IdMap" ).data;

			//
			// Get or create the change bitset for this tile
			//
			std::bitset< FILE_SYSTEM_TILE_CACHE_SIZE * FILE_SYSTEM_TILE_CACHE_SIZE > *changeBits = &changeIt->second;

            //
            // replace the old id and color with the new id and color...
            //
            int3 numVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
            for ( int zv = 0; zv < numVoxels.z; zv++ )
            {
                for ( int yv = 0; yv < numVoxels.y; yv++ )
                {
                    for ( int xv = 0; xv < numVoxels.x; xv++ )
                    {
                        int3 index3D = make_int3( xv, yv, zv );
                        int  index1D = Core::Index3DToIndex1D( index3D, numVoxels );

						if ( changeBits->test( index1D ) )
						{
							currentIdVolume[ index1D ] = newId;
						}
                    }
                }
            }

            //
            // save tile
            //
            SaveTile( tileIndex, volumeDescriptions );

            //
            // unload tile
            //
            UnloadTile( tileIndex );
			 
        }

        //
        // add tiles to the "newId" idTileMap (create a new idTileMap if necessary)
        //
		mTiledDatasetDescription.idTileMap.GetHashMap()[ newId ].insert( mRedoItem.idTileMapAddNewId.begin(), mRedoItem.idTileMapAddNewId.end() );
        //
        // remove tiles from the "oldId" idTileMap
        //
		for ( std::set< int4, Mojo::Core::Int4Comparator >::iterator eraseIterator = mRedoItem.idTileMapRemoveOldId.begin(); eraseIterator != mRedoItem.idTileMapRemoveOldId.end(); ++eraseIterator )
		{
			mTiledDatasetDescription.idTileMap.Get( oldId ).erase( *eraseIterator );
		}

        Core::Printf( "\nRedo operation complete: changed segmentation label ", oldId, " back to segmentation label ", newId, ".\n" );

		//
		// Make this a redo item
		//
		mUndoItem = mRedoItem;
		mRedoItem = FileSystemUndoRedoItem();
	}
}

void FileSystemTileServer::PrepForNextUndoRedoChange()
{
	//
	// Only one undo / redo supported
	//
	mRedoItem = FileSystemUndoRedoItem();
	mUndoItem = FileSystemUndoRedoItem();
}

//
// Tile Methods
//

Core::HashMap< std::string, Core::VolumeDescription > FileSystemTileServer::LoadTile( int4 tileIndex )
{
    std::string tileIndexString = CreateTileString( tileIndex );
    if ( mFileSystemTileCache.GetHashMap().find( tileIndexString ) != mFileSystemTileCache.GetHashMap().end() )
    {
        stdext::hash_map < std::string, FileSystemTileCacheEntry >::iterator ientry =
            mFileSystemTileCache.GetHashMap().find( tileIndexString );
        ientry->second.inUse++;
        ientry->second.timeStamp = clock();

        //Core::Printf("Got tile from cache (inUse=", ientry->second.inUse, ").");

        return ientry->second.volumeDescriptions;
    }
    else
    {
        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

        //
        // source map
        //
        volumeDescriptions.Set( "SourceMap", LoadTileImage( tileIndex, "SourceMap" ) );

        //
        // id map
        //
		if ( mIsSegmentationLoaded )
		{
			Core::VolumeDescription idMapVolumeDescription;
			bool success = TryLoadTileHdf5( tileIndex, "TempIdMap", "IdMap", idMapVolumeDescription );

			if ( !success )
			{
				idMapVolumeDescription = LoadTileHdf5( tileIndex, "IdMap", "IdMap" );
			}

			volumeDescriptions.Set( "IdMap", idMapVolumeDescription );

			Core::VolumeDescription overlayMapVolumeDescription;
			overlayMapVolumeDescription.dxgiFormat = idMapVolumeDescription.dxgiFormat;
			overlayMapVolumeDescription.isSigned = idMapVolumeDescription.isSigned;
			overlayMapVolumeDescription.numBytesPerVoxel = idMapVolumeDescription.numBytesPerVoxel;
			overlayMapVolumeDescription.numVoxels = idMapVolumeDescription.numVoxels;

			int nBytes = overlayMapVolumeDescription.numBytesPerVoxel * overlayMapVolumeDescription.numVoxels.x * overlayMapVolumeDescription.numVoxels.y * overlayMapVolumeDescription.numVoxels.z;
			overlayMapVolumeDescription.data = new unsigned char[ nBytes ];
			memset( overlayMapVolumeDescription.data, 0, nBytes );

			volumeDescriptions.Set( "OverlayMap", overlayMapVolumeDescription );

		}

        //
		// Add this tile to the cache
		//
        FileSystemTileCacheEntry entry = FileSystemTileCacheEntry(  );
        entry.tileIndex = tileIndex;
        entry.inUse = 1;
        entry.needsSaving = false;
        entry.timeStamp = clock();
        entry.volumeDescriptions = volumeDescriptions;
        mFileSystemTileCache.Set( tileIndexString, entry );

        //Core::Printf( "Got tile from disk (cache size is now ", mFileSystemTileCache.GetHashMap().size(), ")." );

        return volumeDescriptions;
    }
}


void FileSystemTileServer::UnloadTile( int4 tileIndex )
{
    //
	// Keep the tile in the cache, but mark it as unused
	//
    std::string tileIndexString = CreateTileString( tileIndex );
    if ( mFileSystemTileCache.GetHashMap().find( tileIndexString ) != mFileSystemTileCache.GetHashMap().end() )
    {
        stdext::hash_map < std::string, FileSystemTileCacheEntry >::iterator ientry =
            mFileSystemTileCache.GetHashMap().find( tileIndexString );
        ientry->second.inUse--;
    }

    ReduceCacheSizeIfNecessary();

}

void FileSystemTileServer::UnloadTiledDatasetInternal()
{
    mIsTiledDatasetLoaded    = false;
    mTiledDatasetDescription = TiledDatasetDescription();
}

void FileSystemTileServer::UnloadSegmentationInternal()
{
    mIsSegmentationLoaded    = false;
    mTiledDatasetDescription = TiledDatasetDescription();
}

Core::VolumeDescription FileSystemTileServer::LoadTileImage( int4 tileIndex, std::string imageName )
{
    Core::VolumeDescription volumeDescription;

    bool success = TryLoadTileImage( tileIndex, imageName, volumeDescription );

    RELEASE_ASSERT( success );

    return volumeDescription;
}

bool FileSystemTileServer::TryLoadTileImage( int4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription )
{
    switch ( mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).dxgiFormat )
    {
    case DXGI_FORMAT_R8_UNORM:
        return TryLoadTileImageInternal< uchar1 >( tileIndex, imageName, volumeDescription );
        break;

    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return TryLoadTileImageInternal< uchar4 >( tileIndex, imageName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        return false;
        break;
    }
}

void FileSystemTileServer::UnloadTileImage( Core::VolumeDescription& volumeDescription )
{
    UnloadTileImageInternal( volumeDescription );
}

void FileSystemTileServer::UnloadTileImageInternal( Core::VolumeDescription& volumeDescription )
{
    RELEASE_ASSERT( volumeDescription.data != NULL );

    delete[] volumeDescription.data;

    volumeDescription = Core::VolumeDescription();
}

Core::VolumeDescription FileSystemTileServer::LoadTileHdf5( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName )
{
    Core::VolumeDescription volumeDescription;

    bool success = TryLoadTileHdf5( tileIndex, hdf5Name, hdf5InternalDatasetName, volumeDescription );

    RELEASE_ASSERT( success );

    return volumeDescription;
}

bool FileSystemTileServer::TryLoadTileHdf5( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, Core::VolumeDescription& volumeDescription )
{
    switch ( mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).dxgiFormat )
    {
    case DXGI_FORMAT_R32_UINT:
        return TryLoadTileHdf5Internal< unsigned int >( tileIndex, hdf5Name, hdf5InternalDatasetName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        return false;
        break;
    }
}

void FileSystemTileServer::UnloadTileHdf5( Core::VolumeDescription& volumeDescription )
{
    UnloadTileHdf5Internal( volumeDescription );
}

void FileSystemTileServer::UnloadTileHdf5Internal( Core::VolumeDescription& volumeDescription )
{
    RELEASE_ASSERT( volumeDescription.data != NULL );

    delete[] volumeDescription.data;

    volumeDescription = Core::VolumeDescription();
}

void FileSystemTileServer::SaveTile( int4 tileIndex, Core::HashMap< std::string, Core::VolumeDescription >& volumeDescriptions )
{
    //
	// Mark this tile as needs saving
	//
    std::string tileIndexString = CreateTileString( tileIndex );
    if ( mFileSystemTileCache.GetHashMap().find( tileIndexString ) != mFileSystemTileCache.GetHashMap().end() )
    {
        mFileSystemTileCache.Get( tileIndexString ).needsSaving = true;
    }
}

void FileSystemTileServer::SaveTileImage( int4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription )
{
    switch ( mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).dxgiFormat )
    {
    case DXGI_FORMAT_R8_UNORM:
        SaveTileImageInternal< uchar1 >( tileIndex, imageName, volumeDescription );
        break;

    case DXGI_FORMAT_R8G8B8A8_UNORM:
        SaveTileImageInternal< uchar4 >( tileIndex, imageName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        break;
    }
}

void FileSystemTileServer::SaveTileHdf5( int4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, const Core::VolumeDescription& volumeDescription )
{
    switch ( mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).dxgiFormat )
    {
    case DXGI_FORMAT_R32_UINT:
        return SaveTileHdf5Internal< unsigned int >( tileIndex, hdf5Name, hdf5InternalDatasetName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        break;
    }
}


//
// Cache Methods
//

std::string FileSystemTileServer::CreateTileString( int4 tileIndex )
{
	return Core::ToString("W", tileIndex.w, "X", tileIndex.x, "Y", tileIndex.y, "Z", tileIndex.z);
	
	//std::string tileString = Core::ToString(
 //       "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), ",",
 //       "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), ",",
 //       "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
 //       "x=", Core::ToStringZeroPad( tileIndex.x, 8 ) );
	//Core::Printf( "Made tile string:", tileString );
	//return tileString;
}

int4 FileSystemTileServer::CreateTileIndex( std::string tileString )
{
	int w, x, y, z;
	sscanf_s( tileString.c_str(), "W%dX%dY%dZ%d", &w, &x, &y, &z );
	//sscanf_s( tileString.c_str(), "w=%dz=%dy=%dx=%d", &w, &z, &y, &x );
	return make_int4( x, y, z, w );
}

void FileSystemTileServer::FlushFileSystemTileCacheChanges()
{
    for( stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator i = mFileSystemTileCache.GetHashMap().begin(); i != mFileSystemTileCache.GetHashMap().end(); i++ )
    {
        if ( i->second.needsSaving )
        {
            //Core::Printf("Saving tile ", i->first, ".");
            SaveTileHdf5( i->second.tileIndex, "TempIdMap", "IdMap", i->second.volumeDescriptions.Get( "IdMap" ) );
            i->second.needsSaving = false;
        }
    }
}


void FileSystemTileServer::ReduceCacheSize()
{
    Core::Printf("Flushing Cache...");
    //
    // The file system tile cache is getting full - remove the oldest tiles
    //
    std::vector < clock_t > timestamps;
    mFileSystemTileCache.GetHashMap().begin();
    for( stdext::hash_map < std::string, FileSystemTileCacheEntry >::iterator i = mFileSystemTileCache.GetHashMap().begin(); i != mFileSystemTileCache.GetHashMap().end(); ++i )
    {
        timestamps.push_back( i->second.timeStamp );
    }
    std::sort(timestamps.begin(), timestamps.end());

    clock_t cutoff = timestamps[mFileSystemTileCache.GetHashMap().size() / 2];

    int removed = 0;

    stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator i = mFileSystemTileCache.GetHashMap().begin();
    while( i != mFileSystemTileCache.GetHashMap().end() )
    {
        if ( i->second.timeStamp < cutoff && i->second.inUse < 1 )
        {
            //
            // Remove this tile
            //
            if ( i->second.needsSaving )
            {
                //Core::Printf("Saving tile ", i->first, ".");
                SaveTileHdf5( i->second.tileIndex, "TempIdMap", "IdMap", i->second.volumeDescriptions.Get( "IdMap" ) );
                i->second.needsSaving = false;
            }

            if ( i->second.volumeDescriptions.GetHashMap().find( "IdMap" ) != i->second.volumeDescriptions.GetHashMap().end() )
            {
                UnloadTileHdf5( i->second.volumeDescriptions.Get( "IdMap" ) );
            }

            if ( i->second.volumeDescriptions.GetHashMap().find( "OverlayMap" ) != i->second.volumeDescriptions.GetHashMap().end() )
            {
                delete[] i->second.volumeDescriptions.Get( "OverlayMap" ).data;
            }

            UnloadTileImage( i->second.volumeDescriptions.Get( "SourceMap" ) );
            i->second.volumeDescriptions.GetHashMap().clear();

            stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator tempi = i;
            ++i;

            mFileSystemTileCache.GetHashMap().erase( tempi );
            ++removed;
        }
        else
        {
            ++i;
        }
    }
    Core::Printf("Removed ", removed, " tiles from the cache.");
}


void FileSystemTileServer::ReduceCacheSizeIfNecessary()
{

    if ( mFileSystemTileCache.GetHashMap().size() >= FILE_SYSTEM_TILE_CACHE_SIZE )
    {
        ReduceCacheSize();
    }

}

void FileSystemTileServer::SaveAndClearFileSystemTileCache( )
{
    stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator i = mFileSystemTileCache.GetHashMap().begin();
    while( i != mFileSystemTileCache.GetHashMap().end() )
    {
        //
		// Save this tile if necessary
		//
        if ( i->second.needsSaving )
        {
            //Core::Printf("Saving tile ", i->first, ".");
            SaveTileHdf5( i->second.tileIndex, "TempIdMap", "IdMap", i->second.volumeDescriptions.Get( "IdMap" ) );
            i->second.needsSaving = false;
        }

        //
		// Unload the tile
		//
		if ( i->second.volumeDescriptions.GetHashMap().find( "IdMap" ) != i->second.volumeDescriptions.GetHashMap().end() )
		{
			UnloadTileHdf5( i->second.volumeDescriptions.Get( "IdMap" ) );
        }

        if ( i->second.volumeDescriptions.GetHashMap().find( "OverlayMap" ) != i->second.volumeDescriptions.GetHashMap().end() )
        {
            delete[] i->second.volumeDescriptions.Get( "OverlayMap" ).data;
        }

        UnloadTileImage( i->second.volumeDescriptions.Get( "SourceMap" ) );
        i->second.volumeDescriptions.GetHashMap().clear();

        stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator tempi = i;
        ++i;

        mFileSystemTileCache.GetHashMap().erase( tempi );
    }
    RELEASE_ASSERT( mFileSystemTileCache.GetHashMap().size() == 0 );
}

}
}