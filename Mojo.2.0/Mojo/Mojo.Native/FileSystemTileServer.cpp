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
    mSplitDrawArea = 0;
	mSplitBorderTargets = 0;
    mSplitResultArea = 0;

    mPrevSplitId = -1;
    mPrevSplitZ = -1;
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

    if ( mSplitDrawArea != 0 )
        delete[] mSplitDrawArea;

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

	//
	// Make sure there are not temp files hanging around from last time
	//
	DeleteTempFiles();

    //Core::Printf( "FileSystemTileServer::LoadSegmentation Returning." );

}

void FileSystemTileServer::UnloadSegmentation()
{
    UnloadSegmentationInternal();
}

bool FileSystemTileServer::IsSegmentationLoaded()
{
    return mIsSegmentationLoaded;
}

void FileSystemTileServer::SaveSegmentation()
{
    //
    // save any tile changes (to temp directory)
    //

    Core::Printf( "Saving tiles (temp)." );

    FlushFileSystemTileCacheChanges();

    //
    // move changed tiles to the save directory
    //

    Core::Printf( "Replacing tiles." );

    int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;

    //size_t shape[] = { numTiles.w, numTiles.z, numTiles.y, numTiles.x };
    //mTileCachePageTable = marray::Marray< int >( shape, shape + 4 );

    TiledVolumeDescription tempVolumeDesctiption = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "TempIdMap" );

    TiledVolumeDescription saveVolumeDesctiption = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

    for ( int w = 0; w < numTiles.w; w++ )
    {
        //Core::Printf( "Saving w=", w, "." );
        for ( int z = 0; z < numTiles.z; z++ )
        {
            for ( int y = 0; y < numTiles.y; y++ )
            {
                for ( int x = 0; x < numTiles.x; x++ )
                {
                    std::string tempTilePathString = Core::ToString(
                        tempVolumeDesctiption.imageDataDirectory, "\\",
                        "w=", Core::ToStringZeroPad( w, 8 ), "\\",
                        "z=", Core::ToStringZeroPad( z, 8 ), "\\",
                        "y=", Core::ToStringZeroPad( y, 8 ), ",",
                        "x=", Core::ToStringZeroPad( x, 8 ), ".",
                        tempVolumeDesctiption.fileExtension );

                    boost::filesystem::path tempTilePath = boost::filesystem::path( tempTilePathString );

                    if ( boost::filesystem::exists( tempTilePath ) )
                    {
                        //Core::Printf( "Moving file: ", tempTilePathString, "." );

                        std::string saveTilePathString = Core::ToString(
                            saveVolumeDesctiption.imageDataDirectory, "\\",
                            "w=", Core::ToStringZeroPad( w, 8 ), "\\",
                            "z=", Core::ToStringZeroPad( z, 8 ), "\\",
                            "y=", Core::ToStringZeroPad( y, 8 ), ",",
                            "x=", Core::ToStringZeroPad( x, 8 ), ".",
                            saveVolumeDesctiption.fileExtension );

                        //Core::Printf( "To: ", saveTilePathString, "." );

                        boost::filesystem::path saveTilePath = boost::filesystem::path( saveTilePathString );
                        boost::filesystem::remove( saveTilePath );
                        boost::filesystem::rename( tempTilePath, saveTilePath );
                    }
                }
            }
        }
    }

    //
    // move the idIndex file
    //
    
	Core::Printf( "Saving idInfo and idTileIndex." );
    mSegmentInfoManager.Save();

    //Core::Printf( "Replacing idIndex." );

    //boost::filesystem::path idIndexPath = boost::filesystem::path( mTiledDatasetDescription.paths.Get( "IdInfo" ) );
    //boost::filesystem::path tempIdIndexPath = boost::filesystem::path( mTiledDatasetDescription.paths.Get( "TempIdInfo" ) );

    //boost::filesystem::remove( idIndexPath );
    //boost::filesystem::rename( tempIdIndexPath, idIndexPath );

    Core::Printf( "Segmentation saved." );

}

void FileSystemTileServer::SaveSegmentationAs( std::string savePath )
{
    //
    // save any tile changes (to temp directory)
    //

    //
    // save the idTileMap (to temp directory)
    //

    //
    // copy all (temp and normal) tiles to the new directory
    //

    Core::Printf( "Segmentation saved to: \"", savePath, "\" (disabled)." );
}

void FileSystemTileServer::AutosaveSegmentation()
{
    if ( mIsSegmentationLoaded )
    {
        bool success = false;
        int attempts = 0;

        while ( !success && attempts < 2 )
        {

            ++attempts;

            try
            {

	            //
                // save any tile changes (to temp directory)
                //

                Core::Printf( "Autosaving tiles (temp)." );

                FlushFileSystemTileCacheChanges();

                //
                // move changed tiles to the autosave directory
                //

                Core::Printf( "Autosave copying all modified tiles." );

                int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;

                //size_t shape[] = { numTiles.w, numTiles.z, numTiles.y, numTiles.x };
                //mTileCachePageTable = marray::Marray< int >( shape, shape + 4 );

                TiledVolumeDescription tempVolumeDesctiption = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "TempIdMap" );
                TiledVolumeDescription autosaveVolumeDesctiption = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "AutosaveIdMap" );

                for ( int w = 0; w < numTiles.w; w++ )
                {
                    //Core::Printf( "Saving w=", w, "." );
                    for ( int z = 0; z < numTiles.z; z++ )
                    {
                        for ( int y = 0; y < numTiles.y; y++ )
                        {
                            for ( int x = 0; x < numTiles.x; x++ )
                            {
                                std::string tempTilePathString = Core::ToString(
                                    tempVolumeDesctiption.imageDataDirectory, "\\",
                                    "w=", Core::ToStringZeroPad( w, 8 ), "\\",
                                    "z=", Core::ToStringZeroPad( z, 8 ), "\\",
                                    "y=", Core::ToStringZeroPad( y, 8 ), ",",
                                    "x=", Core::ToStringZeroPad( x, 8 ), ".",
                                    tempVolumeDesctiption.fileExtension );

                                boost::filesystem::path tempTilePath = boost::filesystem::path( tempTilePathString );

                                std::string autosaveTilePathString = Core::ToString(
                                    autosaveVolumeDesctiption.imageDataDirectory, "\\",
                                    "w=", Core::ToStringZeroPad( w, 8 ), "\\",
                                    "z=", Core::ToStringZeroPad( z, 8 ), "\\",
                                    "y=", Core::ToStringZeroPad( y, 8 ), ",",
                                    "x=", Core::ToStringZeroPad( x, 8 ), ".",
                                    autosaveVolumeDesctiption.fileExtension );

                                boost::filesystem::path autosaveTilePath = boost::filesystem::path( autosaveTilePathString );

					            if ( boost::filesystem::exists( autosaveTilePath ) )
					            {
                                    boost::filesystem::remove( autosaveTilePath );
					            }
					            else
					            {
						            boost::filesystem::create_directories( autosaveTilePath.parent_path() );
					            }

					            if ( boost::filesystem::exists( tempTilePath ) )
                                {
						            boost::filesystem::copy_file( tempTilePath, autosaveTilePath );
                                }
                            }
                        }
                    }
                }

                Core::Printf( "Tiles autosaved." );

                success = true;
            }
            catch ( std::bad_alloc e )
            {
                Core::Printf( "WARNING: std::bad_alloc Error while trying to autosave - reducing tile cache size." );
                ReduceCacheSize();
            }
            catch ( ... )
            {
                Core::Printf( "WARNING: Unexpected error while trying to autosave - attempting to continue." );
                ReduceCacheSize();
            }
        }

        if ( !success )
        {
            Core::Printf( "ERROR: Unable to autosave - possibly out of memory." );
        }
    }
}

void FileSystemTileServer::DeleteTempFiles()
{
    //
    // Delete tile files
    //

    Core::Printf( "Deleting temp files." );

    int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles;

    TiledVolumeDescription tempVolumeDesctiption = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "TempIdMap" );
    TiledVolumeDescription autosaveVolumeDesctiption = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

    for ( int w = 0; w < numTiles.w; w++ )
    {
        //Core::Printf( "Saving w=", w, "." );
        for ( int z = 0; z < numTiles.z; z++ )
        {
            for ( int y = 0; y < numTiles.y; y++ )
            {
                for ( int x = 0; x < numTiles.x; x++ )
                {
                    std::string tempTilePathString = Core::ToString(
                        tempVolumeDesctiption.imageDataDirectory, "\\",
                        "w=", Core::ToStringZeroPad( w, 8 ), "\\",
                        "z=", Core::ToStringZeroPad( z, 8 ), "\\",
                        "y=", Core::ToStringZeroPad( y, 8 ), ",",
                        "x=", Core::ToStringZeroPad( x, 8 ), ".",
                        tempVolumeDesctiption.fileExtension );

                    boost::filesystem::path tempTilePath = boost::filesystem::path( tempTilePathString );

					if ( boost::filesystem::exists( tempTilePath ) )
                    {
						boost::filesystem::remove( tempTilePath );
                    }
                }
            }
        }
    }

    //
    // delete temp idTileMap
    //

    boost::filesystem::path tempIdIndexPath = boost::filesystem::path( mTiledDatasetDescription.paths.Get( "TempIdInfo" ) );
    boost::filesystem::remove( tempIdIndexPath );

    Core::Printf( "Temp files deleted." );

}


int FileSystemTileServer::GetTileCountForId( int segId )
{
    return (int) mSegmentInfoManager.GetTileCount( segId );
}


//
// Edit Methods
//

void FileSystemTileServer::ReplaceSegmentationLabel( int oldId, int newId )
{
	if ( oldId != newId && mIsSegmentationLoaded )
    {
        Core::Printf( "\nReplacing segmentation label ", oldId, " with segmentation label ", newId, "...\n" );

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( oldId );
		FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = newId;
		mNextUndoItem->oldId = oldId;
        mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ oldId ].insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );
		mNextUndoItem->idTileMapAddNewId.insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );
		for ( FileSystemTileSet::iterator eraseIterator = tilesContainingNewId.begin(); eraseIterator != tilesContainingNewId.end(); ++eraseIterator )
		{
			mNextUndoItem->idTileMapAddNewId.erase( *eraseIterator );
		}

        for( FileSystemTileSet::iterator tileIndexi = tilesContainingOldId.begin(); tileIndexi != tilesContainingOldId.end(); ++tileIndexi )
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
			std::bitset< TILE_SIZE * TILE_SIZE > *changeBits = 
				&mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];

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
        tilesContainingNewId.insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );
        mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

        //
        // completely remove old id from our id tile map, since the old id is no longer present in the segmentation 
        //
        tilesContainingOldId.clear();
        mSegmentInfoManager.SetTiles( oldId, tilesContainingOldId );

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

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( oldId );
        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = newId;
		mNextUndoItem->oldId = oldId;
		std::bitset< TILE_SIZE * TILE_SIZE > *changeBits;

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
								mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
							}
                            tilesContainingNewId.insert( previousTileIndex );

                            //
							// Check if we can remove this tile from the oldId map
							//
                            if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, oldId ) )
                            {
                                tilesContainingOldId.erase( previousTileIndex );
                                mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ oldId ].insert( previousTileIndex );
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
						&mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
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
					mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
				}
                tilesContainingNewId.insert( previousTileIndex );

                //
				// Check if we can remove this tile from the oldId map
				//
                if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, oldId ) )
                {
                    tilesContainingOldId.erase( previousTileIndex );
					mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ oldId ].insert( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
		// Update idTileMap
		//
        mSegmentInfoManager.SetTiles( oldId, tilesContainingOldId );
        mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

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

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( oldId );
        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = newId;
		mNextUndoItem->oldId = oldId;
		std::bitset< TILE_SIZE * TILE_SIZE > *changeBits;

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
								mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
							}
                            tilesContainingNewId.insert( previousTileIndex );

                            //
							// Check if we can remove this tile from the oldId map
							//
                            if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, oldId ) )
                            {
                                tilesContainingOldId.erase( previousTileIndex );
								mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ oldId ].insert ( previousTileIndex );
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
						&mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
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
					mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
				}
                tilesContainingNewId.insert( previousTileIndex );

                //
				// Check if we can remove this tile from the oldId map
				//
                if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, oldId ) )
                {
                    tilesContainingOldId.erase( previousTileIndex );
					mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ oldId ].insert ( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
		// Update idTileMap
		//
        mSegmentInfoManager.SetTiles( oldId, tilesContainingOldId );
        mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

        Core::Printf( "\nFinished replacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, ".\n" );
    }
}

//
// 2D Split Methods
//


void FileSystemTileServer::DrawSplit( float3 pointTileSpace, float radius )
{
	DrawRegionValue( pointTileSpace, radius, REGION_SPLIT );
}

void FileSystemTileServer::DrawErase( float3 pointTileSpace, float radius )
{
	DrawRegionValue( pointTileSpace, radius, 0 );
}

void FileSystemTileServer::DrawRegionA( float3 pointTileSpace, float radius )
{
	DrawRegionValue( pointTileSpace, radius, REGION_A );
}

void FileSystemTileServer::DrawRegionB( float3 pointTileSpace, float radius )
{
	DrawRegionValue( pointTileSpace, radius, REGION_B );
}

void FileSystemTileServer::DrawRegionValue( float3 pointTileSpace, float radius, int value )
{
    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

    int3 pVoxelSpace = 
        make_int3(
        (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
        (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
        (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    int areaX = pVoxelSpace.x - mSplitWindowStart.x * numVoxelsPerTile.x;
    int areaY = pVoxelSpace.y - mSplitWindowStart.y * numVoxelsPerTile.y;
    int areaIndex = areaX + areaY * mSplitWindowWidth;

    SimpleSplitTools::ApplyCircleMask( areaIndex, mSplitWindowWidth, mSplitWindowHeight, value, radius, mSplitDrawArea );

    int irad = (int) ( radius + 0.5 );
    int2 upperLeft = make_int2( areaX - irad, areaY - irad );
    int2 lowerRight = make_int2( areaX + irad, areaY + irad );
    UpdateOverlayTilesBoundingBox( upperLeft, lowerRight );

    //Core::Printf( "\nDrew inside bounding box (", upperLeft.x, ",", upperLeft.y, "x", lowerRight.x, ",", lowerRight.y, ").\n" );
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

    for ( unsigned int si = 0; si < mSplitSourcePoints.size(); ++si )
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

void FileSystemTileServer::UpdateOverlayTiles()
{
    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

    int2 upperLeft = make_int2( 0, 0 );
    int2 lowerRight = make_int2( mSplitWindowNTiles.x * numVoxelsPerTile.x, mSplitWindowNTiles.y * numVoxelsPerTile.y );

    UpdateOverlayTilesBoundingBox( upperLeft, lowerRight );
}

void FileSystemTileServer::UpdateOverlayTilesBoundingBox( int2 upperLeft, int2 lowerRight )
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

	for ( int xd = 0; xd < mSplitWindowNTiles.x; ++xd )
	{
		for (int yd = 0; yd < mSplitWindowNTiles.y; ++yd )
		{

            //
            // Check bounding box
            //

			int xOffset = xd * numVoxelsPerTile.x;
			int yOffset = yd * numVoxelsPerTile.y;

            int minX = upperLeft.x - xOffset;
            int maxX = lowerRight.x - xOffset;

            int minY = upperLeft.y - yOffset;
            int maxY = lowerRight.y - yOffset;

            if ( minX < 0 )
                minX = 0;
            if ( maxX >= numVoxelsPerTile.x )
                maxX = numVoxelsPerTile.x - 1;
            if ( minY < 0 )
                minY = 0;
            if ( maxY >= numVoxelsPerTile.y )
                maxY = numVoxelsPerTile.y - 1;

            if ( minX < maxX && minY < maxY )
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

                for ( int tileY = minY; tileY <= maxY; ++tileY )
                {
                    for ( int tileX = minX; tileX <= maxX; ++tileX )
                    {
                        int tileIndex1D = tileX + tileY * numVoxelsPerTile.x;
                        int areaIndex1D = xOffset + tileX + ( yOffset + tileY ) * mSplitWindowWidth;

					    RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

                        if ( mSplitResultArea != 0 )
                        {
                            if ( mSplitResultArea[ areaIndex1D ] && splitData[ tileIndex1D ] != mSplitResultArea[ areaIndex1D ] )
                            {
                                splitData[ tileIndex1D ] = mSplitResultArea[ areaIndex1D ];
                                wQueue[ make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitResultArea[ areaIndex1D ];
                            }
                            else if ( mSplitResultArea[ areaIndex1D ] == 0 && mSplitDrawArea[ areaIndex1D ] && splitData[ tileIndex1D ] != mSplitDrawArea[ areaIndex1D ] )
                            {
                                splitData[ tileIndex1D ] = mSplitDrawArea[ areaIndex1D ];
						        wQueue[ make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitDrawArea[ areaIndex1D ];
                            }
						    else if ( splitData[ tileIndex1D ] && mSplitResultArea[ areaIndex1D ] == 0 && mSplitDrawArea[ areaIndex1D ] == 0)
						    {
                                splitData[ tileIndex1D ] = 0;
						        wQueue[ make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = 0;
						    }
                        }
                        else if ( mSplitDrawArea != 0 )
                        {
                            if ( mSplitDrawArea[ areaIndex1D ] && splitData[ tileIndex1D ] != mSplitDrawArea[ areaIndex1D ] )
                            {
                                splitData[ tileIndex1D ] = mSplitDrawArea[ areaIndex1D ];
						        wQueue[ make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitDrawArea[ areaIndex1D ];
                            }
						    else if ( splitData[ tileIndex1D ] && mSplitDrawArea[ areaIndex1D ] == 0)
						    {
                                splitData[ tileIndex1D ] = 0;
						        wQueue[ make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = 0;
						    }
                        }

                    }
                }

				UnloadTile( tileIndex );
            }
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

void FileSystemTileServer::ResetOverlayTiles()
{
	//
	// Reset the overlay layer
	//
    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
	int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

	Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

	std::queue< int4 > wQueue;

	int tileCount = 0;
	int sourceSplitCount = 0;
	unsigned int* splitData;

	for ( int xd = 0; xd < mSplitWindowNTiles.x; ++xd )
	{
		for (int yd = 0; yd < mSplitWindowNTiles.y; ++yd )
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
        //    Core::Printf( "ResetOverlayTiles: updating voxel: ", thisVoxel.x, ",",  thisVoxel.y, ",",  thisVoxel.z, ",",  thisVoxel.w );
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
	mSplitNPerimiters = 0;

	//memset( mSplitResultArea, 0, mSplitWindowNPix );
	//memset( mSplitDrawArea, 0, mSplitWindowNPix );
	//memcpy( mSplitSearchMask, mSplitBorderTargets, mSplitWindowNPix );

    for ( int i = 0; i < mSplitWindowNPix; ++i )
    {
		mSplitSearchMask[ i ] = mSplitBorderTargets[ i ];
        mSplitResultArea[ i ] = 0;
        mSplitDrawArea[ i ] = 0;
    }

	ResetOverlayTiles();

    Core::Printf( "Reset Split State.\n");

}

void FileSystemTileServer::ResetAdjustState()
{

    for ( int i = 0; i < mSplitWindowNPix; ++i )
    {
        mSplitDrawArea[ i ] = 0;
    }

	ResetOverlayTiles();

    Core::Printf( "Reset Adjust State.\n");

}

void FileSystemTileServer::LoadSplitDistances( int segId )
{
    //
    // Load distances
    //
    Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
    unsigned char* currentSrcVolume;
    int* currentIdVolume;

    int tileCount = 0;
    int areaCount = 0;
    mCentroid.x = 0;
    mCentroid.y = 0;
    mSplitLabelCount = 0;

    for ( int xd = 0; xd < mSplitWindowNTiles.x; ++xd )
    {
        for (int yd = 0; yd < mSplitWindowNTiles.y; ++yd )
        {

            ++tileCount;
            //Core::Printf( "Loading distance tile ", tileCount, "." );

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
                int segVal = ( (int) currentSrcVolume[ tileIndex1D ] ) + 10;
                mSplitStepDist[ areaIndex1D ] = segVal * segVal;
                ++areaCount;

                //
                // Mark border targets
                //
                if ( currentIdVolume[ tileIndex1D ] == segId )
                {
                    ++mSplitLabelCount;
                    mCentroid.x += (float) areaX + mSplitWindowStart.x * numVoxelsPerTile.x;
                    mCentroid.y += (float) areaY + mSplitWindowStart.y * numVoxelsPerTile.y;
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

    mCentroid.x /= (float) mSplitLabelCount;
    mCentroid.y /= (float) mSplitLabelCount;
    Core::Printf( "Segment centroid at ", mCentroid.x, "x", mCentroid.y, "." );


    //Core::Printf( "Loaded: areaCount=", areaCount );

}

void FileSystemTileServer::PrepForSplit( int segId, float3 pointTileSpace )
{
    //
    // Find the size of this segment and load the bounding box of tiles
    //

    if ( mIsSegmentationLoaded )
    {
        bool success = false;
        int attempts = 0;

        while ( !success && attempts < 2 )
        {
            ++attempts;

            try
            {

                Core::Printf( "\nPreparing for split of segment ", segId, " at x=", pointTileSpace.x, ", y=", pointTileSpace.y, ", z=", pointTileSpace.z, ".\n" );

                int3 numVoxels = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels;
                int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
                int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

                int minTileX = numTiles.x;
                int maxTileX = 0;
                int minTileY = numTiles.y;
                int maxTileY = 0;

                FileSystemTileSet tilesContainingSegId = mSegmentInfoManager.GetTiles( segId );

                for ( FileSystemTileSet::iterator tileIterator = tilesContainingSegId.begin(); tileIterator != tilesContainingSegId.end(); ++tileIterator )
                {
                    if ( tileIterator->z == pointTileSpace.z && tileIterator->w == 0 )
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

                // Restrict search tiles to max 2 tiles away from clicked location
                if ( minTileX < ( (int) pointTileSpace.x ) - 2 )
                    minTileX = ( (int) pointTileSpace.x ) - 2;
                if ( maxTileX > ( (int) pointTileSpace.x ) + 2 )
                    maxTileX = ( (int) pointTileSpace.x ) + 2;

                if ( minTileY < ( (int) pointTileSpace.y ) - 2 )
                    minTileY = ( (int) pointTileSpace.y ) - 2;
                if ( maxTileY > ( (int) pointTileSpace.y ) + 2 )
                    maxTileY = ( (int) pointTileSpace.y ) + 2;

                //
                // Calculate sizes
                //
                mSplitWindowStart = make_int3( minTileX, minTileY, (int) pointTileSpace.z );
                mSplitWindowNTiles = make_int3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

                //Core::Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
                //Core::Printf( "mSplitWindowSize=", mSplitWindowNTiles.x, ":", mSplitWindowNTiles.y, ":", mSplitWindowNTiles.z, ".\n" );

		        mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowNTiles.x;
		        mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowNTiles.y;
		        mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

                //Core::Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

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

                if ( mSplitDrawArea != 0 )
                    delete[] mSplitDrawArea;

		        if ( mSplitBorderTargets != 0 )
			        delete[] mSplitBorderTargets;

                if ( mSplitResultArea != 0 )
                    delete[] mSplitResultArea;

		        mSplitStepDist = new int[ mSplitWindowNPix ];
                mSplitResultDist = new int[ mSplitWindowNPix ];
		        mSplitPrev = new int[ mSplitWindowNPix ];
		        mSplitSearchMask = new char[ mSplitWindowNPix ];
		        mSplitDrawArea = new char[ mSplitWindowNPix ];
		        mSplitBorderTargets = new char[ mSplitWindowNPix ];
		        mSplitResultArea = new unsigned int[ mSplitWindowNPix ];

                LoadSplitDistances( segId );

                ResetSplitState();

                Core::Printf( "\nFinished preparing for split of segment ", segId, " at z=", pointTileSpace.z, ".\n" );

                success = true;
            }
            catch ( std::bad_alloc e )
            {
                Core::Printf( "WARNING: std::bad_alloc Error while preparing for split - reducing tile cache size." );
                ReduceCacheSize();
            }
            catch ( ... )
            {
                Core::Printf( "WARNING: Unexpected error while preparing for split - attempting to continue." );
                ReduceCacheSize();
            }
        }

        if ( !success )
        {
            Core::Printf( "ERROR: Unable to prep for split - possibly out of memory." );
        }
    }

}

void FileSystemTileServer::PrepForAdjust( int segId, float3 pointTileSpace )
{
    //
    // Find the size of this segment and load the bounding box of tiles
    //

    if ( mIsSegmentationLoaded )
    {
        bool success = false;
        int attempts = 0;

        while ( !success && attempts < 2 )
        {
            ++attempts;

            try
            {

                Core::Printf( "\nPreparing for adjustment of segment ", segId, " at x=", pointTileSpace.x, ", y=", pointTileSpace.y, ", z=", pointTileSpace.z, ".\n" );

                int3 numVoxels = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxels;
                int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
                int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

                int minTileX = numTiles.x;
                int maxTileX = 0;
                int minTileY = numTiles.y;
                int maxTileY = 0;

                FileSystemTileSet tilesContainingSegId = mSegmentInfoManager.GetTiles( segId );

                for ( FileSystemTileSet::iterator tileIterator = tilesContainingSegId.begin(); tileIterator != tilesContainingSegId.end(); ++tileIterator )
                {
                    if ( tileIterator->z == pointTileSpace.z && tileIterator->w == 0 )
                    {
                        //
                        // Include this tile
                        //
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
                // Include any neighbours
                //
                if ( minTileX > 0 )
                    --minTileX;
                if ( maxTileX < numTiles.x - 1 )
                    ++maxTileX;

                if ( minTileY > 0 )
                    --minTileY;
                if ( maxTileY < numTiles.y - 1 )
                    ++maxTileY;

                //
                // Restrict search tiles to max 2 tiles away from clicked location
                //
                if ( minTileX < ( (int) pointTileSpace.x ) - 2 )
                    minTileX = ( (int) pointTileSpace.x ) - 2;
                if ( maxTileX > ( (int) pointTileSpace.x ) + 2 )
                    maxTileX = ( (int) pointTileSpace.x ) + 2;

                if ( minTileY < ( (int) pointTileSpace.y ) - 2 )
                    minTileY = ( (int) pointTileSpace.y ) - 2;
                if ( maxTileY > ( (int) pointTileSpace.y ) + 2 )
                    maxTileY = ( (int) pointTileSpace.y ) + 2;

                //
                // Calculate sizes
                //
                mSplitWindowStart = make_int3( minTileX, minTileY, (int) pointTileSpace.z );
                mSplitWindowNTiles = make_int3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

                //Core::Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
                //Core::Printf( "mSplitWindowSize=", mSplitWindowNTiles.x, ":", mSplitWindowNTiles.y, ":", mSplitWindowNTiles.z, ".\n" );

		        mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowNTiles.x;
		        mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowNTiles.y;
		        mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

                //Core::Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

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

                if ( mSplitDrawArea != 0 )
                    delete[] mSplitDrawArea;

		        if ( mSplitBorderTargets != 0 )
			        delete[] mSplitBorderTargets;

                if ( mSplitResultArea != 0 )
                    delete[] mSplitResultArea;

		        mSplitStepDist = 0;
                mSplitResultDist = 0;
		        mSplitPrev = 0;
		        mSplitSearchMask = 0;
		        mSplitDrawArea = new char[ mSplitWindowNPix ];
		        mSplitBorderTargets = 0;
		        mSplitResultArea = 0;

                ResetAdjustState();

                Core::Printf( "\nFinished preparing for adjustment of segment ", segId, " at z=", pointTileSpace.z, ".\n" );

                success = true;
            }
            catch ( std::bad_alloc e )
            {
                Core::Printf( "WARNING: std::bad_alloc Error while preparing for adjust - reducing tile cache size." );
                ReduceCacheSize();
            }
            catch ( ... )
            {
                Core::Printf( "WARNING: Unexpected error while preparing for adjust - attempting to continue." );
                ReduceCacheSize();
            }
        }

        if ( !success )
        {
            Core::Printf( "ERROR: Unable to prep for adjust - possibly out of memory." );
        }
    }

}

void FileSystemTileServer::RecordSplitState( int segId, float3 pointTileSpace )
{
    if ( mIsSegmentationLoaded )
    {
	    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
        int currentZ = (int) floor( pointTileSpace.z * numVoxelsPerTile.z );

        bool anyLinePixels = false;

        FileSystemSplitState currentState;
        currentState.splitId = segId;
        //currentState.splitZ = (int) floor( pointTileSpace.z * numVoxelsPerTile.z );

        //
        // Record split line location and draw points
        //
        for ( int areaX = 0; areaX < mSplitWindowWidth; ++areaX )
        {
            for ( int areaY = 0; areaY < mSplitWindowHeight; ++areaY )
            {
                int areaIndex1D = areaX + areaY * mSplitWindowWidth;
                if ( mSplitResultArea[ areaIndex1D ] == PATH_RESULT )
                {
                    currentState.splitLine.push_back( make_float2(
                        (float) ( mSplitWindowStart.x * numVoxelsPerTile.x + areaX ) / (float) numVoxelsPerTile.x,
                        (float) ( mSplitWindowStart.y * numVoxelsPerTile.y + areaY ) / (float) numVoxelsPerTile.y ) );
                    anyLinePixels = true;
                }
                if ( mSplitDrawArea[ areaIndex1D ] != 0 )
                {
                    currentState.splitDrawPoints.push_back( std::pair< float2, char >( make_float2(
                        (float) ( mSplitWindowStart.x * numVoxelsPerTile.x + areaX ) / (float) numVoxelsPerTile.x,
                        (float) ( mSplitWindowStart.y * numVoxelsPerTile.y + areaY ) / (float) numVoxelsPerTile.y ),
                        mSplitDrawArea[ areaIndex1D ] ) );
                }
            }
        }

        if ( anyLinePixels )
        {
            mSplitStates[ currentZ ] = currentState;
        }

    }
}

void FileSystemTileServer::PredictSplit( int segId, float3 pointTileSpace, float radius )
{
    if ( mIsSegmentationLoaded )
    {
	    int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
        int currentZ = (int) floor( pointTileSpace.z * numVoxelsPerTile.z );

        std::vector< float2 > *restoreLine = NULL;

        //
        // Try to restore from stored state
        //
        if ( mSplitStates.find( currentZ ) != mSplitStates.end() )
        {
            std::vector< std::pair< float2, char >> *restorePoints = &mSplitStates.find( currentZ )->second.splitDrawPoints;
            for ( std::vector< std::pair< float2, char >>::iterator drawIt = restorePoints->begin(); drawIt != restorePoints->end(); ++drawIt )
            {
                int areaX = ( (int) floor( drawIt->first.x * numVoxelsPerTile.x ) ) - mSplitWindowStart.x * numVoxelsPerTile.x;
                int areaY = ( (int) floor( drawIt->first.y * numVoxelsPerTile.y ) ) - mSplitWindowStart.y * numVoxelsPerTile.y;
                int areaIndex = areaX + areaY * mSplitWindowWidth;

                if ( areaIndex > 0 && areaIndex < mSplitWindowNPix )
                {
                    mSplitDrawArea[ areaIndex ] = drawIt->second;
                }
            }

            FindBoundaryWithinRegion2D( segId );
            Core::Printf( "Restored draw points from stored state at z=", currentZ, "." );
        }
        else if ( mPrevSplitId == segId && ( mPrevSplitZ == currentZ - 1 || mPrevSplitZ == currentZ + 1 ) )
        {
            restoreLine = &mPrevSplitLine;
            Core::Printf( "Predicing split at z=", currentZ, ", from previous split." );
        }
        else if ( mSplitStates.find( currentZ - 1 ) != mSplitStates.end() )
        {
            restoreLine = &mSplitStates.find( currentZ - 1 )->second.splitLine;
            Core::Printf( "Predicting split at z=", currentZ, ", from neighbour -1." );
        }
        else if ( mSplitStates.find( currentZ + 1 ) != mSplitStates.end() )
        {
            restoreLine = &mSplitStates.find( currentZ + 1 )->second.splitLine;
            Core::Printf( "Predicting split at z=", currentZ, ", from neighbour +1." );
        }
        else
        {
            Core::Printf( "No neighbours for split prediction at z=", currentZ, " (prev=", mPrevSplitZ, ")." );
        }

        if ( restoreLine != NULL )
        {
            //
            // Draw a line where the previous split was
            //
            for ( std::vector< float2 >::iterator splitIt = restoreLine->begin(); splitIt != restoreLine->end(); ++splitIt )
            {
                DrawSplit( make_float3( splitIt->x, splitIt->y, (float) currentZ ), radius );
            }

            //
            // Find another split line here
            //
            FindBoundaryWithinRegion2D( segId );
        }
    }
}

int FileSystemTileServer::CompletePointSplit( int segId, float3 pointTileSpace )
{
	int newId = 0;

	if ( mIsSegmentationLoaded && mSplitSourcePoints.size() > 0 )
    {

		int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

        int3 pMouseOverVoxelSpace = 
            make_int3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

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
						//Core::Printf( "Seed found at:", seedIndex1D, "." );
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

		std::bitset< TILE_SIZE * TILE_SIZE > *changeBits;

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
        bool foundMouseOverPixel = false;

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

            if ( thisVoxel.x == pMouseOverVoxelSpace.x && thisVoxel.y == pMouseOverVoxelSpace.y && thisVoxel.z == pMouseOverVoxelSpace.z )
            {
                foundMouseOverPixel = true;
			    //Core::Printf( "Found mouseover pixel - inverting." );
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
				changeBits = &mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
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
			//if ( tileQueue.size() == 0 && sliceQueue.size() == 0 && nPixChanged > 1 + mSplitLabelCount / 2 && !invert )

			//
			// Check for inversion ( do not re-label the mouse-over segment )
			//
            if ( tileQueue.size() == 0 && sliceQueue.size() == 0 && foundMouseOverPixel && !invert )
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

				for ( int xd = 0; !seedFound && xd < mSplitWindowNTiles.x; ++xd )
				{
					for (int yd = 0; !seedFound && yd < mSplitWindowNTiles.y; ++yd )
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

							if ( currentIdVolume[ tileIndex1D ] == segId && !mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ].test( tileIndex1D ) && mSplitResultArea[ seedIndex1D ] == 0 )
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
                mNextUndoItem->changePixels.GetHashMap().clear();

				if ( !seedFound )
				{
					Core::Printf( "WARNING: Could not find (inverted) seed point - aborting." );
					ResetSplitState();
					return 0;
				}
				else
				{
					//Core::Printf( "Seed found at:", seedIndex1D, "." );
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

        newId = mSegmentInfoManager.AddNewId();
        mTiledDatasetDescription.maxLabelId = newId;
        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( segId );
        FileSystemTileSet tilesContainingNewId;

		mNextUndoItem->newId = newId;
		mNextUndoItem->oldId = segId;

		bool tileChanged;

        //Core::Printf( "Splitting (invert=", invert, ")." );

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
								mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
							}
							tilesContainingNewId.insert( previousTileIndex );

							//
							// Check if we can remove this tile from the oldId map
							//
							if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, segId ) )
							{
								tilesContainingOldId.erase( previousTileIndex );
								mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ segId ].insert( previousTileIndex );
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
						&mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
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
					mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
				}
                tilesContainingNewId.insert( previousTileIndex );

                //
				// Check if we can remove this tile from the oldId map
				//
                if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, segId ) )
                {
                    tilesContainingOldId.erase( previousTileIndex );
					mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ segId ].insert( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
		// Update idTileMap
		//
        mSegmentInfoManager.SetTiles( segId, tilesContainingOldId );
        mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

        Core::Printf( "\nFinished Splitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") to new segmentation label ", newId, "...\n" );
    }

	//
	// Prep for more splitting
	//
	LoadSplitDistances( segId );
    ResetSplitState();

	return newId;

}

int FileSystemTileServer::CompleteDrawSplit( int segId, float3 pointTileSpace, bool join3D, int splitStartZ )
{
	int newId = 0;

	if ( mIsSegmentationLoaded && mSplitNPerimiters > 0 )
    {

        RecordSplitState( segId, pointTileSpace );

		int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
        int3 pMouseOverVoxelSpace = 
            make_int3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

        int currentZ = pMouseOverVoxelSpace.z;
        int direction = 0;

        if ( splitStartZ < currentZ )
        {
            direction = -1;
        }
        else if ( splitStartZ > currentZ )
        {
            direction = 1;
        }

        bool continueZ = true;
        std::vector< std::pair< float2, int >> newCentroids;

        while ( continueZ )
        {

            float centerX = 0;
            float centerY = 0;
            float unchangedCenterX = 0;
            float unchangedCenterY = 0;

            int fillSuccesses = 0;

		    for ( int perimiter = 1; perimiter <= mSplitNPerimiters; ++ perimiter )
		    {
                if ( fillSuccesses == mSplitNPerimiters - 1 )
                {
                    break;
                }

			    //
			    // Find a seed point with this perimiter value
			    //

                newId = 0;
			    bool seedFound = false;
			    int seedIndex1D;

			    for ( seedIndex1D = 0; seedIndex1D < mSplitWindowNPix; ++seedIndex1D )
			    {
				    if ( mSplitSearchMask[ seedIndex1D ] == perimiter )
				    {
                        //
                        // Check there is at least one valid neighbour
                        //
                        if ( ( seedIndex1D % mSplitWindowWidth > 0 && mSplitBorderTargets[ seedIndex1D - 1 ] == 0 ) ||
                             ( seedIndex1D % mSplitWindowWidth < mSplitWindowWidth - 1 && mSplitBorderTargets[ seedIndex1D + 1 ] == 0 ) ||
                             ( seedIndex1D / mSplitWindowWidth > 0 && mSplitBorderTargets[ seedIndex1D - mSplitWindowWidth ] == 0 ) ||
                             ( seedIndex1D / mSplitWindowWidth > mSplitWindowHeight - 1 && mSplitBorderTargets[ seedIndex1D + mSplitWindowWidth ] == 0 ) )
                        {
					        seedFound = true;
					        break;
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

			    std::bitset< TILE_SIZE * TILE_SIZE > *changeBits;

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
                bool foundMouseOverPixel = false;

			    tileQueue.push( make_int4( pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0 ) );

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

                    if ( thisVoxel.x == pMouseOverVoxelSpace.x && thisVoxel.y == pMouseOverVoxelSpace.y && thisVoxel.z == pMouseOverVoxelSpace.z )
                    {
                        foundMouseOverPixel = true;
			            //Core::Printf( "Found mouseover pixel - inverting." );
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
					    changeBits = &mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
				    }

				    int tileX = thisVoxel.x % numVoxelsPerTile.x;
				    int tileY = thisVoxel.y % numVoxelsPerTile.y;

				    int3 index3D = make_int3( tileX, tileY, 0 );
				    int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
				    int  idValue = currentIdVolume[ index1D ];

				    bool isSplitBorder = false;

				    int areaIndex1D = thisVoxel.x - mSplitWindowStart.x * numVoxelsPerTile.x +
					    (thisVoxel.y - mSplitWindowStart.y * numVoxelsPerTile.y) * mSplitWindowWidth;
				    if ( areaIndex1D >= 0 && areaIndex1D < mSplitWindowNPix )
				    {
					    isSplitBorder = mSplitResultArea[ areaIndex1D ] != 0;
				    }

				    if ( idValue == segId && !changeBits->test( index1D ) )
				    {
					    changeBits->set( index1D );
					    wQueue.push( thisVoxel );
					    ++nPixChanged;

                        centerX += (float) thisVoxel.x;
                        centerY += (float) thisVoxel.y;

					    if ( !isSplitBorder )
					    {
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
				    }

                    if ( tileQueue.size() == 0 && sliceQueue.size() == 0 && nPixChanged == 0 )
				    {
                        Core::Printf( "WARNING: No fill pixels found for draw split - attempting to find next perimiter pixel." );
                    }
                    else if ( tileQueue.size() == 0 && sliceQueue.size() == 0 )
				    {
                        centerX = centerX / (float) nPixChanged;
                        centerY = centerY / (float) nPixChanged;
                        Core::Printf( "Split centroid at ", centerX, "x", centerY, ".");
			            //
			            // Check for 3D link
			            //
                        float minDist = -1;
                        int closestId = -1;
                        std::vector< std::pair< float2, int >>::iterator closestIt;

                        if ( join3D && segId == mPrevSplitId && ( mPrevSplitZ == mSplitWindowStart.z - 1 || mPrevSplitZ == mSplitWindowStart.z + 1 ) )
                        {
                            Core::Printf( "Checking for 3D link, ", mPrevSplitCentroids.size(), " candidates." );

                            for ( std::vector< std::pair< float2, int >>::iterator centroidIt = mPrevSplitCentroids.begin(); centroidIt != mPrevSplitCentroids.end(); ++centroidIt )
                            {
                                float thisDist = sqrt ( ( centroidIt->first.x - centerX ) * ( centroidIt->first.x - centerX ) + ( centroidIt->first.y - centerY ) * ( centroidIt->first.y - centerY ) );
                                if ( minDist < 0 || thisDist < minDist )
                                {
                                    minDist = thisDist;
                                    closestId = centroidIt->second;
                                    closestIt = centroidIt;
                                }
                            }

                            if ( closestId != segId )
                            {
                                newId = closestId;
                                mPrevSplitCentroids.erase( closestIt );
                                Core::Printf( "Found centroid at distance ", minDist, " id=", newId, ".");
                            }
                            else if ( invert )
                            {
                                Core::Printf( "Warning - could not find a 3D link.");
                            }
                        }

                        if ( !invert )
                        {
                            if ( join3D && segId == mPrevSplitId && ( mPrevSplitZ == mSplitWindowStart.z - 1 || mPrevSplitZ == mSplitWindowStart.z + 1 ) )
                            {
                                if ( closestId == segId )
                                {
                                    invert = true;
                                    Core::Printf( "Inverting (3D join).");
                                }
                            }
                            else if ( foundMouseOverPixel )
                            {
			                    //
			                    // Check for inversion ( do not re-label the mouse-over segment )
			                    //
                                invert = true;
                                Core::Printf( "Inverting (mouse over).");
                            }

                            if ( invert )
                            {
					            //
					            // Invert - find an alternative fill point
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

					            for ( int xd = 0; !seedFound && xd < mSplitWindowNTiles.x; ++xd )
					            {
						            for (int yd = 0; !seedFound && yd < mSplitWindowNTiles.y; ++yd )
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

								            if ( currentIdVolume[ tileIndex1D ] == segId && !mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ].test( tileIndex1D ) && mSplitResultArea[ seedIndex1D ] == 0 )
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
					            mNextUndoItem->changePixels.GetHashMap().clear();

					            if ( !seedFound )
					            {
						            Core::Printf( "WARNING: Could not find (inverted) seed point - aborting." );
						            ResetSplitState();
						            return 0;
					            }
					            else
					            {
						            //Core::Printf( "Seed found at:", seedIndex1D, "." );
					            }

					            //
					            // Use this new seed point
					            //
					            pVoxelSpace = 
						            make_int3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
						            seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
						            mSplitWindowStart.z );

                                newId = 0;
                                centerX = 0;
                                centerY = 0;
					            nPixChanged = 0;
					            invert = true;

					            tileQueue.push( make_int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

					            Core::Printf( "Filling (invert) at w=0." );
                            }
                        }
				    }
			    }

                if ( nPixChanged > 0 )
                {
                    ++fillSuccesses;
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

                if ( newId == 0 )
                {
                    newId = mSegmentInfoManager.AddNewId();
                }
                mTiledDatasetDescription.maxLabelId = newId;
                FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( segId );
			    FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

			    mNextUndoItem->newId = newId;
			    mNextUndoItem->oldId = segId;

			    bool tileChanged;

			    //Core::Printf( "Splitting (invert=", invert, ")." );

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
									    mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
								    }
								    tilesContainingNewId.insert( previousTileIndex );

								    //
								    // Check if we can remove this tile from the oldId map
								    //
								    if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, segId ) )
								    {
									    tilesContainingOldId.erase( previousTileIndex );
									    mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ segId ].insert( previousTileIndex );
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
							    &mNextUndoItem->changePixels.GetHashMap()[ CreateTileString( tileIndex ) ];
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
						    mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
					    }
					    tilesContainingNewId.insert( previousTileIndex );

					    //
					    // Check if we can remove this tile from the oldId map
					    //
					    if ( !TileContainsId( numVoxelsPerTile, currentIdNumVoxels, currentIdVolume, segId ) )
					    {
						    tilesContainingOldId.erase( previousTileIndex );
						    mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ segId ].insert( previousTileIndex );
					    }

				    }
				    UnloadTile( previousTileIndex );
			    }

			    //
			    // Update idTileMap
			    //
			    mSegmentInfoManager.SetTiles( segId, tilesContainingOldId );
			    mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

                //
                // Record new centroid
                //
                newCentroids.push_back( std::pair< float2, int >( make_float2( centerX, centerY ), newId ));

                //
                // Recalculate centroid of unchanged label
                //
                mSplitLabelCount -= nPixChanged;
                mCentroid.x += ( mCentroid.x - centerX ) * ( (float)nPixChanged / (float)mSplitLabelCount );
                mCentroid.y += ( mCentroid.y - centerY ) * ( (float)nPixChanged / (float)mSplitLabelCount );
                Core::Printf( "Remaining centroid at ", mCentroid.x, "x", mCentroid.y, "." );

			    Core::Printf( "\nFinished Splitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") to new segmentation label ", newId, "...\n" );

		    }

            newCentroids.push_back( std::pair< float2, int>( mCentroid, segId ) );

            //
            // Record centroids for next z
            //
            mPrevSplitId = segId;
            mPrevSplitZ = currentZ;
            mPrevSplitLine = mSplitStates.find( currentZ )->second.splitLine;
            mPrevSplitCentroids = newCentroids;

            currentZ = currentZ + direction;
            if ( direction == 0 || !join3D || mSplitStates.find( currentZ ) == mSplitStates.end() )
            {
                continueZ = false;
            }
            else
            {
                float3 newPointTileSpace = pointTileSpace;
                newPointTileSpace.z = (float) currentZ;
                PrepForSplit( segId, newPointTileSpace );
                PredictSplit( segId, newPointTileSpace, 0 );
            }

        }

    }

    //
    // Reset the 3D split state
    //
    mSplitStates.clear();

	//
	// Prep for more splitting
	//
	LoadSplitDistances( segId );
    ResetSplitState();

	return newId;

}

void FileSystemTileServer::CommitAdjustChange( int segId, float3 pointTileSpace )
{
	if ( mIsSegmentationLoaded )
    {
        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

        int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
		int4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;
        int* currentIdVolume;
        int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

        std::set< int > oldIds;
        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( segId );

		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = segId;
		mNextUndoItem->oldId = -1;
        std::set< int2, Core::Int2Comparator > *changeSet;

        int currentW = 0;
        std::queue< int4 > tileQueue;
        std::queue< int4 > wQueue;

        int tileCount = 0;
        int areaCount = 0;

        bool tileLoaded;
        bool tileChanged;

        //
        // Loop over buffer tiles at w=0
        //
        for ( int xd = 0; xd < mSplitWindowNTiles.x; ++xd )
        {
            for (int yd = 0; yd < mSplitWindowNTiles.y; ++yd )
            {

                ++tileCount;
                //Core::Printf( "Loading distance tile ", tileCount, "." );

                int4 tileIndex = make_int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
                volumeDescriptions = LoadTile( tileIndex );
                tileChanged = false;
                currentIdVolume = (int*)volumeDescriptions.Get( "IdMap" ).data;

                //
                // Copy distance values into the buffer
                //
                int xOffset = xd * numVoxelsPerTile.x;
                int yOffset = yd * numVoxelsPerTile.y;

                for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
                {
                    int areaX = xOffset + tileIndex1D % numVoxelsPerTile.x;
                    int areaY = yOffset + tileIndex1D / numVoxelsPerTile.x;
                    int areaIndex1D = areaX + areaY * mSplitWindowWidth;

                    RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

                    if ( currentIdVolume[ tileIndex1D ] != segId && mSplitDrawArea[ areaIndex1D ] == REGION_A )
                    {
                        if ( !tileChanged )
                        {
                            //
                            // Create the undo change set for this tile
                            //
					        changeSet = 
						        &mNextUndoItem->changeSets.GetHashMap()[ CreateTileString( tileIndex ) ];
                        }

                        changeSet->insert( make_int2( tileIndex1D, currentIdVolume[ tileIndex1D ] ) );
                        oldIds.insert( currentIdVolume[ tileIndex1D ] );
                        currentIdVolume[ tileIndex1D ] = segId;
                        tileChanged = true;

                        if ( tilesContainingNewId.find( tileIndex ) == tilesContainingNewId.end() )
                        {
                            tilesContainingNewId.insert( tileIndex );
                        }

                        //
					    // Add a scaled-down w to the queue
					    //
                        wQueue.push( make_int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + areaX ) / 2,
                            ( mSplitWindowStart.y * numVoxelsPerTile.y + areaY ) / 2, mSplitWindowStart.z, currentW + 1) );
                    }
                }

                if ( tileChanged )
                {
                    //
                    // Check overwritten ids to see if they should be removed from the idTileMap
                    //
                    for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
                    {
                        std::set< int >::iterator matchIt = oldIds.find( currentIdVolume[ tileIndex1D ] );
                        if ( matchIt != oldIds.end() )
                        {
                            oldIds.erase( matchIt );

                            if ( oldIds.size() == 0 )
                                break;
                        }
                    }

                    for ( std::set< int >::iterator removedIt = oldIds.begin(); removedIt != oldIds.end(); ++removedIt )
                    {
                        mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ *removedIt ].insert( tileIndex );

                        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( *removedIt );
                        tilesContainingOldId.erase( tileIndex );
                        mSegmentInfoManager.SetTiles( *removedIt, tilesContainingOldId );
                    }

                    SaveTile( tileIndex, volumeDescriptions );
                }
                UnloadTile( tileIndex );
                oldIds.clear();

            }
        }


        //
		// fill up in w
		//

        currentW = currentW + 1;
        std::swap(tileQueue, wQueue);

		tileLoaded = false;
		tileChanged = false;

        int4 thisVoxel;
        int4 tileIndex;
		int4 previousTileIndex;
        int3 currentIdNumVoxels;

        oldIds.clear();

		while ( currentW < numTiles.w )
		{
			Core::Printf( "Adjusting at w=", currentW, "." );
			while ( tileQueue.size() > 0 )
			{
				thisVoxel = tileQueue.front();
				tileQueue.pop();

				//
				// Find the tile for this pixel
				//
				tileIndex = make_int4( thisVoxel.x / numVoxelsPerTile.x,
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
								mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
							}
							tilesContainingNewId.insert( previousTileIndex );

                            //
                            // Check overwritten ids to see if they should be removed from the idTileMap
                            //
                            for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
                            {
                                std::set< int >::iterator matchIt = oldIds.find( currentIdVolume[ tileIndex1D ] );
                                if ( matchIt != oldIds.end() )
                                {
                                    oldIds.erase( matchIt );

                                    if ( oldIds.size() == 0 )
                                        break;
                                }
                            }

                            for ( std::set< int >::iterator removedIt = oldIds.begin(); removedIt != oldIds.end(); ++removedIt )
                            {
                                mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ *removedIt ].insert( tileIndex );

                                FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( *removedIt );
                                tilesContainingOldId.erase( tileIndex );
                                mSegmentInfoManager.SetTiles( *removedIt, tilesContainingOldId );
                            }

						}
						UnloadTile( previousTileIndex );
                        oldIds.clear();
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

				}
				int tileX = thisVoxel.x % numVoxelsPerTile.x;
				int tileY = thisVoxel.y % numVoxelsPerTile.y;

				int3 index3D = make_int3( tileX, tileY, 0 );
				int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
				int  idValue = currentIdVolume[ index1D ];

                if ( currentIdVolume[ index1D ] != segId )
                {
                    if ( !tileChanged )
                    {
                        //
                        // Create the undo change set for this tile
                        //
					    changeSet = 
						    &mNextUndoItem->changeSets.GetHashMap()[ CreateTileString( tileIndex ) ];
                    }

                    changeSet->insert( make_int2( index1D, currentIdVolume[ index1D ] ) );
                    oldIds.insert( currentIdVolume[ index1D ] );
                    currentIdVolume[ index1D ] = segId;
                    tileChanged = true;

                    if ( tilesContainingNewId.find( tileIndex ) == tilesContainingNewId.end() )
                    {
                        tilesContainingNewId.insert( tileIndex );
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
					mNextUndoItem->idTileMapAddNewId.insert ( previousTileIndex );
				}
                tilesContainingNewId.insert( previousTileIndex );

                //
                // Check overwritten ids to see if they should be removed from the idTileMap
                //
                for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
                {
                    std::set< int >::iterator matchIt = oldIds.find( currentIdVolume[ tileIndex1D ] );
                    if ( matchIt != oldIds.end() )
                    {
                        oldIds.erase( matchIt );

                        if ( oldIds.size() == 0 )
                            break;
                    }
                }

                for ( std::set< int >::iterator removedIt = oldIds.begin(); removedIt != oldIds.end(); ++removedIt )
                {
                    mNextUndoItem->idTileMapRemoveOldIdSets.GetHashMap()[ *removedIt ].insert( tileIndex );

                    FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( *removedIt );
                    tilesContainingOldId.erase( tileIndex );
                    mSegmentInfoManager.SetTiles( *removedIt, tilesContainingOldId );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
		// Update idTileMap
		//
        mSegmentInfoManager.SetTiles( segId, tilesContainingNewId );

        Core::Printf( "\nFinished Adjusting segmentation label ", segId, " in tile z=", pointTileSpace.z, ".\n" );

    }
}

void FileSystemTileServer::FindBoundaryJoinPoints2D( int segId )
{
	//
	// Find a splitting line that links all the given points
	// Save the result into a temp tile
	//
    if ( mIsSegmentationLoaded && mSplitSourcePoints.size() > 0 )
    {
		Core::Printf( "\nFinding Split line for segment ", segId, " with ", (unsigned int) mSplitSourcePoints.size(), " split segments.\n" );

        int3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

        int nSources = (int) mSplitSourcePoints.size();
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
		//memset( mSplitResultArea, 0, mSplitWindowNPix );
		//memcpy( mSplitSearchMask, mSplitBorderTargets, mSplitWindowNPix );

		for ( int i = 0; i < mSplitWindowNPix; ++i )
        {
			mSplitSearchMask[ i ] = mSplitBorderTargets[ i ];
            mSplitResultArea[ i ] = 0;
        }

		//ResetOverlayTiles();

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
                    mSplitSearchMask[ areaIndex ] = 0;
                }
                else
                {
                    mSplitSearchMask[ areaIndex ] = SOURCE_TARGET;
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
                // allow link to border on the last step
                //
                targetMax = nLinks == nSources * 2 - 1 ? BORDER_TARGET : SOURCE_TARGET;
            }

            int toIndex = -1;

            Mojo::Native::SimpleSplitTools::DijkstraSearch( mSplitStepDist, mSplitSearchMask, mSplitDrawArea, sourceIndex, mSplitWindowWidth, mSplitWindowHeight, targetMax, mSplitResultDist, mSplitPrev, &toIndex );

			if ( toIndex != -1 )
			{
				++nLinks;
                ++sourceLinks[ currentSourceIx ];

				//
				// Check the target type
				//
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

				//
			    // Get the result line and mask around it
			    //
				for ( int i = 0; i < mSplitWindowNPix; ++i )
				{
					// debug mask
					//mSplitResultArea[ i ] = mSplitSearchMask[ i ];

					if ( mSplitPrev[ i ] == -PATH_RESULT )
					{
						mSplitResultArea[ i ] = PATH_RESULT;
						SimpleSplitTools::ApplySmallMask( i, mSplitWindowWidth, mSplitWindowHeight, MASK_VALUE, mSplitSearchMask );
					}
				}

				//
				// Unmask the sources
				//
                for ( int si = 0; si < nSources; ++si )
                {
                    int areaIndex = sourceLocations[ si ];
					mSplitResultArea[ areaIndex ] = SOURCE_TARGET;
                    SimpleSplitTools::ApplyLargeMask( areaIndex, mSplitWindowWidth, mSplitWindowHeight, 0, mSplitSearchMask );
                }

			}
			else
			{
				Core::Printf( "WARNING: Could not find shortest path in FindBoundaryJoinPoints2D." );
				break;
			}

		}

		UpdateOverlayTiles();

        delete[] sourceLinks;
        delete[] sourceLocations;

		Core::Printf( "\nFinished splitting label ", segId, ".\n" );

	}
}

void FileSystemTileServer::FindBoundaryWithinRegion2D( int segId )
{
	//
	// Watershed drawn region to highest boundaries
	//
	if ( mIsSegmentationLoaded )
	{
		Core::Printf( "\nFindBoundaryWithinRegion2D: Splitting label ", segId, ".\n" );

		for ( int i = 0; i < mSplitWindowNPix; ++i )
        {
			mSplitSearchMask[ i ] = 0;
            mSplitResultArea[ i ] = 0;
        }

		mSplitNPerimiters = 0;
		std::set< int > perimiterSearchSet;
		std::multimap< int, int > waterPixels;

		for ( int i = 0; i < mSplitWindowNPix; ++i )
        {
			int areaX = i % mSplitWindowWidth;
			int areaY = i / mSplitWindowWidth;

			//
			// Label perimiter pixels just outside the drawn region and watershed seed pixels just inside the draw region
			//
			if ( mSplitDrawArea[ i ] != REGION_SPLIT &&
				mSplitSearchMask[ i ] == 0 &&
				mSplitBorderTargets[ i ] != BORDER_TARGET &&
				( ( areaX > 0 && mSplitDrawArea[ i - 1 ] == REGION_SPLIT ) ||
				( areaX < mSplitWindowWidth - 1 && mSplitDrawArea[ i + 1 ] == REGION_SPLIT ) ||
				( areaY > 0 && mSplitDrawArea[ i - mSplitWindowWidth ] == REGION_SPLIT ) ||
				( areaY < mSplitWindowHeight - 1 && mSplitDrawArea[ i + mSplitWindowWidth ] == REGION_SPLIT ) ) )
			{
				//
				// new perimiter label
				//
				++mSplitNPerimiters;
				perimiterSearchSet.insert( i );
				//Core::Printf( "Added new perimiterLabel=", mSplitNPerimiters, " at ", i, "." );

				while ( perimiterSearchSet.size() > 0 )
				{
					int index1D = *perimiterSearchSet.begin();
					perimiterSearchSet.erase( index1D );
					//Core::Printf( "Searching around perimiterLabel at ", index1D, "." );

					//
					// check neighbours
					//
					areaX = index1D % mSplitWindowWidth;
					areaY = index1D / mSplitWindowWidth;

					if ( mSplitDrawArea[ index1D ] != REGION_SPLIT &&
						mSplitSearchMask[ index1D ] == 0 &&
						mSplitBorderTargets[ index1D ] != BORDER_TARGET &&
						( ( areaX > 0 && mSplitDrawArea[ index1D - 1 ] == REGION_SPLIT ) ||
						( areaX < mSplitWindowWidth - 1 && mSplitDrawArea[ index1D + 1 ] == REGION_SPLIT ) ||
						( areaY > 0 && mSplitDrawArea[ index1D - mSplitWindowWidth ] == REGION_SPLIT ) ||
						( areaY < mSplitWindowHeight - 1 && mSplitDrawArea[ index1D + mSplitWindowWidth ] == REGION_SPLIT ) ) )
					{
						//
						// This is a new boundary edge - mark it and check any 8-connected neighbours
						//
						mSplitSearchMask[ index1D ] = mSplitNPerimiters;

						//DEBUG
						//mSplitResultArea[ index1D ] = mSplitNPerimiters + MASK_VALUE;

						//Core::Printf( "New boundary edge - adding neighbours." );

						if ( areaX > 0 )
							perimiterSearchSet.insert( index1D - 1 );
						if ( areaX < mSplitWindowWidth - 1)
							perimiterSearchSet.insert( index1D + 1 );
						if ( areaY > 0 )
							perimiterSearchSet.insert( index1D - mSplitWindowWidth );
						if ( areaY < mSplitWindowHeight - 1)
							perimiterSearchSet.insert( index1D + mSplitWindowWidth );

						if ( areaX > 0 && areaY > 0)
							perimiterSearchSet.insert( index1D - 1 - mSplitWindowWidth );
						if ( areaX > 0 && areaY < mSplitWindowHeight - 1)
							perimiterSearchSet.insert( index1D - 1 + mSplitWindowWidth );
						if ( areaX < mSplitWindowWidth - 1 && areaY > 0 )
							perimiterSearchSet.insert( index1D + 1 - mSplitWindowWidth );
						if ( areaX < mSplitWindowWidth - 1 && areaY < mSplitWindowHeight - 1 )
							perimiterSearchSet.insert( index1D + 1 + mSplitWindowWidth );

					}
					else if ( mSplitDrawArea[ index1D ] == REGION_SPLIT &&
						mSplitSearchMask[ index1D ] == 0 )
					{
						//
						// Add this pixel to the watershed grow region
						//
						waterPixels.insert( std::pair< int, int >( mSplitStepDist[ index1D ], index1D ) );
						//Core::Printf( "New watershed pixel." );
					}
				}
			}
        }

		//
		// Watershed from the perimiters
		//
		while ( waterPixels.size() > 0 )
		{
			std::multimap< int, int >::iterator mapIt = --waterPixels.end();
			int index1D = mapIt->second;

			//Core::Printf( "Got new watershed pixel at ", index1D, ", score=", mapIt->first, "." );

			waterPixels.erase( mapIt );

			if ( mSplitDrawArea[ index1D ] == REGION_SPLIT &&
				mSplitSearchMask[ index1D ] == 0 &&
				mSplitResultArea[ index1D ] == 0 &&
				mSplitBorderTargets[ index1D ] != BORDER_TARGET )
			{

				//
				// This pixel is a border if any 8-connected neighbours have different labels
				// otherwise label this pixel the same as its neighbours
				//

				int areaX = index1D % mSplitWindowWidth;
				int areaY = index1D / mSplitWindowWidth;

				int minX, maxX, minY, maxY;

				if ( areaX > 0 )
					minX = areaX - 1;
				else
					minX = areaX;

				if ( areaX < mSplitWindowWidth - 1 )
					maxX = areaX + 1;
				else
					maxX = areaX;

				if ( areaY > 0 )
					minY = areaY - 1;
				else
					minY = areaY;

				if ( areaY < mSplitWindowHeight - 1 )
					maxY = areaY + 1;
				else
					maxY = areaY;

				int foundLabel = 0;
				bool isBorder = false;

				for ( int x = minX; x <= maxX; ++x )
				{
					for ( int y = minY; y <= maxY; ++y )
					{
						//
						// only check 4-connected pixels for labels
						//
						if ( ( x == areaX && y == areaY ) ||
							( x != areaX && y != areaY ) )
							continue;

						int nLabel = mSplitSearchMask[ x + y * mSplitWindowWidth ];

						if ( nLabel != 0 )
						{
							if ( foundLabel == 0 )
							{
								foundLabel = nLabel;
							}
							else if ( foundLabel != nLabel )
							{
								isBorder = true;
								mSplitResultArea[ index1D ] = PATH_RESULT;
							}
						}
					}
				}

				if ( !isBorder && foundLabel != 0 )
				{
					//
					// label this pixel
					//
					mSplitSearchMask[ index1D ] = foundLabel;

					//DEBUG
					//mSplitResultArea[ index1D ] = foundLabel + MASK_VALUE;

					//
					// add 4-connected neighbours to the watershed grow region
					//
					for ( int x = minX; x <= maxX; ++x )
					{
						for ( int y = minY; y <= maxY; ++y )
						{
							//
							// only check 4-connected pixels for labels
							//
							if ( ( x == areaX && y == areaY ) ||
								( x != areaX && y != areaY ) )
								continue;

							int addIndex1D = x + y * mSplitWindowWidth;
							if ( mSplitDrawArea[ addIndex1D ] == REGION_SPLIT &&
								mSplitSearchMask[ addIndex1D ] == 0 &&
								mSplitResultArea[ addIndex1D ] == 0 &&
								mSplitBorderTargets[ addIndex1D ] != BORDER_TARGET )
							{
								waterPixels.insert( std::pair< int, int >( mSplitStepDist[ addIndex1D ], addIndex1D ) );
								//Core::Printf( "Added watershed neighbour at ", addIndex1D, "." );
							}
						}
					}
				}
			}

		}

		UpdateOverlayTiles();

		Core::Printf( "\nFindBoundaryWithinRegion2D: Finished splitting label ", segId, ".\n" );

	}
}

void FileSystemTileServer::FindBoundaryBetweenRegions2D( int segId )
{
	//
	// Watershed between drawn regions to highest boundaries
	//
	if ( mIsSegmentationLoaded )
	{
		Core::Printf( "\nFindBoundaryBetweenRegions2D: Splitting label ", segId, ".\n" );

		for ( int i = 0; i < mSplitWindowNPix; ++i )
        {
			mSplitSearchMask[ i ] = 0;
            mSplitResultArea[ i ] = 0;
        }

		mSplitNPerimiters = 2;
		std::multimap< int, int > waterPixels;

		for ( int i = 0; i < mSplitWindowNPix; ++i )
        {
			int areaX = i % mSplitWindowWidth;
			int areaY = i / mSplitWindowWidth;

			//
			// Label perimiter pixels just inside the drawn region and watershed seed pixels just outside the draw region
			//
			if ( mSplitDrawArea[ i ] != 0 &&
				mSplitSearchMask[ i ] == 0 &&
				mSplitBorderTargets[ i ] != BORDER_TARGET )
			{
				if ( ( areaX > 0 && mSplitDrawArea[ i - 1 ] != mSplitDrawArea[ i ] ) ||
				( areaX < mSplitWindowWidth - 1 && mSplitDrawArea[ i + 1 ] != mSplitDrawArea[ i ] ) ||
				( areaY > 0 && mSplitDrawArea[ i - mSplitWindowWidth ] != mSplitDrawArea[ i ] ) ||
				( areaY < mSplitWindowHeight - 1 && mSplitDrawArea[ i + mSplitWindowWidth ] != mSplitDrawArea[ i ] ) )
				{
					//
					// Perimiter - watershed from here
					//
					waterPixels.insert( std::pair< int, int >( mSplitStepDist[ i ], i ) );
				}
				else
				{
					mSplitSearchMask[ i ] = mSplitDrawArea[ i ] == REGION_A ? 1 : 2;
				}
			}
        }

		//
		// Watershed from the perimiters
		//
		while ( waterPixels.size() > 0 )
		{
			std::multimap< int, int >::iterator mapIt = --waterPixels.end();
			int index1D = mapIt->second;

			//Core::Printf( "Got new watershed pixel at ", index1D, ", score=", mapIt->first, "." );

			waterPixels.erase( mapIt );

			if ( mSplitSearchMask[ index1D ] == 0 &&
				mSplitResultArea[ index1D ] == 0 &&
				mSplitBorderTargets[ index1D ] != BORDER_TARGET )
			{

				//
				// This pixel is a border if any 8-connected neighbours have different labels
				// otherwise label this pixel the same as its neighbours
				//

				int areaX = index1D % mSplitWindowWidth;
				int areaY = index1D / mSplitWindowWidth;

				int minX, maxX, minY, maxY;

				if ( areaX > 0 )
					minX = areaX - 1;
				else
					minX = areaX;

				if ( areaX < mSplitWindowWidth - 1 )
					maxX = areaX + 1;
				else
					maxX = areaX;

				if ( areaY > 0 )
					minY = areaY - 1;
				else
					minY = areaY;

				if ( areaY < mSplitWindowHeight - 1 )
					maxY = areaY + 1;
				else
					maxY = areaY;

				int foundLabel = 0;
				bool isBorder = false;

				for ( int x = minX; x <= maxX; ++x )
				{
					for ( int y = minY; y <= maxY; ++y )
					{
						//
						// only check 4-connected pixels for labels
						//
						if ( ( x == areaX && y == areaY ) ||
							( x != areaX && y != areaY ) )
							continue;

						int nLabel = mSplitSearchMask[ x + y * mSplitWindowWidth ];

						if ( nLabel != 0 )
						{
							if ( foundLabel == 0 )
							{
								foundLabel = nLabel;
							}
							else if ( foundLabel != nLabel )
							{
								isBorder = true;
								mSplitResultArea[ index1D ] = PATH_RESULT;
							}
						}
					}
				}

				if ( !isBorder && foundLabel != 0 )
				{
					//
					// label this pixel
					//
					mSplitSearchMask[ index1D ] = foundLabel;

					//DEBUG
					//mSplitResultArea[ index1D ] = foundLabel + MASK_VALUE;

					//
					// add 4-connected neighbours to the watershed grow region
					//
					for ( int x = minX; x <= maxX; ++x )
					{
						for ( int y = minY; y <= maxY; ++y )
						{
							//
							// only check 4-connected pixels for labels
							//
							if ( ( x == areaX && y == areaY ) ||
								( x != areaX && y != areaY ) )
								continue;

							int addIndex1D = x + y * mSplitWindowWidth;
							if ( mSplitSearchMask[ addIndex1D ] == 0 &&
								mSplitResultArea[ addIndex1D ] == 0 &&
								mSplitBorderTargets[ addIndex1D ] != BORDER_TARGET )
							{
								waterPixels.insert( std::pair< int, int >( mSplitStepDist[ addIndex1D ], addIndex1D ) );
								//Core::Printf( "Added watershed neighbour at ", addIndex1D, "." );
							}
						}
					}
				}
			}

		}

		UpdateOverlayTiles();

		Core::Printf( "\nFindBoundaryBetweenRegions2D: Finished splitting label ", segId, ".\n" );

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
		return;
	}

    if ( mUndoDeque.size() > 0 )
    {

        FileSystemUndoRedoItem UndoItem = mUndoDeque.front();

	    int oldId = UndoItem.oldId;
	    int newId = UndoItem.newId;

	    if ( newId != 0 && oldId != 0 && newId != oldId && mIsSegmentationLoaded )
	    {
            Core::Printf( "\nUndo operation: changing segmentation label ", newId, " back to segmentation label ", oldId, "...\n" );
		    stdext::hash_map < std::string, std::bitset< TILE_SIZE * TILE_SIZE > >::iterator changeIt;

		    for ( changeIt = UndoItem.changePixels.GetHashMap().begin(); changeIt != UndoItem.changePixels.GetHashMap().end(); ++changeIt )
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
			    std::bitset< TILE_SIZE * TILE_SIZE > *changeBits = &changeIt->second;

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

		    stdext::hash_map< std::string, std::set< int2, Core::Int2Comparator > >::iterator changeSetIt;

		    for ( changeSetIt = UndoItem.changeSets.GetHashMap().begin(); changeSetIt != UndoItem.changeSets.GetHashMap().end(); ++changeSetIt )
		    {

			    int4 tileIndex = CreateTileIndex( changeSetIt->first );

                //
                // load tile
                //
                Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
                int* currentIdVolume    = (int*)volumeDescriptions.Get( "IdMap" ).data;

			    //
			    // Get the changes
			    //
			    std::set< int2, Core::Int2Comparator > *changeBits = &changeSetIt->second;

                //
                // replace the new id and color with the previous id and color...
                //
                for ( std::set< int2, Core::Int2Comparator >::iterator indexIt = changeSetIt->second.begin(); indexIt != changeSetIt->second.end(); ++indexIt )
                {
                    currentIdVolume[ indexIt->x ] = indexIt->y;
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
            FileSystemTileSet newTiles = mSegmentInfoManager.GetTiles( newId );
		    for ( FileSystemTileSet::iterator eraseIterator = UndoItem.idTileMapAddNewId.begin(); eraseIterator != UndoItem.idTileMapAddNewId.end(); ++eraseIterator )
		    {
			    newTiles.erase( *eraseIterator );
		    }
            mSegmentInfoManager.SetTiles( newId, newTiles );

		    //
		    // put removed tiles back into the "oldId" idTileMap (create a new idTileMap if necessary)
		    //
            for ( std::hash_map< int, FileSystemTileSet >::iterator oldIdIt = UndoItem.idTileMapRemoveOldIdSets.GetHashMap().begin(); oldIdIt != UndoItem.idTileMapRemoveOldIdSets.GetHashMap().end(); ++oldIdIt )
            {
                FileSystemTileSet oldTiles = mSegmentInfoManager.GetTiles( oldIdIt->first );
                oldTiles.insert( oldIdIt->second.begin(), oldIdIt->second.end() );
                mSegmentInfoManager.SetTiles( oldIdIt->first , oldTiles );
            }

            Core::Printf( "\nUndo operation complete: changed segmentation label ", newId, " back to segmentation label ", oldId, ".\n" );

		    //
		    // Make this a redo item
		    //
            mUndoDeque.pop_front();
            mRedoDeque.push_front( UndoItem );
            mNextUndoItem = &mUndoDeque.front();
        }
	}
}

void FileSystemTileServer::RedoChange()
{

    if ( mRedoDeque.size() > 0 )
    {
        FileSystemUndoRedoItem RedoItem = mRedoDeque.front();

	    int oldId = RedoItem.oldId;
	    int newId = RedoItem.newId;

	    if ( newId != 0 && oldId != 0 && newId != oldId && mIsSegmentationLoaded )
	    {
            Core::Printf( "\nRedo operation: changing segmentation label ", oldId, " back to segmentation label ", newId, "...\n" );
		    stdext::hash_map < std::string, std::bitset< TILE_SIZE * TILE_SIZE > >::iterator changeIt;

		    for ( changeIt = RedoItem.changePixels.GetHashMap().begin(); changeIt != RedoItem.changePixels.GetHashMap().end(); ++changeIt )
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
			    std::bitset< TILE_SIZE * TILE_SIZE > *changeBits = &changeIt->second;

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

		    stdext::hash_map< std::string, std::set< int2, Core::Int2Comparator > >::iterator changeSetIt;

		    for ( changeSetIt = RedoItem.changeSets.GetHashMap().begin(); changeSetIt != RedoItem.changeSets.GetHashMap().end(); ++changeSetIt )
		    {

			    int4 tileIndex = CreateTileIndex( changeSetIt->first );

                //
                // load tile
                //
                Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
                int* currentIdVolume    = (int*)volumeDescriptions.Get( "IdMap" ).data;

			    //
			    // Get or create the change bitset for this tile
			    //
			    std::set< int2, Core::Int2Comparator > *changeBits = &changeSetIt->second;

                //
                // replace the old id and color with the new id and color...
                //
                for ( std::set< int2, Core::Int2Comparator >::iterator indexIt = changeSetIt->second.begin(); indexIt != changeSetIt->second.end(); ++indexIt )
                {
                    currentIdVolume[ indexIt->x ] = newId;
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
            FileSystemTileSet newTiles = mSegmentInfoManager.GetTiles( newId );
            newTiles.insert( RedoItem.idTileMapAddNewId.begin(), RedoItem.idTileMapAddNewId.end() );
            mSegmentInfoManager.SetTiles( newId, newTiles );

            //
            // remove tiles from the "oldId" idTileMap
            //
            for ( std::hash_map< int, FileSystemTileSet >::iterator oldIdIt = RedoItem.idTileMapRemoveOldIdSets.GetHashMap().begin(); oldIdIt != RedoItem.idTileMapRemoveOldIdSets.GetHashMap().end(); ++oldIdIt )
            {
                FileSystemTileSet oldTiles = mSegmentInfoManager.GetTiles( oldIdIt->first );
                for ( FileSystemTileSet::iterator eraseIterator = oldIdIt->second.begin(); eraseIterator != oldIdIt->second.end(); ++eraseIterator )
		        {
			        oldTiles.erase( *eraseIterator );
		        }
                mSegmentInfoManager.SetTiles( oldIdIt->first, oldTiles );
            }

            Core::Printf( "\nRedo operation complete: changed segmentation label ", oldId, " back to segmentation label ", newId, ".\n" );

		    //
		    // Make this a redo item
		    //
            mRedoDeque.pop_front();
            mUndoDeque.push_front( RedoItem );
            mNextUndoItem = &mUndoDeque.front();
	    }
    }
}

void FileSystemTileServer::PrepForNextUndoRedoChange()
{
    //
    // Clear the redo items
    //
    mRedoDeque.clear();

    //
    // Add a new undo item
    //
    mUndoDeque.push_front( FileSystemUndoRedoItem() );

    mNextUndoItem = &mUndoDeque.front();

    //
    // Clear undo items from the back
    //
    while( mUndoDeque.size() > MAX_UNDO_OPERATIONS )
    {
        mUndoDeque.pop_back();
    }
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
    //
    // release id maps
    //
    mSegmentInfoManager.CloseDB();
    mSegmentInfoManager = FileSystemSegmentInfoManager();
    mTiledDatasetDescription.maxLabelId = 0;

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

    clock_t cutoff = timestamps[mFileSystemTileCache.GetHashMap().size() / 4];

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

marray::Marray< unsigned char > FileSystemTileServer::GetIdColorMap()
{
    return mSegmentInfoManager.GetIdColorMap();
}


}
}