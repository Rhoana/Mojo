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

using namespace Mojo::Core;

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

    mPrevSplitId = 0;
    mPrevSplitZ = -2;

	mCurrentOperationProgress = 0;
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
        LoadTiledDatasetInternal( tiledDatasetDescription );
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
        LoadSegmentationInternal( tiledDatasetDescription );
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

    TempSaveFileSystemTileCacheChanges();

    //
    // move changed tiles to the save directory
    //

    Core::Printf( "Replacing tiles." );

    MojoInt4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles();

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

    Core::Printf( "Segmentation saved." );

	mLogger.Log( "Segmentation saved." );

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

	mLogger.Log( Core::ToString( "Segmentation saved to: \"", savePath, "\" (disabled)." ) );

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

                TempSaveFileSystemTileCacheChanges();

                //
                // move changed tiles to the autosave directory
                //

                Core::Printf( "Autosave copying all modified tiles." );

                MojoInt4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles();

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

    MojoInt4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).numTiles();

    TiledVolumeDescription tempVolumeDesctiption = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "TempIdMap" );
    TiledVolumeDescription autosaveVolumeDesctiption = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

    boost::filesystem::path tempFolder = boost::filesystem::path( tempVolumeDesctiption.imageDataDirectory );
    if ( boost::filesystem::exists( tempFolder ) )
    {
		boost::filesystem::remove_all( tempFolder );
    }

}


int FileSystemTileServer::GetTileCountForId( unsigned int segId )
{
    return (int) mSegmentInfoManager.GetTileCount( segId );
}

MojoInt3 FileSystemTileServer::GetSegmentCentralTileLocation( unsigned int segId )
{
    if ( mIsSegmentationLoaded )
    {
        MojoInt4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles();

        int minTileX = numTiles.x;
        int maxTileX = 0;
        int minTileY = numTiles.y;
        int maxTileY = 0;
        int minTileZ = numTiles.z;
        int maxTileZ = 0;

        FileSystemTileSet tilesContainingSegId = mSegmentInfoManager.GetTiles( segId );

        for ( FileSystemTileSet::iterator tileIterator = tilesContainingSegId.begin(); tileIterator != tilesContainingSegId.end(); ++tileIterator )
        {
            if ( tileIterator->w == 0 )
            {
                //
                // Include this tile
                //
                minTileX = MIN( minTileX, tileIterator->x );
                maxTileX = MAX( maxTileX, tileIterator->x );
                minTileY = MIN( minTileY, tileIterator->y );
                maxTileY = MAX( maxTileY, tileIterator->y );
                minTileZ = MIN( minTileZ, tileIterator->z );
                maxTileZ = MAX( maxTileZ, tileIterator->z );
            }
        }

        if ( minTileX > maxTileX || minTileY > maxTileY || minTileZ > maxTileZ )
	    {
		    //
		    // No tile found at w=0
		    //
		    return MojoInt3( 0, 0, 0 );
	    }

        return MojoInt3( minTileX + ( ( maxTileX - minTileX ) / 2 ), minTileY + ( ( maxTileY - minTileY ) / 2 ), minTileZ + ( ( maxTileZ - minTileZ ) / 2 ) );

    }

    return MojoInt3( 0, 0, 0 );

}

MojoInt4 FileSystemTileServer::GetSegmentZTileBounds( unsigned int segId, int zIndex )
{
    if ( mIsSegmentationLoaded )
    {
        MojoInt4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles();

        int minTileX = numTiles.x;
        int maxTileX = 0;
        int minTileY = numTiles.y;
        int maxTileY = 0;

        FileSystemTileSet tilesContainingSegId = mSegmentInfoManager.GetTiles( segId );

        for ( FileSystemTileSet::iterator tileIterator = tilesContainingSegId.begin(); tileIterator != tilesContainingSegId.end(); ++tileIterator )
        {
            if ( tileIterator->w == 0 && tileIterator->z == zIndex )
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
		    // No tile found at w=0
		    //
		    return MojoInt4( 0, 0, 0, 0 );
	    }

        return MojoInt4( minTileX, minTileY, maxTileX, maxTileY );

    }

    return MojoInt4( 0, 0, 0, 0 );

}


//
// Edit Methods
//

void FileSystemTileServer::RemapSegmentLabels( std::set< unsigned int > fromSegIds, unsigned int toSegId )
{
	if ( mIsSegmentationLoaded && toSegId != 0 && mSegmentInfoManager.GetConfidence( toSegId ) < 100 )
	{
		Core::Printf( "\nRemapping ", (int)fromSegIds.size(), " segmentation labels to segmentation label ", toSegId, "...\n" );
		FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( toSegId );

		//
		// Prepare the undo / redo item
		//
		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = toSegId;
		//mNextUndoItem->oldId = 0;

		for ( std::set< unsigned int >::iterator fromIt = fromSegIds.begin(); fromIt != fromSegIds.end(); ++fromIt )
		{
			unsigned int fromSegId = *fromIt;

			if ( fromSegId != 0 && fromSegId != toSegId && mSegmentInfoManager.GetConfidence( fromSegId ) < 100 )
			{
				//Core::Printf( "\nRemapping segmentation label ", fromSegId, " to segmentation label ", toSegId, "...\n" );

				FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( fromSegId );

				//
				// Keep track of tile operations
				//
				mNextUndoItem->idTileMapRemoveOldIdSets[ fromSegId ].insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );

				for ( FileSystemTileSet::iterator addIterator = tilesContainingOldId.begin(); addIterator != tilesContainingOldId.end(); ++addIterator )
				{
					if ( tilesContainingNewId.find( *addIterator ) == tilesContainingNewId.end() )
					{
						mNextUndoItem->idTileMapAddNewId.insert( *addIterator );
					}
				}

				//
				// Remap
				//
				mSegmentInfoManager.RemapSegmentLabel( fromSegId, toSegId );

				//
				// Update the segment sizes
				//
				long voxelChangeCount = mSegmentInfoManager.GetVoxelCount( fromSegId );
				mSegmentInfoManager.SetVoxelCount( toSegId, mSegmentInfoManager.GetVoxelCount( toSegId ) + voxelChangeCount );
				mSegmentInfoManager.SetVoxelCount( fromSegId, 0 );

				mNextUndoItem->remapFromIdsAndSizes[ fromSegId ] = voxelChangeCount;

				//
				// add all the tiles containing old id to the list of tiles corresponding to the new id
				//
				tilesContainingNewId.insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );
				mSegmentInfoManager.SetTiles( toSegId, tilesContainingNewId );

				//
				// completely remove old id from our id tile map, since the old id is no longer present in the segmentation 
				//
				tilesContainingOldId.clear();
				mSegmentInfoManager.SetTiles( fromSegId, tilesContainingOldId );

				//Core::Printf( "Finished remapping from segmentation label ", fromSegId, " to segmentation label ", toSegId, ".\n" );
			}
		}
		Core::Printf( "Finished remapping to segmentation label ", toSegId, ".\n" );

		mLogger.Log( Core::ToString( "RemapSegmentLabels: fromId=", fromSegIds, " toId=", toSegId ) );

	}
}

void FileSystemTileServer::ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )
{
	if ( oldId != newId && mIsSegmentationLoaded && mSegmentInfoManager.GetConfidence( newId ) < 100 && mSegmentInfoManager.GetConfidence( oldId ) < 100 )
    {
        long voxelChangeCount = 0;

        Core::Printf( "\nReplacing segmentation label ", oldId, " with segmentation label ", newId, "...\n" );

		MojoInt4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles();
		MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( oldId );
		FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = newId;

		mNextUndoItem->idTileMapRemoveOldIdSets[ oldId ].insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );
		mNextUndoItem->idTileMapAddNewId.insert( tilesContainingOldId.begin(), tilesContainingOldId.end() );
		for ( FileSystemTileSet::iterator eraseIterator = tilesContainingNewId.begin(); eraseIterator != tilesContainingNewId.end(); ++eraseIterator )
		{
			mNextUndoItem->idTileMapAddNewId.erase( *eraseIterator );
		}

		int maxProgress = (int)tilesContainingOldId.size();
		int currentProgress = 0;

        for( FileSystemTileSet::iterator tileIndexi = tilesContainingOldId.begin(); tileIndexi != tilesContainingOldId.end(); ++tileIndexi )
        {

            MojoInt4 tileIndex = *tileIndexi;
			if ( tileIndex.w == 0 )
			{
				//
				// load tile
				//
				Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
				unsigned int* currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;

				//
				// Get or create the change record for this tile
				//
				TileChangeIdMap *tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];

				//
				// replace the old id and color with the new id and color...
				//
				MojoInt3 numVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
				for ( int zv = 0; zv < numVoxels.z; zv++ )
				{
					for ( int yv = 0; yv < numVoxels.y; yv++ )
					{
						for ( int xv = 0; xv < numVoxels.x; xv++ )
						{
							MojoInt3 index3D = MojoInt3( xv, yv, zv );
							int  index1D = Core::Index3DToIndex1D( index3D, numVoxels );
							int  idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ index1D ] );

							if ( idValue == oldId )
							{
								(*tileChange)[ currentIdVolume[ index1D ] ].set( index1D );
								currentIdVolume[ index1D ] = newId;
								if ( tileIndex.w == 0 )
									++voxelChangeCount;
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
				StrideUpIdTileChange( numTiles, numVoxelsPerTile, tileIndex, currentIdVolume );

				//
				// unload tile
				//
				UnloadTile( tileIndex );
			}

			++currentProgress;
			mCurrentOperationProgress = (float)currentProgress / (float)maxProgress;

        }

        //
        // Update the segment sizes
        //
        mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) + voxelChangeCount );
        mSegmentInfoManager.SetVoxelCount( oldId, mSegmentInfoManager.GetVoxelCount( oldId ) - voxelChangeCount );

        if ( mSegmentInfoManager.GetVoxelCount( oldId ) != 0 )
        {
            Core::Printf( "WARNING: Replaced all voxels belonging segment ", oldId, " but segment size is not zero (", mSegmentInfoManager.GetVoxelCount( oldId ), "). Tile index and segment database should be regenerated." );
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

		mLogger.Log( Core::ToString( "ReplaceSegmentationLabel: oldId=", oldId, ", newId=", newId ) );

		mCurrentOperationProgress = 1;

    }
}

bool FileSystemTileServer::TileContainsId ( MojoInt3 numVoxelsPerTile, MojoInt3 currentIdNumVoxels, unsigned int* currentIdVolume, unsigned int segId )
{
    bool found = false;
    int maxIndex3D = Core::Index3DToIndex1D( MojoInt3( numVoxelsPerTile.x-1, numVoxelsPerTile.y-1, 0 ), currentIdNumVoxels );
    for (int i1D = 0; i1D < maxIndex3D; ++i1D )
    {
		if ( mSegmentInfoManager.GetIdForLabel( currentIdVolume[ i1D ] ) == segId )
        {
            found = true;
            break;
        }
    }
    return found;
}

void FileSystemTileServer::ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, MojoFloat3 pointTileSpace )
{
    if ( oldId != newId && mIsSegmentationLoaded && mSegmentInfoManager.GetConfidence( newId ) < 100 && mSegmentInfoManager.GetConfidence( oldId ) < 100 )
    {
        long voxelChangeCount = 0;

        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        MojoInt3 numVoxels = tiledVolumeDescription.numVoxels();
        MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
        MojoInt4 numTiles = tiledVolumeDescription.numTiles();

        MojoInt3 pVoxelSpace = 
            MojoInt3 (
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

        Core::Printf( "\nReplacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, "...\n" );

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( oldId );
        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = newId;

		TileChangeIdMap *tileChange;

		MojoInt4 previousTileIndex;
        bool tileLoaded = false;
        bool tileChanged = false;

        int currentW = 0;
        std::queue< MojoInt4 > tileQueue;
        std::multimap< MojoInt4, MojoInt4, Mojo::Core::Int4Comparator > sliceQueue;
        //std::queue< MojoInt4 > wQueue;

        unsigned int* currentIdVolume;
        MojoInt3 currentIdNumVoxels;
        MojoInt4 thisVoxel;

		int maxProgress = 0;
		int currentProgress = 0;
		int ignoreProgressTiles = 0;

		//
		// Determine the (approximate) max amount of work to be done
		//
		for( FileSystemTileSet::iterator tileIndexi = tilesContainingOldId.begin(); tileIndexi != tilesContainingOldId.end(); ++tileIndexi )
        {
			if ( tileIndexi->z == pVoxelSpace.z )
			{
				++maxProgress;
			}
		}

		ignoreProgressTiles = (int)tilesContainingOldId.size() - maxProgress;

        tileQueue.push( MojoInt4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

        //while ( currentW < numTiles.w )
        //{
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

					currentProgress = maxProgress - ( (int)tilesContainingOldId.size() - ignoreProgressTiles );
					mCurrentOperationProgress = (float)currentProgress / (float)maxProgress;
                }

                //
				// Find the tile for this pixel
				//
                MojoInt4 tileIndex = MojoInt4( thisVoxel.x / numVoxelsPerTile.x,
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
							StrideUpIdTileChange( numTiles, numVoxelsPerTile, previousTileIndex, currentIdVolume );

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
                                mNextUndoItem->idTileMapRemoveOldIdSets[ oldId ].insert( previousTileIndex );
                            }

                        }
                        UnloadTile( previousTileIndex );
                    }

                    //
					// Load the current tile
					//
                    volumeDescriptions = LoadTile( tileIndex );
                    previousTileIndex = tileIndex;
                    currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;
                    currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
                    tileLoaded = true;
                    tileChanged = false;

					//
					// Get or create the change record for this tile
					//
					tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];
                }

                int tileX = thisVoxel.x % numVoxelsPerTile.x;
                int tileY = thisVoxel.y % numVoxelsPerTile.y;


                MojoInt3 index3D = MojoInt3( tileX, tileY, 0 );
                int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
                int  idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ index1D ] );

                if ( idValue == oldId )
                {
					(*tileChange)[ currentIdVolume[ index1D ] ].set( index1D );
                    currentIdVolume[ index1D ] = newId;
                    tileChanged = true;

                    //
					// Only do flood fill at highest resolution - the rest is done with the wQueue
					//
                    if ( currentW == 0 )
                    {
                        ++voxelChangeCount;

                        //
						// Add neighbours to the appropriate queue
						//
                        if (thisVoxel.x > 0)
                        {
                            if (tileX > 0)
                            {
                                tileQueue.push( MojoInt4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                    MojoInt4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    MojoInt4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.x < numVoxels.x - 1)
                        {
                            if (tileX < numVoxelsPerTile.x - 1)
                            {
                                tileQueue.push( MojoInt4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                    MojoInt4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    MojoInt4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }
                        if (thisVoxel.y > 0)
                        {
                            if (tileY > 0)
                            {
                                tileQueue.push( MojoInt4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                    MojoInt4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                                    MojoInt4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW ) ) );
                            }
                        }
                        if (thisVoxel.y < numVoxels.y - 1)
                        {
                            if (tileY < numVoxelsPerTile.y - 1)
                            {
                                tileQueue.push( MojoInt4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                    MojoInt4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                                    MojoInt4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW ) ) );
                            }
                        }
                    }

                    //
					// Add a scaled-down w to the queue
					//
                    //if (currentW < numTiles.w-1) wQueue.push( MojoInt4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
                }

            }
            //std::swap(tileQueue, wQueue);
            //++currentW;
        //}

        if ( tileLoaded )
        {
            //
			// Save and unload the previous tile
			//
            if ( tileChanged )
            {
                SaveTile( previousTileIndex, volumeDescriptions );
				StrideUpIdTileChange( numTiles, numVoxelsPerTile, previousTileIndex, currentIdVolume );

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
					mNextUndoItem->idTileMapRemoveOldIdSets[ oldId ].insert( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
        // Update the segment sizes
        //
        mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) + voxelChangeCount );
        mSegmentInfoManager.SetVoxelCount( oldId, mSegmentInfoManager.GetVoxelCount( oldId ) - voxelChangeCount );

        //
		// Update idTileMap
		//
        mSegmentInfoManager.SetTiles( oldId, tilesContainingOldId );
        mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

        Core::Printf( "\nFinished replacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, ".\n" );

		mLogger.Log( Core::ToString( "ReplaceSegmentationLabelCurrentSlice: oldId=", oldId, ", newId=", newId, ", x=", pVoxelSpace.x, ", y=", pVoxelSpace.y, ", z=", pVoxelSpace.z ) );

		mCurrentOperationProgress = 1;

    }
}

void FileSystemTileServer::ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, MojoFloat3 pointTileSpace )
{
    if ( oldId != newId && mIsSegmentationLoaded && mSegmentInfoManager.GetConfidence( newId ) < 100 && mSegmentInfoManager.GetConfidence( oldId ) < 100 )
    {
        long voxelChangeCount = 0;

        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        MojoInt3 numVoxels = tiledVolumeDescription.numVoxels();
        MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
        MojoInt4 numTiles = tiledVolumeDescription.numTiles();
        MojoInt3 pVoxelSpace = 
            MojoInt3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

        Core::Printf( "\nReplacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " in 3D from zslice ", pVoxelSpace.z, "...\n" );

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( oldId );
        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = newId;
		
		TileChangeIdMap *tileChange;

		MojoInt4 previousTileIndex;
        bool tileLoaded = false;
        bool tileChanged = false;

        int currentW = 0;
        std::queue< MojoInt4 > tileQueue;
        std::multimap< MojoInt4, MojoInt4, Mojo::Core::Int4Comparator > sliceQueue;
        //std::queue< MojoInt4 > wQueue;

        unsigned int* currentIdVolume;
        MojoInt3 currentIdNumVoxels;
        MojoInt4 thisVoxel;

		int maxProgress = 0;
		int currentProgress = 0;

		//
		// Determine the (approximate) max amount of work to be done
		//
		maxProgress = (int)tilesContainingOldId.size();

        tileQueue.push( MojoInt4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

        //while ( currentW < numTiles.w )
        //{
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

					currentProgress = maxProgress - (int)tilesContainingOldId.size();
					mCurrentOperationProgress = (float)currentProgress / (float)maxProgress;
                }

                //
				// Find the tile for this pixel
				//
                MojoInt4 tileIndex = MojoInt4( thisVoxel.x / numVoxelsPerTile.x,
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
							StrideUpIdTileChange( numTiles, numVoxelsPerTile, previousTileIndex, currentIdVolume );

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
								mNextUndoItem->idTileMapRemoveOldIdSets[ oldId ].insert ( previousTileIndex );
                            }

                        }
                        UnloadTile( previousTileIndex );
                    }

                    //
					// Load the current tile
					//
                    volumeDescriptions = LoadTile( tileIndex );
                    previousTileIndex = tileIndex;
                    currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;
                    currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
                    tileLoaded = true;
                    tileChanged = false;

					//
					// Get or create the change record for this tile
					//
					tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];
                }

                int tileX = thisVoxel.x % numVoxelsPerTile.x;
                int tileY = thisVoxel.y % numVoxelsPerTile.y;

                MojoInt3 index3D = MojoInt3( tileX, tileY, 0 );
                int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
				int  idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ index1D ] );

                if ( idValue == oldId )
                {
					(*tileChange)[ currentIdVolume[ index1D ] ].set( index1D );
                    currentIdVolume[ index1D ] = newId;
                    tileChanged = true;

                    //
					// Only do flood fill at highest resolution - the rest is done with the wQueue
					//
                    if ( currentW == 0 )
                    {
                        ++voxelChangeCount;

                        //
						// Add neighbours to the appropriate queue
						//
                        if (thisVoxel.x > 0)
                        {
                            if (tileX > 0)
                            {
                                tileQueue.push( MojoInt4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                    MojoInt4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    MojoInt4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.x < numVoxels.x - 1)
                        {
                            if (tileX < numVoxelsPerTile.x - 1)
                            {
                                tileQueue.push( MojoInt4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                    MojoInt4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    MojoInt4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.y > 0)
                        {
                            if (tileY > 0)
                            {
                                tileQueue.push( MojoInt4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                    MojoInt4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                                    MojoInt4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.y < numVoxels.y - 1)
                        {
                            if (tileY < numVoxelsPerTile.y - 1)
                            {
                                tileQueue.push( MojoInt4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                    MojoInt4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                                    MojoInt4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.z > 0)
                        {
                            sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                MojoInt4( tileIndex.x, tileIndex.y, tileIndex.z - 1, tileIndex.w ),
                                MojoInt4( thisVoxel.x, thisVoxel.y, thisVoxel.z - 1, currentW ) ) );
                        }

                        if (thisVoxel.z < numVoxels.z - 1)
                        {
                            sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                                MojoInt4( tileIndex.x, tileIndex.y, tileIndex.z + 1, tileIndex.w ),
                                MojoInt4( thisVoxel.x, thisVoxel.y, thisVoxel.z + 1, currentW ) ) );
                        }
                    }

                    //
					// Add a scaled-down w to the queue
					//
                    //if (currentW < numTiles.w-1) wQueue.push( MojoInt4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
                }

            }
            //std::swap(tileQueue, wQueue);
            //++currentW;
        //}

        if ( tileLoaded )
        {
            //
			// Save and unload the previous tile
			//
            if ( tileChanged )
            {
                SaveTile( previousTileIndex, volumeDescriptions );
				StrideUpIdTileChange( numTiles, numVoxelsPerTile, previousTileIndex, currentIdVolume );

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
					mNextUndoItem->idTileMapRemoveOldIdSets[ oldId ].insert ( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
        // Update the segment sizes
        //
        mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) + voxelChangeCount );
        mSegmentInfoManager.SetVoxelCount( oldId, mSegmentInfoManager.GetVoxelCount( oldId ) - voxelChangeCount );

        //
		// Update idTileMap
		//
        mSegmentInfoManager.SetTiles( oldId, tilesContainingOldId );
        mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

        Core::Printf( "\nFinished replacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " in 3D from zslice ", pVoxelSpace.z, ".\n" );

		mLogger.Log( Core::ToString( "ReplaceSegmentationLabelCurrentConnectedComponent: oldId=", oldId, ", newId=", newId, ", x=", pVoxelSpace.x, ", y=", pVoxelSpace.y, ", z=", pVoxelSpace.z ) );

		mCurrentOperationProgress = 1;

    }
}

//
// 2D Split Methods
//


void FileSystemTileServer::DrawSplit( MojoFloat3 pointTileSpace, float radius )
{
	DrawRegionValue( pointTileSpace, radius, REGION_SPLIT );
}

void FileSystemTileServer::DrawErase( MojoFloat3 pointTileSpace, float radius )
{
	DrawRegionValue( pointTileSpace, radius, 0 );
}

void FileSystemTileServer::DrawRegionA( MojoFloat3 pointTileSpace, float radius )
{
	DrawRegionValue( pointTileSpace, radius, REGION_A );
}

void FileSystemTileServer::DrawRegionB( MojoFloat3 pointTileSpace, float radius )
{
	DrawRegionValue( pointTileSpace, radius, REGION_B );
}

void FileSystemTileServer::DrawRegionValue( MojoFloat3 pointTileSpace, float radius, int value )
{
	//
	// Check for out-of-bounds
	//
	if ( pointTileSpace.x < mSplitWindowStart.x || pointTileSpace.x > (mSplitWindowStart.x + mSplitWindowNTiles.x) ||
		pointTileSpace.y < mSplitWindowStart.y || pointTileSpace.y > (mSplitWindowStart.y + mSplitWindowNTiles.y) ||
		pointTileSpace.z != mSplitWindowStart.z )
	{
		return;
	}

    MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();

    MojoInt3 pVoxelSpace = 
        MojoInt3(
        (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
        (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
        (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    int areaX = pVoxelSpace.x - mSplitWindowStart.x * numVoxelsPerTile.x;
    int areaY = pVoxelSpace.y - mSplitWindowStart.y * numVoxelsPerTile.y;
    int areaIndex = areaX + areaY * mSplitWindowWidth;

    SimpleSplitTools::ApplyCircleMask( areaIndex, mSplitWindowWidth, mSplitWindowHeight, value, radius, mSplitDrawArea );

    int irad = (int) ( radius + 0.5 );
    MojoInt2 upperLeft = MojoInt2( areaX - irad, areaY - irad );
    MojoInt2 lowerRight = MojoInt2( areaX + irad, areaY + irad );
    UpdateOverlayTilesBoundingBox( upperLeft, lowerRight );

    //Core::Printf( "\nDrew inside bounding box (", upperLeft.x, ",", upperLeft.y, "x", lowerRight.x, ",", lowerRight.y, ").\n" );
    //Core::Printf( "\nDrew split circle voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, "), with radius ", radius, ".\n" );
}

void FileSystemTileServer::AddSplitSource( MojoFloat3 pointTileSpace )
{
    MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();

    MojoInt3 pVoxelSpace = 
        MojoInt3(
        (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
        (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
        (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    //
    // Check for duplicates
    //
    bool duplicate = false;

    for ( unsigned int si = 0; si < mSplitSourcePoints.size(); ++si )
    {
        MojoInt3 existingPoint = mSplitSourcePoints[ si ];
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
    MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();

    MojoInt2 upperLeft = MojoInt2( 0, 0 );
    MojoInt2 lowerRight = MojoInt2( mSplitWindowNTiles.x * numVoxelsPerTile.x, mSplitWindowNTiles.y * numVoxelsPerTile.y );

    UpdateOverlayTilesBoundingBox( upperLeft, lowerRight );
}

void FileSystemTileServer::UpdateOverlayTilesBoundingBox( MojoInt2 upperLeft, MojoInt2 lowerRight )
{
	//
	// Export result to OverlayMap tiles
	//
    TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

    MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
	MojoInt4 numTiles = tiledVolumeDescription.numTiles();

	Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

    //std::map< MojoInt4, int, Core::Int4Comparator > wQueue;

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

			bool anyChanges = false;

            if ( minX < maxX && minY < maxY )
            {
				++tileCount;
				//Core::Printf( "Copying result tile ", tileCount, "." );

				MojoInt4 tileIndex = MojoInt4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
				volumeDescriptions = LoadTile( tileIndex );
				splitData = (unsigned int*) volumeDescriptions.Get( "OverlayMap" ).data;

				//
				// Copy result values into the tile
				//

                for ( int tileY = minY; tileY <= maxY; ++tileY )
                {
                    for ( int tileX = minX; tileX <= maxX; ++tileX )
                    {
                        int tileIndex1D = tileX + tileY * numVoxelsPerTile.x;
                        int areaIndex1D = xOffset + tileX + ( yOffset + tileY ) * mSplitWindowWidth;

					    //RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

                        if ( mSplitResultArea != 0 )
                        {
                            if ( mSplitResultArea[ areaIndex1D ] && splitData[ tileIndex1D ] != mSplitResultArea[ areaIndex1D ] )
                            {
                                splitData[ tileIndex1D ] = mSplitResultArea[ areaIndex1D ];
								anyChanges = true;
                                //wQueue[ MojoInt4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitResultArea[ areaIndex1D ];
                            }
                            else if ( mSplitResultArea[ areaIndex1D ] == 0 && mSplitDrawArea[ areaIndex1D ] && splitData[ tileIndex1D ] != mSplitDrawArea[ areaIndex1D ] )
                            {
                                splitData[ tileIndex1D ] = mSplitDrawArea[ areaIndex1D ];
						        anyChanges = true;
                                //wQueue[ MojoInt4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitDrawArea[ areaIndex1D ];
                            }
						    else if ( splitData[ tileIndex1D ] && mSplitResultArea[ areaIndex1D ] == 0 && mSplitDrawArea[ areaIndex1D ] == 0)
						    {
                                splitData[ tileIndex1D ] = 0;
						        anyChanges = true;
                                //wQueue[ MojoInt4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = 0;
						    }
                        }
                        else if ( mSplitDrawArea != 0 )
                        {
                            if ( mSplitDrawArea[ areaIndex1D ] && splitData[ tileIndex1D ] != mSplitDrawArea[ areaIndex1D ] )
                            {
                                splitData[ tileIndex1D ] = mSplitDrawArea[ areaIndex1D ];
						        anyChanges = true;
                                //wQueue[ MojoInt4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitDrawArea[ areaIndex1D ];
                            }
						    else if ( splitData[ tileIndex1D ] && mSplitDrawArea[ areaIndex1D ] == 0)
						    {
                                splitData[ tileIndex1D ] = 0;
						        anyChanges = true;
                                //wQueue[ MojoInt4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = 0;
						    }
                        }

                    }
                }


				if ( anyChanges )
				{
					//
					// Stride up in w
					//
					for ( int wIndex = 1; wIndex < numTiles.w; ++wIndex )
					{
						int stride = (int) pow( 2.0, wIndex );

						int strideMinX = ( minX / stride ) * stride;
						int strideMinY = ( minY / stride ) * stride;

						MojoInt4 strideTileIndex = MojoInt4( tileIndex.x / stride, tileIndex.y / stride, tileIndex.z, wIndex );

						int strideOffsetX = ( tileIndex.x % stride ) * ( numVoxelsPerTile.x / stride );
						int strideOffsetY = ( tileIndex.y % stride ) * ( numVoxelsPerTile.y / stride );

						Core::HashMap< std::string, Core::VolumeDescription > strideVolumeDescriptions = LoadTile( strideTileIndex );
						unsigned int* strideData = (unsigned int*) strideVolumeDescriptions.Get( "OverlayMap" ).data;

						int wTileY = strideOffsetY + strideMinY / stride;
						for ( int tileY = strideMinY; tileY <= maxY; tileY += stride )
						{
							int wTileX = strideOffsetX + strideMinX / stride;
							for ( int tileX = strideMinX; tileX <= maxX; tileX += stride )
							{
								int fromIndex1D = tileX + tileY * numVoxelsPerTile.x;
								int toIndex1D = wTileX + wTileY * numVoxelsPerTile.x;
								strideData[ toIndex1D ] = splitData[ fromIndex1D ];
								++wTileX;
							}
							++wTileY;
						}

						UnloadTile( strideTileIndex );

					}
				}

				UnloadTile( tileIndex );

            }
		}
	}

}

void FileSystemTileServer::StrideUpIdTileChange( MojoInt4 numTiles, MojoInt3 numVoxelsPerTile, MojoInt4 tileIndex, unsigned int* data )
{

	//
	// Stride up in w
	//

	int minX = 0;
	int minY = 0;
	int maxX = numVoxelsPerTile.x - 1;
	int maxY = numVoxelsPerTile.y - 1;

	for ( int wIndex = 1; wIndex < numTiles.w; ++wIndex )
	{
		int stride = (int) pow( 2.0, wIndex );

		int strideMinX = ( minX / stride ) * stride;
		int strideMinY = ( minY / stride ) * stride;

		MojoInt4 strideTileIndex = MojoInt4( tileIndex.x / stride, tileIndex.y / stride, tileIndex.z, wIndex );

		int strideOffsetX = ( tileIndex.x % stride ) * ( numVoxelsPerTile.x / stride );
		int strideOffsetY = ( tileIndex.y % stride ) * ( numVoxelsPerTile.y / stride );

		Core::HashMap< std::string, Core::VolumeDescription > strideVolumeDescriptions = LoadTile( strideTileIndex );
		unsigned int* strideData = (unsigned int*) strideVolumeDescriptions.Get( "IdMap" ).data;

		int wTileY = strideOffsetY + strideMinY / stride;
		for ( int tileY = strideMinY; tileY <= maxY; tileY += stride )
		{
			int wTileX = strideOffsetX + strideMinX / stride;
			for ( int tileX = strideMinX; tileX <= maxX; tileX += stride )
			{
				int fromIndex1D = tileX + tileY * numVoxelsPerTile.x;
				int toIndex1D = wTileX + wTileY * numVoxelsPerTile.x;

				//RELEASE_ASSERT( toIndex1D < TILE_PIXELS * TILE_PIXELS );
				//RELEASE_ASSERT( fromIndex1D < TILE_PIXELS * TILE_PIXELS );

				strideData[ toIndex1D ] = data[ fromIndex1D ];
				++wTileX;
			}
			++wTileY;
		}

		SaveTile( strideTileIndex, strideVolumeDescriptions );
		UnloadTile( strideTileIndex );

	}
}

void FileSystemTileServer::ResetOverlayTiles()
{
	//
	// Reset the overlay layer
	//
    TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );
    MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
	MojoInt4 numTiles = tiledVolumeDescription.numTiles();

	std::set< MojoInt4, Int4Comparator > clearedTiles;

	Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

	for ( int xd = 0; xd < mSplitWindowNTiles.x; ++xd )
	{
		for (int yd = 0; yd < mSplitWindowNTiles.y; ++yd )
		{
			int tileX = mSplitWindowStart.x + xd;
			int tileY = mSplitWindowStart.y + yd;

			//
			// Load and clear all tiles in w stack
			//
			for( int wIndex = 0; wIndex < numTiles.w; ++wIndex )
			{
				MojoInt4 thisTile = MojoInt4(
					tileX / (int) pow( 2.0, wIndex ),
					tileY / (int) pow( 2.0, wIndex ),
					mSplitWindowStart.z, wIndex );
				if ( clearedTiles.find( thisTile ) == clearedTiles.end() )
				{
					volumeDescriptions = LoadTile( thisTile );
					memset( volumeDescriptions.Get( "OverlayMap" ).data, 0, TILE_PIXELS * TILE_PIXELS * sizeof(unsigned int) );
					UnloadTile( thisTile );
				}
			}
		}
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

void FileSystemTileServer::ResetDrawMergeState()
{

    for ( int i = 0; i < mSplitWindowNPix; ++i )
    {
        mSplitDrawArea[ i ] = 0;
    }

	ResetOverlayTiles();

    Core::Printf( "Reset Draw Merge State.\n");

}

void FileSystemTileServer::LoadSplitDistances( unsigned int segId )
{
    //
    // Load distances
    //
    Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

    MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();
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

            MojoInt4 tileIndex = MojoInt4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
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

                //RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

                //
                // Distance calculation
                //
                int segVal = ( (int) currentSrcVolume[ tileIndex1D ] ) + 10;
                mSplitStepDist[ areaIndex1D ] = segVal * segVal;
                ++areaCount;

                //
                // Mark border targets
                //
				if ( mSegmentInfoManager.GetIdForLabel( currentIdVolume[ tileIndex1D ] ) == segId )
                {
                    ++mSplitLabelCount;
                    mCentroid.x += (float) areaX + mSplitWindowStart.x * numVoxelsPerTile.x;
                    mCentroid.y += (float) areaY + mSplitWindowStart.y * numVoxelsPerTile.y;
                    mSplitBorderTargets[ areaIndex1D ] = 0;
                }
                else if ( mSegmentInfoManager.GetIdForLabel( currentIdVolume[ tileIndex1D ] ) != 0 ||
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

void FileSystemTileServer::PrepForSplit( unsigned int segId, MojoFloat3 pointTileSpace )
{
    //
    // Find the size of this segment and load the bounding box of tiles
    //

    if ( mIsSegmentationLoaded )
    {
        //
        // Reset split state if we are more than 2 tiles away
        //
        if ( mPrevSplitId != 0 && abs( mPrevSplitZ - (int)pointTileSpace.z ) > 2 )
        {
            mPrevSplitId = 0;
            mPrevSplitZ = -2;
            mPrevSplitLine.clear();
            mPrevSplitCentroids.clear();
        }

        bool success = false;
        int attempts = 0;

        while ( !success && attempts < 2 )
        {
            ++attempts;

            try
            {

                Core::Printf( "\nPreparing for split of segment ", segId, " at x=", pointTileSpace.x, ", y=", pointTileSpace.y, ", z=", pointTileSpace.z, ".\n" );

                TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

                MojoInt3 numVoxels = tiledVolumeDescription.numVoxels();
                MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
                MojoInt4 numTiles = tiledVolumeDescription.numTiles();

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
                if ( minTileX < ( (int) pointTileSpace.x ) - SPLIT_ADJUST_BUFFER_TILE_HALO )
                    minTileX = ( (int) pointTileSpace.x ) - SPLIT_ADJUST_BUFFER_TILE_HALO;
                if ( maxTileX > ( (int) pointTileSpace.x ) + SPLIT_ADJUST_BUFFER_TILE_HALO )
                    maxTileX = ( (int) pointTileSpace.x ) + SPLIT_ADJUST_BUFFER_TILE_HALO;

                if ( minTileY < ( (int) pointTileSpace.y ) - SPLIT_ADJUST_BUFFER_TILE_HALO )
                    minTileY = ( (int) pointTileSpace.y ) - SPLIT_ADJUST_BUFFER_TILE_HALO;
                if ( maxTileY > ( (int) pointTileSpace.y ) + SPLIT_ADJUST_BUFFER_TILE_HALO )
                    maxTileY = ( (int) pointTileSpace.y ) + SPLIT_ADJUST_BUFFER_TILE_HALO;

                //
                // Calculate sizes
                //
                mSplitWindowStart = MojoInt3( minTileX, minTileY, (int) pointTileSpace.z );
                mSplitWindowNTiles = MojoInt3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

                //Core::Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
                //Core::Printf( "mSplitWindowSize=", mSplitWindowNTiles.x, ":", mSplitWindowNTiles.y, ":", mSplitWindowNTiles.z, ".\n" );

		        mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowNTiles.x;
		        mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowNTiles.y;
		        mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

                Core::Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

                int maxBufferSize = ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * TILE_PIXELS * TILE_PIXELS;
                RELEASE_ASSERT( mSplitWindowNPix <= maxBufferSize );

		        mSplitLabelCount = 0;

                //
                // Allocate working space (if necessary)
                //

                if ( mSplitStepDist == 0 )
		            mSplitStepDist = new int[ maxBufferSize ];
                    
                if ( mSplitResultDist == 0 )
                    mSplitResultDist = new int[ maxBufferSize ];

                if ( mSplitPrev == 0 )
		            mSplitPrev = new int[ maxBufferSize ];

                if ( mSplitSearchMask == 0 )
		            mSplitSearchMask = new char[ maxBufferSize ];

                if ( mSplitDrawArea == 0 )
		            mSplitDrawArea = new char[ maxBufferSize ];

		        if ( mSplitBorderTargets == 0 )
		            mSplitBorderTargets = new char[ maxBufferSize ];

                if ( mSplitResultArea == 0 )
		            mSplitResultArea = new unsigned int[ maxBufferSize ];


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

void FileSystemTileServer::PrepForAdjust( unsigned int segId, MojoFloat3 pointTileSpace )
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

                TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

                MojoInt3 numVoxels = tiledVolumeDescription.numVoxels();
                MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
                MojoInt4 numTiles = tiledVolumeDescription.numTiles();

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
                if ( minTileX < ( (int) pointTileSpace.x ) - SPLIT_ADJUST_BUFFER_TILE_HALO )
                    minTileX = ( (int) pointTileSpace.x ) - SPLIT_ADJUST_BUFFER_TILE_HALO;
                if ( maxTileX > ( (int) pointTileSpace.x ) + SPLIT_ADJUST_BUFFER_TILE_HALO )
                    maxTileX = ( (int) pointTileSpace.x ) + SPLIT_ADJUST_BUFFER_TILE_HALO;

                if ( minTileY < ( (int) pointTileSpace.y ) - SPLIT_ADJUST_BUFFER_TILE_HALO )
                    minTileY = ( (int) pointTileSpace.y ) - SPLIT_ADJUST_BUFFER_TILE_HALO;
                if ( maxTileY > ( (int) pointTileSpace.y ) + SPLIT_ADJUST_BUFFER_TILE_HALO )
                    maxTileY = ( (int) pointTileSpace.y ) + SPLIT_ADJUST_BUFFER_TILE_HALO;

                //
                // Calculate sizes
                //
                mSplitWindowStart = MojoInt3( minTileX, minTileY, (int) pointTileSpace.z );
                mSplitWindowNTiles = MojoInt3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

                //Core::Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
                //Core::Printf( "mSplitWindowSize=", mSplitWindowNTiles.x, ":", mSplitWindowNTiles.y, ":", mSplitWindowNTiles.z, ".\n" );

		        mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowNTiles.x;
		        mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowNTiles.y;
		        mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

                int maxBufferSize = ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * TILE_PIXELS * TILE_PIXELS;
                RELEASE_ASSERT( mSplitWindowNPix <= maxBufferSize );

                //Core::Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

		        mSplitLabelCount = 0;

                //
                // Allocate working space (if necessary)
                //

                if ( mSplitDrawArea == 0 )
		            mSplitDrawArea = new char[ maxBufferSize ];

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

void FileSystemTileServer::PrepForDrawMerge( MojoFloat3 pointTileSpace )
{
    //
    // Reset the bounding box of overlay tiles around this point
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

                Core::Printf( "\nPreparing for draw merge at x=", pointTileSpace.x, ", y=", pointTileSpace.y, ", z=", pointTileSpace.z, ".\n" );

                TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

                MojoInt3 numVoxels = tiledVolumeDescription.numVoxels();
                MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
                MojoInt4 numTiles = tiledVolumeDescription.numTiles();

                //
                // Restrict search tiles to max 2 tiles away from clicked location
                //
                int minTileX = ( (int) pointTileSpace.x ) - SPLIT_ADJUST_BUFFER_TILE_HALO;
                int maxTileX = ( (int) pointTileSpace.x ) + SPLIT_ADJUST_BUFFER_TILE_HALO;
                int minTileY = ( (int) pointTileSpace.y ) - SPLIT_ADJUST_BUFFER_TILE_HALO;
                int maxTileY = ( (int) pointTileSpace.y ) + SPLIT_ADJUST_BUFFER_TILE_HALO;

                //
                // Don't go over the edges
                //
                if ( minTileX < 0 )
                    minTileX = 0;
                if ( maxTileX > numTiles.x - 1 )
                    maxTileX = numTiles.x - 1;

                if ( minTileY < 0 )
                    minTileY = 0;
                if ( maxTileY > numTiles.y - 1 )
                    maxTileY = numTiles.y - 1;

                //
                // Calculate sizes
                //
                mSplitWindowStart = MojoInt3( minTileX, minTileY, (int) pointTileSpace.z );
                mSplitWindowNTiles = MojoInt3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

                //Core::Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
                //Core::Printf( "mSplitWindowSize=", mSplitWindowNTiles.x, ":", mSplitWindowNTiles.y, ":", mSplitWindowNTiles.z, ".\n" );

		        mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowNTiles.x;
		        mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowNTiles.y;
		        mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

                int maxBufferSize = ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * TILE_PIXELS * TILE_PIXELS;
                RELEASE_ASSERT( mSplitWindowNPix <= maxBufferSize );

                //Core::Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

		        mSplitLabelCount = 0;

                //
                // Allocate working space (if necessary)
                //

                if ( mSplitDrawArea == 0 )
				{
		            mSplitDrawArea = new char[ maxBufferSize ];
					ResetDrawMergeState();
				}

                Core::Printf( "\nFinished preparing for draw merge at z=", pointTileSpace.z, ".\n" );

                success = true;
            }
            catch ( std::bad_alloc e )
            {
                Core::Printf( "WARNING: std::bad_alloc Error while preparing for draw merge - reducing tile cache size." );
                ReduceCacheSize();
            }
            catch ( ... )
            {
                Core::Printf( "WARNING: Unexpected error while preparing for draw merge - attempting to continue." );
                ReduceCacheSize();
            }
        }

        if ( !success )
        {
            Core::Printf( "ERROR: Unable to prep for draw merge - possibly out of memory." );
        }
    }

}

void FileSystemTileServer::RecordSplitState( unsigned int segId, MojoFloat3 pointTileSpace )
{
    if ( mIsSegmentationLoaded )
    {
	    MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();
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
                    currentState.splitLine.push_back( MojoFloat2(
                        (float) ( mSplitWindowStart.x * numVoxelsPerTile.x + areaX ) / (float) numVoxelsPerTile.x,
                        (float) ( mSplitWindowStart.y * numVoxelsPerTile.y + areaY ) / (float) numVoxelsPerTile.y ) );
                    anyLinePixels = true;
                }
                if ( mSplitDrawArea[ areaIndex1D ] != 0 )
                {
                    currentState.splitDrawPoints.push_back( std::pair< MojoFloat2, char >( MojoFloat2(
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

void FileSystemTileServer::PredictSplit( unsigned int segId, MojoFloat3 pointTileSpace, float radius )
{
    if ( mIsSegmentationLoaded )
    {
	    MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();
        int currentZ = (int) floor( pointTileSpace.z * numVoxelsPerTile.z );

        std::vector< MojoFloat2 > *restoreLine = NULL;

        //
        // Try to restore from stored state
        //
        if ( mSplitStates.find( currentZ ) != mSplitStates.end() )
        {
            std::vector< std::pair< MojoFloat2, char >> *restorePoints = &mSplitStates.find( currentZ )->second.splitDrawPoints;
            for ( std::vector< std::pair< MojoFloat2, char >>::iterator drawIt = restorePoints->begin(); drawIt != restorePoints->end(); ++drawIt )
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

		//
		// Restrict prediction to 1 z-step
		//
        //else if ( mSplitStates.find( currentZ - 1 ) != mSplitStates.end() )
        //{
        //    restoreLine = &mSplitStates.find( currentZ - 1 )->second.splitLine;
        //    Core::Printf( "Predicting split at z=", currentZ, ", from neighbour -1." );
        //}
        //else if ( mSplitStates.find( currentZ + 1 ) != mSplitStates.end() )
        //{
        //    restoreLine = &mSplitStates.find( currentZ + 1 )->second.splitLine;
        //    Core::Printf( "Predicting split at z=", currentZ, ", from neighbour +1." );
        //}
        //else
        //{
        //    Core::Printf( "No neighbours for split prediction at z=", currentZ, " (prev=", mPrevSplitZ, ")." );
        //}

        if ( restoreLine != NULL )
        {
            //
            // Draw a line where the previous split was
            //
            for ( std::vector< MojoFloat2 >::iterator splitIt = restoreLine->begin(); splitIt != restoreLine->end(); ++splitIt )
            {
                DrawSplit( MojoFloat3( splitIt->x, splitIt->y, (float) currentZ ), radius );
            }

            //
            // Find another split line here
            //
            FindBoundaryWithinRegion2D( segId );

        }
    }
}

unsigned int FileSystemTileServer::CompletePointSplit( unsigned int segId, MojoFloat3 pointTileSpace )
{
	unsigned int newId = 0;

	if ( mIsSegmentationLoaded && mSplitSourcePoints.size() > 0 && mSegmentInfoManager.GetConfidence( segId ) < 100 )
    {
        long voxelChangeCount = 0;

		MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();

        MojoInt3 pMouseOverVoxelSpace = 
            MojoInt3(
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

		MojoInt3 pVoxelSpace = 
			MojoInt3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
			seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
			mSplitWindowStart.z );

        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

		MojoInt3 numVoxels = tiledVolumeDescription.numVoxels();
		MojoInt4 numTiles = tiledVolumeDescription.numTiles();

        Core::Printf( "\nSplitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ")...\n" );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

		PrepForNextUndoRedoChange();

		TileChangeIdMap *tileChange;

		MojoInt4 previousTileIndex;
        bool tileLoaded = false;

        std::queue< MojoInt4 > tileQueue;
        std::multimap< MojoInt4, MojoInt4, Mojo::Core::Int4Comparator > sliceQueue;
		std::queue< MojoInt4 > wQueue;

        unsigned int* currentIdVolume;
        MojoInt3 currentIdNumVoxels;
        MojoInt4 thisVoxel;

		int nPixChanged = 0;
		bool invert = false;
        bool foundMouseOverPixel = false;

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( segId );

		int maxProgress = 0;
		int currentProgress = 0;

		//
		// Determine the (approximate) max amount of work to be done
		//
		for( FileSystemTileSet::iterator tileIndexi = tilesContainingOldId.begin(); tileIndexi != tilesContainingOldId.end(); ++tileIndexi )
        {
			if ( tileIndexi->z == pVoxelSpace.z )
			{
				++maxProgress;
			}
		}

		//
		// Compensate for possible inversion and search / fill time
		//
		maxProgress *= 2;
		
		tileQueue.push( MojoInt4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

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

				if ( currentProgress < maxProgress )
				{
					++currentProgress;
					mCurrentOperationProgress = (float)currentProgress / (float)maxProgress;
				}
            }

            if ( thisVoxel.x == pMouseOverVoxelSpace.x && thisVoxel.y == pMouseOverVoxelSpace.y && thisVoxel.z == pMouseOverVoxelSpace.z )
            {
                foundMouseOverPixel = true;
			    //Core::Printf( "Found mouseover pixel - inverting." );
            }

            //
			// Find the tile for this pixel
			//
            MojoInt4 tileIndex = MojoInt4( thisVoxel.x / numVoxelsPerTile.x,
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
                currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;
                currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
                tileLoaded = true;

				//
				// Get or create the change record for this tile
				//
				tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];
            }

            int tileX = thisVoxel.x % numVoxelsPerTile.x;
            int tileY = thisVoxel.y % numVoxelsPerTile.y;

            MojoInt3 index3D = MojoInt3( tileX, tileY, 0 );
            int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
			int  idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ index1D ] );

			bool isSplitBorder = false;

			areaIndex1D = thisVoxel.x - mSplitWindowStart.x * numVoxelsPerTile.x +
				(thisVoxel.y - mSplitWindowStart.y * numVoxelsPerTile.y) * mSplitWindowWidth;
			if ( areaIndex1D >= 0 && areaIndex1D < mSplitWindowNPix )
			{
				isSplitBorder = mSplitResultArea[ areaIndex1D ] != 0;
			}

			if ( idValue == segId && !isSplitBorder && !(*tileChange)[ currentIdVolume[ index1D ] ][ index1D ] )
            {
				(*tileChange)[ currentIdVolume[ index1D ] ].set( index1D );
				wQueue.push( thisVoxel );
				++nPixChanged;

                //
				// Add neighbours to the appropriate queue
				//
                if (thisVoxel.x > 0)
                {
                    if (tileX > 0)
                    {
                        tileQueue.push( MojoInt4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                            MojoInt4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                            MojoInt4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0 ) ) );
                    }
                }

                if (thisVoxel.x < numVoxels.x - 1)
                {
                    if (tileX < numVoxelsPerTile.x - 1)
                    {
                        tileQueue.push( MojoInt4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                            MojoInt4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                            MojoInt4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0 ) ) );
                    }
                }
                if (thisVoxel.y > 0)
                {
                    if (tileY > 0)
                    {
                        tileQueue.push( MojoInt4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                            MojoInt4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                            MojoInt4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0 ) ) );
                    }
                }
                if (thisVoxel.y < numVoxels.y - 1)
                {
                    if (tileY < numVoxelsPerTile.y - 1)
                    {
                        tileQueue.push( MojoInt4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
                            MojoInt4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                            MojoInt4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0 ) ) );
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

				wQueue = std::queue< MojoInt4 >();

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
						tileIndex = MojoInt4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
						volumeDescriptions = LoadTile( tileIndex );
						currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;

						int xOffset = xd * numVoxelsPerTile.x;
						int yOffset = yd * numVoxelsPerTile.y;
						int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

						for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
						{
							int areaX = xOffset + tileIndex1D % numVoxelsPerTile.x;
							int areaY = yOffset + tileIndex1D / numVoxelsPerTile.x;
							seedIndex1D = areaX + areaY * mSplitWindowWidth;

							if ( mSegmentInfoManager.GetIdForLabel( currentIdVolume[ tileIndex1D ] ) == segId &&
								( mNextUndoItem->tileChangeMap.find( tileIndex ) == mNextUndoItem->tileChangeMap.end() || 
								!mNextUndoItem->tileChangeMap[ tileIndex ][ currentIdVolume[ index1D ] ][ index1D ] ) &&
								mSplitResultArea[ seedIndex1D ] == 0 )
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
				mNextUndoItem->tileChangeMap.clear();

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
					MojoInt3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
					seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
					mSplitWindowStart.z );

				nPixChanged = 0;
				invert = true;

				tileQueue.push( MojoInt4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

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

        FileSystemTileSet tilesContainingNewId;

		mNextUndoItem->newId = newId;

		bool tileChanged;

        //Core::Printf( "Splitting (invert=", invert, ")." );

		//
		// Do the split and fill up in w
		//

        std::swap(tileQueue, wQueue);
		int currentW = 0;

		tileLoaded = false;
		tileChanged = false;

		//while ( currentW < numTiles.w )
		//{
			Core::Printf( "Splitting at w=", currentW, "." );
			while ( tileQueue.size() > 0 )
			{
				thisVoxel = tileQueue.front();
				tileQueue.pop();

				if ( currentProgress < maxProgress )
				{
					++currentProgress;
					mCurrentOperationProgress = (float)currentProgress / (float)maxProgress;
				}

				//
				// Find the tile for this pixel
				//
				MojoInt4 tileIndex = MojoInt4( thisVoxel.x / numVoxelsPerTile.x,
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
							StrideUpIdTileChange( numTiles, numVoxelsPerTile, previousTileIndex, currentIdVolume );

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
								mNextUndoItem->idTileMapRemoveOldIdSets[ segId ].insert( previousTileIndex );
							}

						}
						UnloadTile( previousTileIndex );
					}

					//
					// Load the current tile
					//
					volumeDescriptions = LoadTile( tileIndex );
					previousTileIndex = tileIndex;
					currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;
					currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
					tileLoaded = true;
					tileChanged = false;

					//
					// Get or create the change record for this tile
					//
					tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];
				}
				int tileX = thisVoxel.x % numVoxelsPerTile.x;
				int tileY = thisVoxel.y % numVoxelsPerTile.y;

				MojoInt3 index3D = MojoInt3( tileX, tileY, 0 );
				int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
				int  idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ index1D ] );

				if ( idValue == segId )
				{
					(*tileChange)[ currentIdVolume[ index1D ] ].set( index1D );
					currentIdVolume[ index1D ] = newId;
					tileChanged = true;
                    if ( currentW == 0 )
                        ++voxelChangeCount;

					//
					// Add a scaled-down w to the queue
					//
					//if (currentW < numTiles.w-1) wQueue.push( MojoInt4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
				}

			}
			//std::swap(tileQueue, wQueue);
			//++currentW;
		//}

        if ( tileLoaded )
        {
            //
			// Save and unload the previous tile
			//
            if ( tileChanged )
            {
                SaveTile( previousTileIndex, volumeDescriptions );
				StrideUpIdTileChange( numTiles, numVoxelsPerTile, previousTileIndex, currentIdVolume );

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
					mNextUndoItem->idTileMapRemoveOldIdSets[ segId ].insert( previousTileIndex );
                }

            }
            UnloadTile( previousTileIndex );
        }

        //
        // Update the segment sizes
        //
        mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) + voxelChangeCount );
        mSegmentInfoManager.SetVoxelCount( segId, mSegmentInfoManager.GetVoxelCount( segId ) - voxelChangeCount );

        //
		// Update idTileMap
		//
        mSegmentInfoManager.SetTiles( segId, tilesContainingOldId );
        mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

        Core::Printf( "\nFinished Splitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") to new segmentation label ", newId, "...\n" );

		mLogger.Log( Core::ToString( "CompletePointSplit: segId=", segId, ", newId=", newId, ", z=", pVoxelSpace.z, ", npixels=", voxelChangeCount ) );

		mCurrentOperationProgress = 1;

    }

	//
	// Prep for more splitting
	//
	LoadSplitDistances( segId );
    ResetSplitState();

	return newId;

}

unsigned int FileSystemTileServer::CompleteDrawSplit( unsigned int segId, MojoFloat3 pointTileSpace, bool join3D, int splitStartZ )
{
	unsigned int newId = 0;

	if ( mIsSegmentationLoaded && mSplitNPerimiters > 0 && mSegmentInfoManager.GetConfidence( segId ) < 100 )
    {
        long voxelChangeCount = 0;

        RecordSplitState( segId, pointTileSpace );

		MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();
        MojoInt3 pMouseOverVoxelSpace = 
            MojoInt3(
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
        std::vector< std::pair< MojoFloat2, int >> newCentroids;

        while ( continueZ )
        {

            float centerX = 0;
            float centerY = 0;
            float unchangedCenterX = 0;
            float unchangedCenterY = 0;

            int fillSuccesses = 0;

		    for ( int perimiter = 1; perimiter <= mSplitNPerimiters; ++ perimiter )
		    {
				voxelChangeCount = 0;

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

			    MojoInt3 pVoxelSpace = 
				    MojoInt3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
				    seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
				    mSplitWindowStart.z );

                TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

			    MojoInt3 numVoxels = tiledVolumeDescription.numVoxels();
			    MojoInt4 numTiles = tiledVolumeDescription.numTiles();

			    Core::Printf( "\nSplitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ")...\n" );

			    Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

			    PrepForNextUndoRedoChange();

			    TileChangeIdMap *tileChange;

			    MojoInt4 previousTileIndex;
			    bool tileLoaded = false;

			    std::queue< MojoInt4 > tileQueue;
			    std::multimap< MojoInt4, MojoInt4, Mojo::Core::Int4Comparator > sliceQueue;
			    std::queue< MojoInt4 > wQueue;

			    unsigned int* currentIdVolume;
			    MojoInt3 currentIdNumVoxels;
			    MojoInt4 thisVoxel;

			    int nPixChanged = 0;
			    bool invert = false;
                bool foundMouseOverPixel = false;

                FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( segId );

				int maxProgress = 0;
				int currentProgress = 0;

				//
				// Determine the (approximate) max amount of work to be done
				//
				for( FileSystemTileSet::iterator tileIndexi = tilesContainingOldId.begin(); tileIndexi != tilesContainingOldId.end(); ++tileIndexi )
				{
					if ( tileIndexi->z == pVoxelSpace.z )
					{
						++maxProgress;
					}
				}

				//
				// Compensate for possible inversion and search / fill time
				//
				maxProgress *= 2;

				tileQueue.push( MojoInt4( pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0 ) );

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

						if ( currentProgress < maxProgress )
						{
							++currentProgress;
							mCurrentOperationProgress = (float)currentProgress / (float)maxProgress;
						}
				    }

                    if ( thisVoxel.x == pMouseOverVoxelSpace.x && thisVoxel.y == pMouseOverVoxelSpace.y && thisVoxel.z == pMouseOverVoxelSpace.z )
                    {
                        foundMouseOverPixel = true;
			            //Core::Printf( "Found mouseover pixel - inverting." );
                    }

				    //
				    // Find the tile for this pixel
				    //
				    MojoInt4 tileIndex = MojoInt4( thisVoxel.x / numVoxelsPerTile.x,
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
					    currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;
					    currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
					    tileLoaded = true;

						//
						// Get or create the change record for this tile
						//
						tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];
				    }

				    int tileX = thisVoxel.x % numVoxelsPerTile.x;
				    int tileY = thisVoxel.y % numVoxelsPerTile.y;

				    MojoInt3 index3D = MojoInt3( tileX, tileY, 0 );
				    int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
					int  idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ index1D ] );

				    bool isSplitBorder = false;

				    int areaIndex1D = thisVoxel.x - mSplitWindowStart.x * numVoxelsPerTile.x +
					    (thisVoxel.y - mSplitWindowStart.y * numVoxelsPerTile.y) * mSplitWindowWidth;
				    if ( areaIndex1D >= 0 && areaIndex1D < mSplitWindowNPix )
				    {
					    isSplitBorder = mSplitResultArea[ areaIndex1D ] != 0;
				    }
					
					if ( idValue == segId && !(*tileChange)[ currentIdVolume[ index1D ] ][ index1D ] )
				    {
						(*tileChange)[ currentIdVolume[ index1D ] ].set( index1D );
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
								    tileQueue.push( MojoInt4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0) );
							    }
							    else
							    {
								    sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
									    MojoInt4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
									    MojoInt4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0 ) ) );
							    }
						    }

						    if (thisVoxel.x < numVoxels.x - 1)
						    {
							    if (tileX < numVoxelsPerTile.x - 1)
							    {
								    tileQueue.push( MojoInt4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0) );
							    }
							    else
							    {
								    sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
									    MojoInt4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
									    MojoInt4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0 ) ) );
							    }
						    }
						    if (thisVoxel.y > 0)
						    {
							    if (tileY > 0)
							    {
								    tileQueue.push( MojoInt4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0) );
							    }
							    else
							    {
								    sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
									    MojoInt4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
									    MojoInt4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0 ) ) );
							    }
						    }
						    if (thisVoxel.y < numVoxels.y - 1)
						    {
							    if (tileY < numVoxelsPerTile.y - 1)
							    {
								    tileQueue.push( MojoInt4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0) );
							    }
							    else
							    {
								    sliceQueue.insert( std::pair< MojoInt4, MojoInt4 > (
									    MojoInt4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
									    MojoInt4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0 ) ) );
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
						//
						// Split is complete
						//
                        centerX = centerX / (float) nPixChanged;
                        centerY = centerY / (float) nPixChanged;
                        Core::Printf( "Split centroid at ", centerX, "x", centerY, ".");

                        if ( !invert )
                        {
                            if ( foundMouseOverPixel )
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

					            wQueue = std::queue< MojoInt4 >();

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
							            tileIndex = MojoInt4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
							            volumeDescriptions = LoadTile( tileIndex );
							            currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;

							            int xOffset = xd * numVoxelsPerTile.x;
							            int yOffset = yd * numVoxelsPerTile.y;
							            int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

							            for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
							            {
								            int areaX = xOffset + tileIndex1D % numVoxelsPerTile.x;
								            int areaY = yOffset + tileIndex1D / numVoxelsPerTile.x;
								            seedIndex1D = areaX + areaY * mSplitWindowWidth;

											if ( mSegmentInfoManager.GetIdForLabel( currentIdVolume[ tileIndex1D ] ) == segId &&
												( mNextUndoItem->tileChangeMap.find( tileIndex ) == mNextUndoItem->tileChangeMap.end() || 
												 !mNextUndoItem->tileChangeMap[ tileIndex ][ currentIdVolume[ tileIndex1D ] ][ tileIndex1D ] ) &&
												mSplitResultArea[ seedIndex1D ] == 0 )
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
								mNextUndoItem->tileChangeMap.clear();

					            if ( !seedFound )
					            {
						            Core::Printf( "WARNING: Could not find (inverted) seed point - ignoring." );
									mUndoDeque.pop_front();
									break;
					            }
					            else
					            {
						            //Core::Printf( "Seed found at:", seedIndex1D, "." );
					            }

					            //
					            // Use this new seed point
					            //
					            pVoxelSpace = 
						            MojoInt3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
						            seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
						            mSplitWindowStart.z );

                                newId = 0;
                                centerX = 0;
                                centerY = 0;
					            nPixChanged = 0;
					            invert = true;

					            tileQueue.push( MojoInt4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

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

			    FileSystemTileSet tilesContainingNewId;

			    mNextUndoItem->newId = newId;

			    bool tileChanged;

			    //Core::Printf( "Splitting (invert=", invert, ")." );

			    //
			    // Do the split and fill up in w
			    //

			    std::swap(tileQueue, wQueue);
			    int currentW = 0;

			    tileLoaded = false;
			    tileChanged = false;

			    //while ( currentW < numTiles.w )
			    //{
				    Core::Printf( "Splitting at w=", currentW, "." );
				    while ( tileQueue.size() > 0 )
				    {
					    thisVoxel = tileQueue.front();
					    tileQueue.pop();

						if ( currentProgress < maxProgress )
						{
							++currentProgress;
							mCurrentOperationProgress = (float)currentProgress / (float)maxProgress;
						}

					    //
					    // Find the tile for this pixel
					    //
					    MojoInt4 tileIndex = MojoInt4( thisVoxel.x / numVoxelsPerTile.x,
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
									StrideUpIdTileChange( numTiles, numVoxelsPerTile, previousTileIndex, currentIdVolume );

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
									    mNextUndoItem->idTileMapRemoveOldIdSets[ segId ].insert( previousTileIndex );
								    }

							    }
							    UnloadTile( previousTileIndex );
						    }

						    //
						    // Load the current tile
						    //
						    volumeDescriptions = LoadTile( tileIndex );
						    previousTileIndex = tileIndex;
						    currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;
						    currentIdNumVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
						    tileLoaded = true;
						    tileChanged = false;

							//
							// Get or create the change record for this tile
							//
							tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];
					    }
					    int tileX = thisVoxel.x % numVoxelsPerTile.x;
					    int tileY = thisVoxel.y % numVoxelsPerTile.y;

					    MojoInt3 index3D = MojoInt3( tileX, tileY, 0 );
					    int  index1D = Mojo::Core::Index3DToIndex1D( index3D, currentIdNumVoxels );
						int  idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ index1D ] );

					    if ( idValue == segId )
					    {
							(*tileChange)[ currentIdVolume[ index1D ] ].set( index1D );
						    currentIdVolume[ index1D ] = newId;
						    tileChanged = true;
                            if ( currentW == 0 )
                                ++voxelChangeCount;

						    //
						    // Add a scaled-down w to the queue
						    //
						    //if (currentW < numTiles.w-1) wQueue.push( MojoInt4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
					    }

				    }
				    //std::swap(tileQueue, wQueue);
				    //++currentW;
			    //}

			    if ( tileLoaded )
			    {
				    //
				    // Save and unload the previous tile
				    //
				    if ( tileChanged )
				    {
					    SaveTile( previousTileIndex, volumeDescriptions );
						StrideUpIdTileChange( numTiles, numVoxelsPerTile, previousTileIndex, currentIdVolume );

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
						    mNextUndoItem->idTileMapRemoveOldIdSets[ segId ].insert( previousTileIndex );
					    }

				    }
				    UnloadTile( previousTileIndex );
			    }

				if ( voxelChangeCount > 0 )
				{

					//
					// Update the segment sizes
					//
					mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) + voxelChangeCount );
					mSegmentInfoManager.SetVoxelCount( segId, mSegmentInfoManager.GetVoxelCount( segId ) - voxelChangeCount );

					//
					// Update idTileMap
					//
					mSegmentInfoManager.SetTiles( segId, tilesContainingOldId );
					mSegmentInfoManager.SetTiles( newId, tilesContainingNewId );

					//
					// Record new centroid
					//
					newCentroids.push_back( std::pair< MojoFloat2, int >( MojoFloat2( centerX, centerY ), newId ));

					//
					// Recalculate centroid of unchanged label
					//
					mSplitLabelCount -= voxelChangeCount;
					mCentroid.x += ( mCentroid.x - centerX ) * ( (float)voxelChangeCount / (float)mSplitLabelCount );
					mCentroid.y += ( mCentroid.y - centerY ) * ( (float)voxelChangeCount / (float)mSplitLabelCount );
					Core::Printf( "Remaining centroid at ", mCentroid.x, "x", mCentroid.y, "." );

					Core::Printf( "\nFinished Splitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") to new segmentation label ", newId, "...\n" );

					mLogger.Log( Core::ToString( "CompleteDrawSplit: segId=", segId, ", newId=", newId, ", z=", pVoxelSpace.z, ", npixels=", voxelChangeCount ) );

					mCurrentOperationProgress = 1;

				}

		    }

			if ( mSplitLabelCount > 0 )
			{
				newCentroids.push_back( std::pair< MojoFloat2, int>( mCentroid, segId ) );
			}

            if ( join3D && segId == mPrevSplitId && ( mPrevSplitZ == mSplitWindowStart.z - 1 || mPrevSplitZ == mSplitWindowStart.z + 1 ) )
            {

 				//
				// Check for 3D links
				//

				Core::Printf( "Checking for 3D links: ", (unsigned int) newCentroids.size(), " segments, ", (unsigned int) mPrevSplitCentroids.size(), " neighbours." );

				int closestId = -1;
				std::vector< std::pair< MojoFloat2, int >>::iterator closestIt;
				std::map< int, std::pair< int, float > > matches;

                for ( std::vector< std::pair< MojoFloat2, int >>::iterator prevCentroidIt = mPrevSplitCentroids.begin(); prevCentroidIt != mPrevSplitCentroids.end(); ++prevCentroidIt )
                {
					float minDist = -1;
					for ( std::vector< std::pair< MojoFloat2, int >>::iterator newCentroidIt = newCentroids.begin(); newCentroidIt != newCentroids.end(); ++newCentroidIt )
					{
						float thisDist = sqrt ( ( prevCentroidIt->first.x - newCentroidIt->first.x ) * ( prevCentroidIt->first.x - newCentroidIt->first.x ) + ( prevCentroidIt->first.y - newCentroidIt->first.y ) * ( prevCentroidIt->first.y - newCentroidIt->first.y ) );
						if ( minDist < 0 || thisDist < minDist )
						{
							minDist = thisDist;
							closestId = newCentroidIt->second;
							closestIt = newCentroidIt;
						}
					}

					std::map< int, std::pair< int, float > >::iterator existingMatch = matches.find( closestId );
					if ( existingMatch != matches.end() )
					{
						//
						// Already matched - check the score
						//

						if ( existingMatch->second.second > minDist )
						{
							matches[ closestId ] = std::pair< int, float >( prevCentroidIt->second, minDist );
						}
					}
					else
					{
						matches[ closestId ] = std::pair< int, float >( prevCentroidIt->second, minDist );
					}
				}


				//
				// Remap best matching segment ids for 3d Joins
				//

				for ( std::map< int, std::pair< int, float > >::iterator matchIt = matches.begin(); matchIt != matches.end(); ++matchIt )
				{
					int thisId = matchIt->first;
					int prevId = matchIt->second.first;
					float dist = matchIt->second.second;

					if ( thisId != segId && prevId != segId )
					{
						Core::Printf( "Found centroid match at distance ", dist, " previd=", prevId, " newid=", thisId, ".");
						ReplaceSegmentationLabel( thisId, prevId );

						//
						// Update centroid info
						//
						for ( std::vector< std::pair< MojoFloat2, int >>::iterator newCentroidIt = newCentroids.begin(); newCentroidIt != newCentroids.end(); ++newCentroidIt )
						{
							if ( newCentroidIt->second == thisId )
							{
								newCentroidIt->second = prevId;
								break;
							}
						}
					}
					else if ( thisId != segId && prevId == segId )
					{
						//
						// Possible re-merge
						//
						Core::Printf( "WARNING: 3d join attempt to remerge back to original id.");
						Core::Printf( "Found centroid match at distance ", dist, " previd=", prevId, " newid=", thisId, " (not merged).");
					}
					else if ( thisId == segId && prevId != segId )
					{
						//
						// Possible re-merge
						//
						Core::Printf( "WARNING: 3d join attempt merge original id to new segment - possible mouse over conflict.");
						Core::Printf( "Found centroid match at distance ", dist, " previd=", prevId, " newid=", thisId, " (not merged).");
					}
					else
					{
						//
						// Both segId
						//
						Core::Printf( "Found centroid match at distance ", dist, " previd=", prevId, " newid=", thisId, " (nothing to do).");
					}
				}

            }

            //
            // Record join info for next split (up or down in z)
            //
            mPrevSplitId = segId;
            mPrevSplitZ = currentZ;
            mPrevSplitLine = mSplitStates.find( currentZ )->second.splitLine;
            mPrevSplitCentroids = newCentroids;

			//
			// Restrict 3d joins to one z-step at a time
			//
			continueZ = false;

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

void FileSystemTileServer::CommitAdjustChange( unsigned int segId, MojoFloat3 pointTileSpace )
{
	if ( mIsSegmentationLoaded )
    {
        long voxelChangeCount = 0;
        std::map< unsigned int, long > idChangeCounts;

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
		MojoInt4 numTiles = tiledVolumeDescription.numTiles();
        unsigned int* currentIdVolume;
        int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( segId );

		PrepForNextUndoRedoChange();
		mNextUndoItem->newId = segId;

		TileChangeIdMap *tileChange;

        int currentW = 0;
        std::queue< MojoInt4 > tileQueue;
        std::queue< MojoInt4 > wQueue;

        int tileCount = 0;
        int areaCount = 0;

        //bool tileLoaded;
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

                MojoInt4 tileIndex = MojoInt4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
                volumeDescriptions = LoadTile( tileIndex );
                tileChanged = false;
                currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;

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

                    //RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

					unsigned int idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ tileIndex1D ] );
					if ( idValue != segId && mSplitDrawArea[ areaIndex1D ] == REGION_A && mSegmentInfoManager.GetConfidence( idValue ) < 100 )
                    {
                        if ( !tileChanged )
                        {
                            //
                            // Get or create the change record for this tile
                            //
							tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];
                        }

						(*tileChange)[ currentIdVolume[ tileIndex1D ] ].set( tileIndex1D );
                        currentIdVolume[ tileIndex1D ] = segId;
                        tileChanged = true;

                        ++voxelChangeCount;
                        ++idChangeCounts[ idValue ];

                        if ( tilesContainingNewId.find( tileIndex ) == tilesContainingNewId.end() )
                        {
                            tilesContainingNewId.insert( tileIndex );
                        }

                        //
					    // Add a scaled-down w to the queue
					    //
                        //wQueue.push( MojoInt4( ( mSplitWindowStart.x * numVoxelsPerTile.x + areaX ) / 2,
                        //    ( mSplitWindowStart.y * numVoxelsPerTile.y + areaY ) / 2, mSplitWindowStart.z, currentW + 1) );
                    }
                }

                if ( tileChanged )
                {

                    //
                    // Update segment sizes
                    //
                    for ( std::map< unsigned int, long >::iterator changedIt = idChangeCounts.begin(); changedIt != idChangeCounts.end(); ++changedIt )
                    {
                        mSegmentInfoManager.SetVoxelCount( changedIt->first, mSegmentInfoManager.GetVoxelCount( changedIt->first ) - changedIt->second );
                    }

                    //
                    // Check overwritten ids to see if they should be removed from the idTileMap
                    //
                    for ( int tileIndex1D = 0; tileIndex1D < nVoxels; ++tileIndex1D )
                    {
						std::map< unsigned int, long >::iterator matchIt = idChangeCounts.find( mSegmentInfoManager.GetIdForLabel( currentIdVolume[ tileIndex1D ] ) );
                        if ( matchIt != idChangeCounts.end() )
                        {
                            idChangeCounts.erase( matchIt );

                            if ( idChangeCounts.size() == 0 )
                                break;
                        }
                    }

                    for ( std::map< unsigned int, long >::iterator removedIt = idChangeCounts.begin(); removedIt != idChangeCounts.end(); ++removedIt )
                    {
                        mNextUndoItem->idTileMapRemoveOldIdSets[ removedIt->first ].insert( tileIndex );

                        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( removedIt->first );
                        tilesContainingOldId.erase( tileIndex );
                        mSegmentInfoManager.SetTiles( removedIt->first, tilesContainingOldId );
                    }

                    SaveTile( tileIndex, volumeDescriptions );
					StrideUpIdTileChange( numTiles, numVoxelsPerTile, tileIndex, currentIdVolume );
                }
                UnloadTile( tileIndex );
                idChangeCounts.clear();

            }
        }

        //
        // Update segment size
        //
        mSegmentInfoManager.SetVoxelCount( segId, mSegmentInfoManager.GetVoxelCount( segId ) + voxelChangeCount );

        //
        // Update idTileMap
        //
        mSegmentInfoManager.SetTiles( segId, tilesContainingNewId );

        Core::Printf( "\nFinished Adjusting segmentation label ", segId, " in tile z=", pointTileSpace.z, ".\n" );

		mLogger.Log( Core::ToString( "CommitAdjustChange: segId=", segId, ", z=", pointTileSpace.z, ", npixels=", voxelChangeCount ) );

    }
}

std::set< unsigned int > FileSystemTileServer::GetDrawMergeIds( MojoFloat3 pointTileSpace )
{
	std::set< unsigned int > mergeIds;

	if ( mIsSegmentationLoaded )
    {

        Core::Printf( "\nFinding Merge Ids.\n" );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
        int* currentIdVolume;
        int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

        int tileCount = 0;

        //
        // Loop over buffer tiles at w=0
        //
        for ( int xd = 0; xd < mSplitWindowNTiles.x; ++xd )
        {
            for (int yd = 0; yd < mSplitWindowNTiles.y; ++yd )
            {

                ++tileCount;
                //Core::Printf( "Loading distance tile ", tileCount, "." );

                MojoInt4 tileIndex = MojoInt4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
                volumeDescriptions = LoadTile( tileIndex );
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

                    //RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

					if ( mSplitDrawArea[ areaIndex1D ] == REGION_A )
					{
						unsigned int idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ tileIndex1D ] );
						if ( mSegmentInfoManager.GetConfidence( idValue ) < 100 )
						{
							mergeIds.insert( idValue );
						}
					}
                }

                UnloadTile( tileIndex );

            }
        }
    }

	return mergeIds;

}

std::map< unsigned int, MojoFloat3 > FileSystemTileServer::GetDrawMergeIdsAndPoints( MojoFloat3 pointTileSpace )
{
	std::map< unsigned int, MojoFloat3 > mergeIdsAndPoints;

	if ( mIsSegmentationLoaded )
    {

        Core::Printf( "\nFinding Merge Ids.\n" );

        Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions;

        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        MojoInt3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile();
        int* currentIdVolume;
        int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

        int tileCount = 0;

        //
        // Loop over buffer tiles at w=0
        //
        for ( int xd = 0; xd < mSplitWindowNTiles.x; ++xd )
        {
            for (int yd = 0; yd < mSplitWindowNTiles.y; ++yd )
            {

                ++tileCount;
                //Core::Printf( "Loading distance tile ", tileCount, "." );

                MojoInt4 tileIndex = MojoInt4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
                volumeDescriptions = LoadTile( tileIndex );
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

                    //RELEASE_ASSERT( areaIndex1D < mSplitWindowNPix );

					if ( mSplitDrawArea[ areaIndex1D ] == REGION_A )
					{
						unsigned int idValue = mSegmentInfoManager.GetIdForLabel( currentIdVolume[ tileIndex1D ] );
						if ( mSegmentInfoManager.GetConfidence( idValue ) < 100 )
						{
							mergeIdsAndPoints[ idValue ] = MojoFloat3(
								((float) tileIndex.x * numVoxelsPerTile.x + tileIndex1D % numVoxelsPerTile.x ) / ((float) numVoxelsPerTile.x ),
								((float) tileIndex.y * numVoxelsPerTile.y + tileIndex1D / numVoxelsPerTile.x ) / ((float) numVoxelsPerTile.y ),
								(float)mSplitWindowStart.z );
						}
					}
                }

                UnloadTile( tileIndex );

            }
        }
    }

	return mergeIdsAndPoints;

}

unsigned int FileSystemTileServer::CommitDrawMerge( std::set< unsigned int > mergeIds, MojoFloat3 pointTileSpace )
{
	unsigned int largestSegId = 0;

    if ( mIsSegmentationLoaded && mergeIds.size() > 1 )
	{
		//
		// Find the largest segId to merge segments to
		//

		long maxSize = 0;

		for ( std::set< unsigned int >::iterator mergeIt = mergeIds.begin(); mergeIt != mergeIds.end(); ++mergeIt )
		{
			if ( (*mergeIt) != 0 && mSegmentInfoManager.GetVoxelCount( *mergeIt ) > maxSize )
			{
				maxSize = mSegmentInfoManager.GetVoxelCount( *mergeIt );
				largestSegId = *mergeIt;
			}
		}

		//
		// Merge to this segment
		//
		if ( largestSegId != 0 )
		{
			mergeIds.erase( largestSegId );
			RemapSegmentLabels( mergeIds, largestSegId );
		}

	}

    ResetDrawMergeState();

	return largestSegId;

}

unsigned int FileSystemTileServer::CommitDrawMergeCurrentSlice( MojoFloat3 pointTileSpace )
{
	std::map< unsigned int, MojoFloat3 > mergeIdsAndPoints = GetDrawMergeIdsAndPoints( pointTileSpace );

	unsigned int largestSegId = 0;

	if ( mIsSegmentationLoaded && mergeIdsAndPoints.size() == 1 )
	{
		largestSegId = mergeIdsAndPoints.begin()->first;
	}
    else if ( mIsSegmentationLoaded && mergeIdsAndPoints.size() > 1 )
	{
		//
		// Find the largest segId to merge segments to
		//

		long maxSize = 0;

		for ( std::map< unsigned int, MojoFloat3 >::iterator mergeIt = mergeIdsAndPoints.begin(); mergeIt != mergeIdsAndPoints.end(); ++mergeIt )
		{
			if ( mergeIt->first != 0 && mSegmentInfoManager.GetVoxelCount( mergeIt->first ) > maxSize )
			{
				maxSize = mSegmentInfoManager.GetVoxelCount( mergeIt->first );
				largestSegId = mergeIt->first;
			}
		}

		//
		// Merge to this segment
		//
		if ( largestSegId != 0 )
		{
			mergeIdsAndPoints.erase( largestSegId );
			for ( std::map< unsigned int, MojoFloat3 >::iterator mergeIt = mergeIdsAndPoints.begin(); mergeIt != mergeIdsAndPoints.end(); ++mergeIt )
			{
				ReplaceSegmentationLabelCurrentSlice( mergeIt->first, largestSegId, mergeIt->second );
			}
		}

	}

    ResetDrawMergeState();

	return largestSegId;

}

unsigned int FileSystemTileServer::CommitDrawMergeCurrentConnectedComponent( MojoFloat3 pointTileSpace )
{
	std::map< unsigned int, MojoFloat3 > mergeIdsAndPoints = GetDrawMergeIdsAndPoints( pointTileSpace );

	unsigned int largestSegId = 0;


	if ( mIsSegmentationLoaded && mergeIdsAndPoints.size() == 1 )
	{
		largestSegId = mergeIdsAndPoints.begin()->first;
	}
    else if ( mIsSegmentationLoaded && mergeIdsAndPoints.size() > 1 )
	{
		//
		// Find the largest segId to merge segments to
		//

		long maxSize = 0;

		for ( std::map< unsigned int, MojoFloat3 >::iterator mergeIt = mergeIdsAndPoints.begin(); mergeIt != mergeIdsAndPoints.end(); ++mergeIt )
		{
			if ( mergeIt->first != 0 && mSegmentInfoManager.GetVoxelCount( mergeIt->first ) > maxSize )
			{
				maxSize = mSegmentInfoManager.GetVoxelCount( mergeIt->first );
				largestSegId = mergeIt->first;
			}
		}

		//
		// Merge to this segment
		//
		if ( largestSegId != 0 )
		{
			mergeIdsAndPoints.erase( largestSegId );
			for ( std::map< unsigned int, MojoFloat3 >::iterator mergeIt = mergeIdsAndPoints.begin(); mergeIt != mergeIdsAndPoints.end(); ++mergeIt )
			{
				ReplaceSegmentationLabelCurrentConnectedComponent( mergeIt->first, largestSegId, mergeIt->second );
			}
		}

	}

    ResetDrawMergeState();

	return largestSegId;

}
void FileSystemTileServer::FindBoundaryJoinPoints2D( unsigned int segId )
{
	//
	// Find a splitting line that links all the given points
	// Save the result into a temp tile
	//
    if ( mIsSegmentationLoaded && mSplitSourcePoints.size() > 0 )
    {
		Core::Printf( "\nFinding Split line for segment ", segId, " with ", (unsigned int) mSplitSourcePoints.size(), " split segments.\n" );

        MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();

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

void FileSystemTileServer::FindBoundaryWithinRegion2D( unsigned int segId )
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

void FileSystemTileServer::FindBoundaryBetweenRegions2D( unsigned int segId )
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

unsigned int FileSystemTileServer:: GetNewId()
{
	return mSegmentInfoManager.AddNewId();
}

float FileSystemTileServer::GetCurrentOperationProgress()
{
	return mCurrentOperationProgress;
}

//
// Undo / Redo Methods
//

std::list< unsigned int > FileSystemTileServer::UndoChange()
{
	std::list< unsigned int > remappedIds;
	//
	// If we are in the middle of the split reset the state (no redo)
	//
	if ( mSplitSourcePoints.size() > 0 )
	{
		ResetSplitState();
		return remappedIds;
	}

    //
    // Reset predict split
    //
    if ( mPrevSplitId != 0 )
    {
        mPrevSplitId = 0;
        mPrevSplitZ = -2;
        mPrevSplitLine.clear();
        mPrevSplitCentroids.clear();
    }

    if ( mUndoDeque.size() > 0 )
    {
        long voxelChangeCount = 0;
        std::map< unsigned int, long > idChangeCounts;

        FileSystemUndoRedoItem UndoItem = mUndoDeque.front();

		MojoInt4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles();
		MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();

	    unsigned int newId = UndoItem.newId;

	    if ( newId != 0 && mIsSegmentationLoaded )
	    {
			if ( UndoItem.remapFromIdsAndSizes.size() > 0 )
			{
				Core::Printf( "\nUndo operation: Unmapping ", (int) UndoItem.remapFromIdsAndSizes.size(), " segmentation labels away from ", newId, "...\n" );

				mLogger.Log( Core::ToString( "Undo (remap): newId=", newId, " Remaps=", (int)UndoItem.remapFromIdsAndSizes.size() ) );

				for ( std::map< unsigned int, long >::iterator mapIt = UndoItem.remapFromIdsAndSizes.begin(); mapIt != UndoItem.remapFromIdsAndSizes.end(); ++mapIt )
				{
					mSegmentInfoManager.RemapSegmentLabel( mapIt->first, mapIt->first );
					mSegmentInfoManager.SetVoxelCount( mapIt->first, mapIt->second );
					mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) - mapIt->second );
					remappedIds.push_back( mapIt->first );
				}
			}

			if ( UndoItem.tileChangeMap.size() > 0 )
			{
				Core::Printf( "\nUndo operation: changing segmentation label ", newId, " back to multiple segmentation labels...\n" );

				mLogger.Log( Core::ToString( "Undo (tile change): newId=", newId, " Tiles=", (int)UndoItem.tileChangeMap.size() ) );

				stdext::hash_map< std::string, std::set< MojoInt2, Core::Int2Comparator > >::iterator changeSetIt;

				for ( TileChangeMap::iterator tileChangeIt = UndoItem.tileChangeMap.begin(); tileChangeIt != UndoItem.tileChangeMap.end(); ++tileChangeIt )
				{

					MojoInt4 tileIndex = tileChangeIt->first;

					if ( tileIndex.w == 0 )
					{
						//
						// load tile
						//
						Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
						unsigned int* currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;

						//
						// replace the new id and color with the previous id and color(s)...
						//
						for ( TileChangeIdMap::iterator changeBitsIt = tileChangeIt->second.begin(); changeBitsIt != tileChangeIt->second.end(); ++changeBitsIt )
						{
							for ( unsigned int i = 0; i < changeBitsIt->second.size(); ++i )
							{
								if ( changeBitsIt->second[ i ] )
								{
									++voxelChangeCount;
									++idChangeCounts[ mSegmentInfoManager.GetIdForLabel( changeBitsIt->first ) ];
									//RELEASE_ASSERT( indexIt->x < TILE_PIXELS * TILE_PIXELS );
									currentIdVolume[ i ] = changeBitsIt->first;
								}
							}
						}

						//
						// save tile
						//
						SaveTile( tileIndex, volumeDescriptions );
						StrideUpIdTileChange( numTiles, numVoxelsPerTile, tileIndex, currentIdVolume );

						//
						// unload tile
						//
						UnloadTile( tileIndex );

					}
				}
			}

            //
            // Update the segment sizes
            //
            mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) - voxelChangeCount );
            for ( std::map< unsigned int, long >::iterator oldIdSizeIt = idChangeCounts.begin(); oldIdSizeIt != idChangeCounts.end(); ++oldIdSizeIt )
            {
                mSegmentInfoManager.SetVoxelCount( oldIdSizeIt->first, mSegmentInfoManager.GetVoxelCount( oldIdSizeIt->first ) + oldIdSizeIt->second );
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
            for ( std::map< unsigned int, FileSystemTileSet >::iterator oldIdIt = UndoItem.idTileMapRemoveOldIdSets.begin(); oldIdIt != UndoItem.idTileMapRemoveOldIdSets.end(); ++oldIdIt )
            {
                FileSystemTileSet oldTiles = mSegmentInfoManager.GetTiles( oldIdIt->first );
                oldTiles.insert( oldIdIt->second.begin(), oldIdIt->second.end() );
                mSegmentInfoManager.SetTiles( oldIdIt->first , oldTiles );
            }

            Core::Printf( "\nUndo operation complete.\n" );

		    //
		    // Make this a redo item
		    //
			mUndoDeque.pop_front();
            mRedoDeque.push_front( UndoItem );
            mNextUndoItem = &mUndoDeque.front();
        }
		else
		{
			Core::Printf( "\nWarning - invalid undo item - discarding.\n" );
			mUndoDeque.pop_front();
			mNextUndoItem = &mUndoDeque.front();
		}
	}
	else
	{
		Core::Printf( "\nWarning - invalid undo item - discarding.\n" );
        mUndoDeque.pop_front();
        mNextUndoItem = &mUndoDeque.front();
	}

	return remappedIds;

}

std::list< unsigned int > FileSystemTileServer::RedoChange()
{
	std::list< unsigned int > remappedIds;

    if ( mRedoDeque.size() > 0 )
    {
        long voxelChangeCount = 0;
        std::map< unsigned int, long > idChangeCounts;

        FileSystemUndoRedoItem RedoItem = mRedoDeque.front();

		MojoInt4 numTiles = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles();
		MojoInt3 numVoxelsPerTile = mTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile();

	    int newId = RedoItem.newId;

	    if ( newId != 0 && mIsSegmentationLoaded )
	    {
			if ( RedoItem.remapFromIdsAndSizes.size() > 0 )
			{
				Core::Printf( "\nRedo operation: Remapping ", (int) RedoItem.remapFromIdsAndSizes.size(), " segmentation labels to ", newId, "...\n" );

				mLogger.Log( Core::ToString( "Redo (remap): newId=", newId, " Remaps=", (int)RedoItem.remapFromIdsAndSizes.size() ) );

				for ( std::map< unsigned int, long >::iterator mapIt = RedoItem.remapFromIdsAndSizes.begin(); mapIt != RedoItem.remapFromIdsAndSizes.end(); ++mapIt )
				{
					mSegmentInfoManager.RemapSegmentLabel( mapIt->first, newId );
					mSegmentInfoManager.SetVoxelCount( mapIt->first, 0 );
					mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) + mapIt->second );
					remappedIds.push_back( mapIt->first );
				}
			}

			if ( RedoItem.tileChangeMap.size() > 0 )
			{

				Core::Printf( "\nRedo operation: changing multiple segmentation labels back to ", newId, "...\n" );

				mLogger.Log( Core::ToString( "Redo (tile change): newId=", newId, " Tiles=", (int)RedoItem.tileChangeMap.size() ) );

				stdext::hash_map< std::string, std::set< MojoInt2, Core::Int2Comparator > >::iterator changeSetIt;

				for ( TileChangeMap::iterator tileChangeIt = RedoItem.tileChangeMap.begin(); tileChangeIt != RedoItem.tileChangeMap.end(); ++tileChangeIt )
				{

					MojoInt4 tileIndex = tileChangeIt->first;

					if ( tileIndex.w == 0 )
					{

						//
						// load tile
						//
						Core::HashMap< std::string, Core::VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
						unsigned int* currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;

						//
						// replace the old id and color with the new id and color...
						//
						for ( TileChangeIdMap::iterator changeBitsIt = tileChangeIt->second.begin(); changeBitsIt != tileChangeIt->second.end(); ++changeBitsIt )
						{
							for ( unsigned int i = 0; i < changeBitsIt->second.size(); ++i )
							{
								if ( changeBitsIt->second[ i ] )
								{
									++voxelChangeCount;
									++idChangeCounts[ mSegmentInfoManager.GetIdForLabel( changeBitsIt->first ) ];
									//RELEASE_ASSERT( indexIt->x < TILE_PIXELS * TILE_PIXELS );
									//RELEASE_ASSERT( currentIdVolume[ i ] == changeBitsIt->first );
									currentIdVolume[ i ] = newId;
								}
							}
						}

						//
						// save tile
						//
						SaveTile( tileIndex, volumeDescriptions );
						StrideUpIdTileChange( numTiles, numVoxelsPerTile, tileIndex, currentIdVolume );

						//
						// unload tile
						//
						UnloadTile( tileIndex );

					}
				}
			}

            //
            // Update the segment sizes
            //
            mSegmentInfoManager.SetVoxelCount( newId, mSegmentInfoManager.GetVoxelCount( newId ) + voxelChangeCount );
            for ( std::map< unsigned int, long >::iterator oldIdSizeIt = idChangeCounts.begin(); oldIdSizeIt != idChangeCounts.end(); ++oldIdSizeIt )
            {
                mSegmentInfoManager.SetVoxelCount( oldIdSizeIt->first, mSegmentInfoManager.GetVoxelCount( oldIdSizeIt->first ) - oldIdSizeIt->second );
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
            for ( std::map< unsigned int, FileSystemTileSet >::iterator oldIdIt = RedoItem.idTileMapRemoveOldIdSets.begin(); oldIdIt != RedoItem.idTileMapRemoveOldIdSets.end(); ++oldIdIt )
            {
                FileSystemTileSet oldTiles = mSegmentInfoManager.GetTiles( oldIdIt->first );
                for ( FileSystemTileSet::iterator eraseIterator = oldIdIt->second.begin(); eraseIterator != oldIdIt->second.end(); ++eraseIterator )
		        {
			        oldTiles.erase( *eraseIterator );
		        }
                mSegmentInfoManager.SetTiles( oldIdIt->first, oldTiles );
            }

            Core::Printf( "\nRedo operation complete.\n" );

		    //
		    // Make this a redo item
		    //
            mRedoDeque.pop_front();
            mUndoDeque.push_front( RedoItem );
            mNextUndoItem = &mUndoDeque.front();
	    }
		else
		{
			Core::Printf( "\nWarning - invalid redo item - discarding.\n" );
			mRedoDeque.pop_front();
		}
    }
	else
	{
		Core::Printf( "\nWarning - invalid redo item - discarding.\n" );
        mRedoDeque.pop_front();
	}

	return remappedIds;

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
	FileSystemUndoRedoItem newUndoRedoItem;
    mUndoDeque.push_front( newUndoRedoItem );

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

Core::HashMap< std::string, Core::VolumeDescription > FileSystemTileServer::LoadTile( MojoInt4 tileIndex )
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


void FileSystemTileServer::UnloadTile( MojoInt4 tileIndex )
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

    //
    // release log file
    //
	mLogger.Log( "Segmentation Unloaded.");
	mLogger.CloseLog();

    //
    // Not necessary if we are closing
    //
    //mSegmentInfoManager = FileSystemSegmentInfoManager();

    mTiledDatasetDescription.maxLabelId = 0;

    mIsSegmentationLoaded    = false;
    mTiledDatasetDescription = TiledDatasetDescription();
}

Core::VolumeDescription FileSystemTileServer::LoadTileImage( MojoInt4 tileIndex, std::string imageName )
{
    Core::VolumeDescription volumeDescription;

    bool success = TryLoadTileImage( tileIndex, imageName, volumeDescription );

    RELEASE_ASSERT( success );

    return volumeDescription;
}

bool FileSystemTileServer::TryLoadTileImage( MojoInt4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription )
{
    switch ( mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).dxgiFormat )
    {
    case DXGI_FORMAT_R8_UNORM:
        return TryLoadTileImageInternalUChar1( tileIndex, imageName, volumeDescription );
        break;

    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return TryLoadTileImageInternalUChar4( tileIndex, imageName, volumeDescription );
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

Core::VolumeDescription FileSystemTileServer::LoadTileHdf5( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName )
{
    Core::VolumeDescription volumeDescription;

    bool success = TryLoadTileHdf5( tileIndex, hdf5Name, hdf5InternalDatasetName, volumeDescription );

    RELEASE_ASSERT( success );

    return volumeDescription;
}

bool FileSystemTileServer::TryLoadTileHdf5( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, Core::VolumeDescription& volumeDescription )
{
    switch ( mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).dxgiFormat )
    {
    case DXGI_FORMAT_R32_UINT:
        return TryLoadTileHdf5Internal( tileIndex, hdf5Name, hdf5InternalDatasetName, volumeDescription );
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

void FileSystemTileServer::SaveTile( MojoInt4 tileIndex, Core::HashMap< std::string, Core::VolumeDescription >& volumeDescriptions )
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

void FileSystemTileServer::SaveTileImage( MojoInt4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription )
{
    switch ( mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).dxgiFormat )
    {
    case DXGI_FORMAT_R8_UNORM:
        SaveTileImageInternalUChar1( tileIndex, imageName, volumeDescription );
        break;

    case DXGI_FORMAT_R8G8B8A8_UNORM:
        SaveTileImageInternalUChar4( tileIndex, imageName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        break;
    }
}

void FileSystemTileServer::SaveTileHdf5( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, const Core::VolumeDescription& volumeDescription )
{
    switch ( mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).dxgiFormat )
    {
    case DXGI_FORMAT_R32_UINT:
        return SaveTileHdf5Internal( tileIndex, hdf5Name, hdf5InternalDatasetName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        break;
    }
}


//
// Cache Methods
//

std::string FileSystemTileServer::CreateTileString( MojoInt4 tileIndex )
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

MojoInt4 FileSystemTileServer::CreateTileIndex( std::string tileString )
{
	int w, x, y, z;
	sscanf_s( tileString.c_str(), "W%dX%dY%dZ%d", &w, &x, &y, &z );
	//sscanf_s( tileString.c_str(), "w=%dz=%dy=%dx=%d", &w, &z, &y, &x );
	return MojoInt4( x, y, z, w );
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

void FileSystemTileServer::TempSaveFileSystemTileCacheChanges()
{
    for( stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator i = mFileSystemTileCache.GetHashMap().begin(); i != mFileSystemTileCache.GetHashMap().end(); i++ )
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
    }
}

void FileSystemTileServer::ClearFileSystemTileCache()
{
    stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator i = mFileSystemTileCache.GetHashMap().begin();
    while( i != mFileSystemTileCache.GetHashMap().end() )
    {
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
}

void FileSystemTileServer::TempSaveAndClearFileSystemTileCache( )
{
	TempSaveFileSystemTileCacheChanges();
	ClearFileSystemTileCache();
    RELEASE_ASSERT( mFileSystemTileCache.GetHashMap().size() == 0 );
}

marray::Marray< unsigned char >* FileSystemTileServer::GetIdColorMap()
{
    return mSegmentInfoManager.GetIdColorMap();
}

marray::Marray< unsigned int >* FileSystemTileServer::GetLabelIdMap()
{
    return mSegmentInfoManager.GetLabelIdMap();
}

marray::Marray< unsigned char >* FileSystemTileServer::GetIdConfidenceMap()
{
    return mSegmentInfoManager.GetIdConfidenceMap();
}

void FileSystemTileServer::SortSegmentInfoById( bool reverse )
{
	mSegmentInfoManager.SortSegmentInfoById( reverse );
}

void FileSystemTileServer::SortSegmentInfoByName( bool reverse )
{
	mSegmentInfoManager.SortSegmentInfoByName( reverse );
}

void FileSystemTileServer::SortSegmentInfoBySize( bool reverse )
{
	mSegmentInfoManager.SortSegmentInfoBySize( reverse );
}

void FileSystemTileServer::SortSegmentInfoByConfidence( bool reverse )
{
	mSegmentInfoManager.SortSegmentInfoByConfidence( reverse );
}

void FileSystemTileServer::LockSegmentLabel( unsigned int segId )
{
	mSegmentInfoManager.LockSegmentLabel( segId );
	mLogger.Log( Core::ToString( "LockSegmentLabel: segId=", segId ) );
}

void FileSystemTileServer::UnlockSegmentLabel( unsigned int segId )
{
	mSegmentInfoManager.UnlockSegmentLabel( segId );
	mLogger.Log( Core::ToString( "UnlockSegmentLabel: segId=", segId ) );
}

unsigned int FileSystemTileServer::GetSegmentInfoCount()
{
	return mSegmentInfoManager.GetSegmentInfoCount();
}

unsigned int FileSystemTileServer::GetSegmentInfoCurrentListLocation( unsigned int segId )
{
	return mSegmentInfoManager.GetSegmentInfoCurrentListLocation( segId );
}

std::list< SegmentInfo > FileSystemTileServer::GetSegmentInfoRange( int begin, int end )
{
	return mSegmentInfoManager.GetSegmentInfoRange( begin, end );
}

SegmentInfo FileSystemTileServer::GetSegmentInfo( unsigned int segId )
{
	return mSegmentInfoManager.GetSegmentInfo( segId );
}

FileSystemSegmentInfoManager* FileSystemTileServer::GetSegmentInfoManager()
{
	return &mSegmentInfoManager;
}

//
// Internal template functions moved here to resolve x64 linking errors
//


void FileSystemTileServer::LoadTiledDatasetInternal( TiledDatasetDescription& tiledDatasetDescription )
{
    mTiledDatasetDescription = tiledDatasetDescription;
    mIsTiledDatasetLoaded    = true;
}

void FileSystemTileServer::LoadSegmentationInternal( TiledDatasetDescription& tiledDatasetDescription )
{
    mTiledDatasetDescription = tiledDatasetDescription;

    Core::Printf( "Loading idMaps..." );

    mSegmentInfoManager = FileSystemSegmentInfoManager( mTiledDatasetDescription.paths.Get( "ColorMap" ), mTiledDatasetDescription.paths.Get( "SegmentInfo" ) );
	mSegmentInfoManager.OpenDB();

    Core::Printf( "Opening log file (", mTiledDatasetDescription.paths.Get( "Log" ), ")..." );

	mLogger.OpenLog( mTiledDatasetDescription.paths.Get( "Log" ) );
	mLogger.Log("Segmentation Loaded.");

    Core::Printf( "Loaded." );

    mTiledDatasetDescription.maxLabelId = mSegmentInfoManager.GetMaxId();

    mIsSegmentationLoaded    = true;

    //Core::Printf( "FileSystemTileServer::LoadSegmentationInternal Returning." );
}

bool FileSystemTileServer::TryLoadTileImageInternalUChar1( MojoInt4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName );

        volumeDescription.data             = new unsigned char[ tiledVolumeDescription.numVoxelsPerTileY * tiledVolumeDescription.numVoxelsPerTileX * tiledVolumeDescription.numBytesPerVoxel ];
        volumeDescription.dxgiFormat       = tiledVolumeDescription.dxgiFormat;
        volumeDescription.isSigned         = tiledVolumeDescription.isSigned;
        volumeDescription.numBytesPerVoxel = tiledVolumeDescription.numBytesPerVoxel;
        volumeDescription.numVoxels        = tiledVolumeDescription.numVoxelsPerTile();

        int flags = 0; // force greyscale
        cv::Mat tileImage = cv::imread( tilePath, flags );

        RELEASE_ASSERT( tileImage.cols        == tiledVolumeDescription.numVoxelsPerTileX );
        RELEASE_ASSERT( tileImage.rows        == tiledVolumeDescription.numVoxelsPerTileY );
        RELEASE_ASSERT( tileImage.elemSize()  == 1 );
        RELEASE_ASSERT( tileImage.elemSize1() == 1 );
        RELEASE_ASSERT( tileImage.channels()  == 1 );
        RELEASE_ASSERT( tileImage.type()      == CV_8UC1 );
        RELEASE_ASSERT( tileImage.depth()     == CV_8U );
        RELEASE_ASSERT( tileImage.isContinuous() );

        memcpy( volumeDescription.data, tileImage.ptr(), tiledVolumeDescription.numVoxelsPerTileY * tiledVolumeDescription.numVoxelsPerTileX * tiledVolumeDescription.numBytesPerVoxel );

        return true;
    }
    else
    {
        return false;
    }
}

bool FileSystemTileServer::TryLoadTileImageInternalUChar4( MojoInt4 tileIndex, std::string imageName, Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        TiledVolumeDescription tiledVolumeDescription = mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName );
        MojoInt3 numVoxelsPerTile              = tiledVolumeDescription.numVoxelsPerTile();
        int  numBytesPerVoxel              = tiledVolumeDescription.numBytesPerVoxel;

        volumeDescription.data             = new unsigned char[ numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel ];
        volumeDescription.dxgiFormat       = tiledVolumeDescription.dxgiFormat;
        volumeDescription.isSigned         = tiledVolumeDescription.isSigned;
        volumeDescription.numBytesPerVoxel = tiledVolumeDescription.numBytesPerVoxel;
        volumeDescription.numVoxels        = numVoxelsPerTile;

        cv::Mat tileImage = cv::imread( tilePath );

        std::vector<cv::Mat> tileImageChannels;
        cv::split( tileImage, tileImageChannels );

        cv::Mat tileImageR;
        cv::Mat tileImageG;
        cv::Mat tileImageB;
        cv::Mat tileImageA;

        if ( mConstParameters.Get< bool >( "SWAP_COLOR_CHANNELS" ) )
        {
            tileImageR = tileImageChannels[ 2 ];
            tileImageG = tileImageChannels[ 1 ];
            tileImageB = tileImageChannels[ 0 ];
            tileImageA = cv::Mat::zeros( tileImageR.rows, tileImageR.cols, CV_8UC1 );
        }
        else
        {
            tileImageR = tileImageChannels[ 0 ];
            tileImageG = tileImageChannels[ 1 ];
            tileImageB = tileImageChannels[ 2 ];
            tileImageA = cv::Mat::zeros( tileImageR.rows, tileImageR.cols, CV_8UC1 );
        }

        tileImageChannels.clear();

        tileImageChannels.push_back( tileImageR );
        tileImageChannels.push_back( tileImageG );
        tileImageChannels.push_back( tileImageB );
        tileImageChannels.push_back( tileImageA );

        cv::merge( tileImageChannels, tileImage );

        RELEASE_ASSERT( tileImage.cols        == numVoxelsPerTile.x );
        RELEASE_ASSERT( tileImage.rows        == numVoxelsPerTile.y );
        RELEASE_ASSERT( tileImage.elemSize()  == 4 );
        RELEASE_ASSERT( tileImage.elemSize1() == 1 );
        RELEASE_ASSERT( tileImage.channels()  == 4 );
        RELEASE_ASSERT( tileImage.type()      == CV_8UC4 );
        RELEASE_ASSERT( tileImage.depth()     == CV_8U );
        RELEASE_ASSERT( tileImage.isContinuous() );

        memcpy( volumeDescription.data, tileImage.ptr(), numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel );

        return true;
    }
    else
    {
        return false;
    }
}

//
// Type is always unsigned int
//

bool FileSystemTileServer::TryLoadTileHdf5Internal( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        MojoInt3 numVoxelsPerTile              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).numVoxelsPerTile();
        int  numBytesPerVoxel              = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).numBytesPerVoxel;

        volumeDescription.data             = new unsigned char[ numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel ];
        volumeDescription.dxgiFormat       = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).dxgiFormat;
        volumeDescription.isSigned         = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).isSigned;
        volumeDescription.numBytesPerVoxel = mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).numBytesPerVoxel;
        volumeDescription.numVoxels        = numVoxelsPerTile;

		//Core::Printf( "Loading tile ", tilePath, "...");

        hid_t hdf5FileHandle = marray::hdf5::openFile( tilePath );
        marray::Marray< unsigned int > marray;
        try
        {
            marray::hdf5::load( hdf5FileHandle, hdf5InternalDatasetName, marray );
        }
        catch (...)
        {
            Core::Printf( "Warning - error loading hdf5 tile. Attempting to reduce cache size." );
            ReduceCacheSize();
            marray::hdf5::load( hdf5FileHandle, hdf5InternalDatasetName, marray );
        }
        marray::hdf5::closeFile( hdf5FileHandle );

		//Core::Printf( "Done.");

        RELEASE_ASSERT( marray.dimension() == 2 );
        RELEASE_ASSERT( marray.shape( 0 ) == numVoxelsPerTile.y && marray.shape( 1 ) == numVoxelsPerTile.y );

        memcpy( volumeDescription.data, &marray( 0 ), numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel );

        return true;
    }
    else
    {
        return false;
    }
}

void FileSystemTileServer::SaveTileImageInternalUChar4( MojoInt4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).fileExtension );

    RELEASE_ASSERT( boost::filesystem::exists( tilePath ) );

    cv::Mat tileImage = cv::Mat::zeros( volumeDescription.numVoxels.y, volumeDescription.numVoxels.x, CV_8UC4 );

    memcpy( tileImage.ptr(), volumeDescription.data, volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );

    std::vector<cv::Mat> tileImageChannels;
    cv::split( tileImage, tileImageChannels );

    cv::Mat tileImageR;
    cv::Mat tileImageG;
    cv::Mat tileImageB;

    if ( mConstParameters.Get< bool >( "SWAP_COLOR_CHANNELS" ) )
    {
        tileImageR = tileImageChannels[ 2 ];
        tileImageG = tileImageChannels[ 1 ];
        tileImageB = tileImageChannels[ 0 ];
    }
    else
    {
        tileImageR = tileImageChannels[ 0 ];
        tileImageG = tileImageChannels[ 1 ];
        tileImageB = tileImageChannels[ 2 ];
    }

    tileImageChannels.clear();

    tileImageChannels.push_back( tileImageR );
    tileImageChannels.push_back( tileImageG );
    tileImageChannels.push_back( tileImageB );

    cv::merge( tileImageChannels, tileImage );

    RELEASE_ASSERT( tileImage.cols        == volumeDescription.numVoxels.x );
    RELEASE_ASSERT( tileImage.rows        == volumeDescription.numVoxels.y );
    RELEASE_ASSERT( tileImage.elemSize()  == 3 );
    RELEASE_ASSERT( tileImage.elemSize1() == 1 );
    RELEASE_ASSERT( tileImage.channels()  == 3 );
    RELEASE_ASSERT( tileImage.type()      == CV_8UC3 );
    RELEASE_ASSERT( tileImage.depth()     == CV_8U );
    RELEASE_ASSERT( tileImage.isContinuous() );

    cv::imwrite( tilePath, tileImage );
}

void FileSystemTileServer::SaveTileImageInternalUChar1( MojoInt4 tileIndex, std::string imageName, const Core::VolumeDescription& volumeDescription )
{
    std::string tilePath = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( imageName ).fileExtension );

    RELEASE_ASSERT( boost::filesystem::exists( tilePath ) );

    cv::Mat tileImage = cv::Mat::zeros( volumeDescription.numVoxels.y, volumeDescription.numVoxels.x, CV_8UC1 );

    memcpy( tileImage.ptr(), volumeDescription.data, volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );

    RELEASE_ASSERT( tileImage.cols        == volumeDescription.numVoxels.x );
    RELEASE_ASSERT( tileImage.rows        == volumeDescription.numVoxels.y );
    RELEASE_ASSERT( tileImage.elemSize()  == 3 );
    RELEASE_ASSERT( tileImage.elemSize1() == 1 );
    RELEASE_ASSERT( tileImage.channels()  == 3 );
    RELEASE_ASSERT( tileImage.type()      == CV_8UC3 );
    RELEASE_ASSERT( tileImage.depth()     == CV_8U );
    RELEASE_ASSERT( tileImage.isContinuous() );

    cv::imwrite( tilePath, tileImage );
}

//
// Type is always unsigned int
//

void FileSystemTileServer::SaveTileHdf5Internal( MojoInt4 tileIndex, std::string hdf5Name, std::string hdf5InternalDatasetName, const Core::VolumeDescription& volumeDescription )
{

    size_t shape[] = { volumeDescription.numVoxels.y, volumeDescription.numVoxels.x };
    marray::Marray< unsigned int > marray( shape, shape + 2 );

    memcpy( &marray( 0 ), volumeDescription.data, volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );

    std::string tilePathString = Core::ToString(
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).imageDataDirectory, "\\",
        "w=", Core::ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", Core::ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", Core::ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", Core::ToStringZeroPad( tileIndex.x, 8 ), ".",
        mTiledDatasetDescription.tiledVolumeDescriptions.Get( hdf5Name ).fileExtension );

    boost::filesystem::path tilePath = boost::filesystem::path( tilePathString );

    if ( !boost::filesystem::exists( tilePath ) )
    {
        boost::filesystem::create_directories( tilePath.parent_path() );

        hid_t hdf5FileHandle = marray::hdf5::createFile( tilePath.native_file_string() );
        marray::hdf5::save( hdf5FileHandle, hdf5InternalDatasetName, marray );
        marray::hdf5::closeFile( hdf5FileHandle );
    }
    else
    {
        size_t origin[]       = { 0, 0 };
        size_t shape[]        = { marray.shape( 0 ), marray.shape( 1 ) };
        hid_t  hdf5FileHandle = marray::hdf5::openFile( tilePath.native_file_string(), marray::hdf5::READ_WRITE );
        
        marray::hdf5::saveHyperslab( hdf5FileHandle, hdf5InternalDatasetName, origin, origin + 2, shape, marray );
        marray::hdf5::closeFile( hdf5FileHandle );
    }
}

}
}