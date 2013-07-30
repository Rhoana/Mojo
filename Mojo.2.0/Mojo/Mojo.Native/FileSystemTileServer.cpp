#include "FileSystemTileServer.hpp"

#include "Stl.hpp"

#include <marray/marray.hxx>
#include <marray/marray_hdf5.hxx>

#include "OpenCV.hpp"
#include "ForEach.hpp"
#include "D3D11Texture.hpp"
#include "ForEach.hpp"

#include "TileCacheEntry.hpp"
#include "SimpleSplitTools.hpp"

namespace Mojo
{
namespace Native
{

FileSystemTileServer::FileSystemTileServer( PrimitiveMap constParameters ) :
    mConstParameters      ( constParameters ),
    mAreSourceImagesLoaded( false ),
    mIsSegmentationLoaded ( false )
{
    mSplitStepDist            = 0;
    mSplitResultDist          = 0;
    mSplitPrev                = 0;
    mSplitSearchMask          = 0;
    mSplitDrawArea            = 0;
    mSplitBorderTargets       = 0;
    mSplitResultArea          = 0;

    mPrevSplitId              = 0;
    mPrevSplitZ               = -2;

    mCurrentOperationProgress = 0;
}

FileSystemTileServer::~FileSystemTileServer()
{
    if ( mIsSegmentationLoaded )
    {
        UnloadSegmentation();
    }

    if ( mAreSourceImagesLoaded )
    {
        UnloadSourceImages();
    }

    //
    // CODE QUALITY ISSUE:
    // This memory is deallocated in the destructor, but not allocated in the constructor. The memory allocation pattern asymmetrical. If memory is allocated on load, it should be deallocated on unload. -MR
    // -MR
    //

    if ( mSplitStepDist != NULL )
        delete[] mSplitStepDist;

    if ( mSplitResultDist != NULL )
        delete[] mSplitResultDist;

    if ( mSplitPrev != NULL )
        delete[] mSplitPrev;

    if ( mSplitSearchMask != NULL )
        delete[] mSplitSearchMask;

    if ( mSplitDrawArea != NULL )
        delete[] mSplitDrawArea;

    if ( mSplitBorderTargets != NULL )
        delete[] mSplitBorderTargets;

    if ( mSplitResultArea != NULL )
        delete[] mSplitResultArea;
}

//
// Dataset Methods
//

void FileSystemTileServer::LoadSourceImages( TiledDatasetDescription& tiledDatasetDescription )
{
    RELEASE_ASSERT( tiledDatasetDescription.tiledVolumeDescriptions.Get( "SourceMap" ).dxgiFormat == DXGI_FORMAT_R8_UNORM );

    mSourceImagesTiledDatasetDescription = tiledDatasetDescription;
    mAreSourceImagesLoaded                = true;
}

void FileSystemTileServer::UnloadSourceImages()
{
    ClearFileSystemTileCache();

    mAreSourceImagesLoaded                = false;
    mSourceImagesTiledDatasetDescription = TiledDatasetDescription();
}

bool FileSystemTileServer::AreSourceImagesLoaded()
{
    return mAreSourceImagesLoaded;
}

void FileSystemTileServer::LoadSegmentation( TiledDatasetDescription& tiledDatasetDescription )
{
    mSegmentationTiledDatasetDescription = tiledDatasetDescription;

    //
    // Make sure there are not temp files hanging around from last time (e.g., if Mojo crashed).
    //
    ClearFileSystemTileCache();
    DeleteTempFiles();

    //
    // We need to copy the DB to a temp location and copy back when saving.
    // This is required for "save as" functionality. The user might load,
    // make changes to the segmentation, then "save as". In order to persist
    // the user's changes, we need to flush the in-memory DB changes to disk.
    // Since the user only performed a "save as", rather than a save, we shouldn't
    // touch the original segmentation. Therefore, we need to work in a temp location.
    //
    boost::filesystem::path idTileIndexDBPath     = boost::filesystem::path( mSegmentationTiledDatasetDescription.paths.Get( "SegmentInfo" ) );
    boost::filesystem::path tempIdTileIndexDBPath = boost::filesystem::path( mSegmentationTiledDatasetDescription.paths.Get( "TempSegmentInfo" ) );
    boost::filesystem::path colorMapPath          = boost::filesystem::path( mSegmentationTiledDatasetDescription.paths.Get( "ColorMap" ) );
    boost::filesystem::path tempColorMapPath      = boost::filesystem::path( mSegmentationTiledDatasetDescription.paths.Get( "TempColorMap" ) );

    if ( boost::filesystem::exists( tempIdTileIndexDBPath ) )
    {
        boost::filesystem::remove( tempIdTileIndexDBPath );
    }
    else
    {
        boost::filesystem::create_directories( tempIdTileIndexDBPath.parent_path() );
    }

    if ( boost::filesystem::exists( tempColorMapPath ) )
    {
        boost::filesystem::remove( tempColorMapPath );
    }
    else
    {
        boost::filesystem::create_directories( tempColorMapPath.parent_path() );
    }

    boost::filesystem::copy_file( idTileIndexDBPath, tempIdTileIndexDBPath );
    boost::filesystem::copy_file( colorMapPath, tempColorMapPath );

    //
    // Now we load up a SegmentInfoManager using the DB in the Temp Location.
    //
    mSegmentInfoManager = FileSystemSegmentInfoManager( mSegmentationTiledDatasetDescription.paths.Get( "TempColorMap" ), mSegmentationTiledDatasetDescription.paths.Get( "TempSegmentInfo" ) );

    mSegmentInfoManager.OpenDB();

    //
    // Start logging
    //
    mLogger.OpenLog( mSegmentationTiledDatasetDescription.paths.Get( "Log" ) );
    mLogger.Log("Segmentation Loaded.");

    //
    // Set internal state
    //
    mSegmentationTiledDatasetDescription.maxLabelId = mSegmentInfoManager.GetMaxId();
    mIsSegmentationLoaded                           = true;
}

void FileSystemTileServer::UnloadSegmentation()
{
    mIsSegmentationLoaded = false;

    mLogger.Log("Segmentation Unloaded.");
    mLogger.CloseLog();

    mSegmentInfoManager.CloseDB();

    DeleteTempFiles();
    ClearFileSystemTileCache();

    mSegmentInfoManager                  = FileSystemSegmentInfoManager();
    mSegmentationTiledDatasetDescription = TiledDatasetDescription();
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

    Printf( "Saving tiles (temp)." );

    TempSaveFileSystemTileCacheChanges();

    //
    // move changed tiles to the save directory
    //

    Printf( "Replacing tiles." );

    Int4 numTiles = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

    for ( int w = 0; w < numTiles.w; w++ )
    {
        for ( int z = 0; z < numTiles.z; z++ )
        {
            for ( int y = 0; y < numTiles.y; y++ )
            {
                for ( int x = 0; x < numTiles.x; x++ )
                {
                    std::string tileSubString = ToString(
                        "w=", ToStringZeroPad( w, 8 ), "\\",
                        "z=", ToStringZeroPad( z, 8 ), "\\",
                        "y=", ToStringZeroPad( y, 8 ), ",",
                        "x=", ToStringZeroPad( x, 8 ), ".",
                        mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).fileExtension );

                    std::string tempTilePathString       = ToString( mSegmentationTiledDatasetDescription.paths.Get( "TempIdMap" ), "\\", tileSubString );
                    boost::filesystem::path tempTilePath = boost::filesystem::path( tempTilePathString );

                    if ( boost::filesystem::exists( tempTilePath ) )
                    {
                        std::string saveTilePathString       = ToString( mSegmentationTiledDatasetDescription.paths.Get( "IdMap" ), "\\", tileSubString );
                        boost::filesystem::path saveTilePath = boost::filesystem::path( saveTilePathString );

                        boost::filesystem::remove( saveTilePath );
                        boost::filesystem::rename( tempTilePath, saveTilePath );
                    }
                }
            }
        }
    }

    // 
    // Note that because the DB actually resides in a temp location, we use mSegmentInfoManager.SaveAs(...) to copy the temp DB to the non-temp location.
    //
    Printf( "Saving idInfo and idTileIndex." );
    mSegmentInfoManager.Save();

    mSegmentInfoManager.SaveAs( mSegmentationTiledDatasetDescription.paths.Get( "ColorMap" ), mSegmentationTiledDatasetDescription.paths.Get( "SegmentInfo" ) );

    //
    // Successfully completed
    //
    Printf( "Segmentation saved." );

    mLogger.Log( "Segmentation saved." );

}

void FileSystemTileServer::SaveSegmentationAs( std::string savePath )
{
    //
    // save any tile changes (to temp directory)
    //

    Printf( "Saving tiles (temp)." );

    TempSaveFileSystemTileCacheChanges();

    //
    // move changed tiles to the save directory
    //

    Printf( "Copying tiles to new folder."  );

    Int4 numTiles = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

    for ( int w = 0; w < numTiles.w; w++ )
    {
        for ( int z = 0; z < numTiles.z; z++ )
        {
            for ( int y = 0; y < numTiles.y; y++ )
            {
                for ( int x = 0; x < numTiles.x; x++ )
                {
                    std::string tileSubString = ToString(
                        "w=", ToStringZeroPad( w, 8 ), "\\",
                        "z=", ToStringZeroPad( z, 8 ), "\\",
                        "y=", ToStringZeroPad( y, 8 ), ",",
                        "x=", ToStringZeroPad( x, 8 ), ".",
                        mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).fileExtension );

                    std::string tempTilePathString = ToString( mSegmentationTiledDatasetDescription.paths.Get( "TempIdMap" ), "\\", tileSubString );
                    std::string idTilePathString   = ToString( mSegmentationTiledDatasetDescription.paths.Get( "IdMap" ),     "\\", tileSubString );
                    std::string saveTilePathString = ToString( savePath,                                                      "\\", mSegmentationTiledDatasetDescription.paths.Get( "IdMapRelativePath" ), "\\", tileSubString );

                    boost::filesystem::path tempTilePath = boost::filesystem::path( tempTilePathString );
                    boost::filesystem::path idTilePath   = boost::filesystem::path( idTilePathString );
                    boost::filesystem::path saveTilePath = boost::filesystem::path( saveTilePathString );

                    if ( boost::filesystem::exists( saveTilePath ) )
                    {
                        boost::filesystem::remove( saveTilePath );
                    }
                    else
                    {
                        boost::filesystem::create_directories( saveTilePath.parent_path() );
                    }

                    if ( boost::filesystem::exists( tempTilePath ) )
                    {
                        boost::filesystem::copy_file( tempTilePath, saveTilePath );
                    }
                    else
                    {
                        if( boost::filesystem::exists( idTilePath ) )
                        {
                            boost::filesystem::copy_file( idTilePath, saveTilePath );
                        }
                    }
                }
            }
        }
    }

    //
    // move the idIndex file
    //
    
    Printf( "Saving idInfo and idTileIndex." );
    mSegmentInfoManager.SaveAs(
        ToString( savePath, "\\", mSegmentationTiledDatasetDescription.paths.Get( "ColorMapRelativePath" ) ),
        ToString( savePath, "\\", mSegmentationTiledDatasetDescription.paths.Get( "SegmentInfoRelativePath" ) ) );

    //
    // copy the tiledVolumeDescription file
    //

    std::string idMapTiledVolumeDescriptionPathString     = mSegmentationTiledDatasetDescription.paths.Get( "IdMapTiledVolumeDescription" );
    std::string saveIdMapTiledVolumeDescriptionPathString = ToString( savePath, "\\", mSegmentationTiledDatasetDescription.paths.Get( "IdMapTiledVolumeDescriptionRelativePath" ) );

    boost::filesystem::path idMapTiledVolumeDescriptionPath     = boost::filesystem::path( idMapTiledVolumeDescriptionPathString );
    boost::filesystem::path saveIdMapTiledVolumeDescriptionPath = boost::filesystem::path( saveIdMapTiledVolumeDescriptionPathString );

    if ( boost::filesystem::exists( saveIdMapTiledVolumeDescriptionPath ) )
    {
        boost::filesystem::remove( saveIdMapTiledVolumeDescriptionPath );
    }
    else
    {
        boost::filesystem::create_directories( saveIdMapTiledVolumeDescriptionPath.parent_path() );
    }

    boost::filesystem::copy_file( idMapTiledVolumeDescriptionPath, saveIdMapTiledVolumeDescriptionPath );

    //
    // copy the log file
    //

    std::string logPathString     = mSegmentationTiledDatasetDescription.paths.Get( "Log" );
    std::string saveLogPathString = ToString( savePath, "\\", mSegmentationTiledDatasetDescription.paths.Get( "LogRelativePath" ) );

    boost::filesystem::path logPath     = boost::filesystem::path( logPathString );
    boost::filesystem::path saveLogPath = boost::filesystem::path( saveLogPathString );

    if ( boost::filesystem::exists( saveLogPath ) )
    {
        boost::filesystem::remove( saveLogPath );
    }
    else
    {
        boost::filesystem::create_directories( saveLogPath.parent_path() );
    }

    boost::filesystem::copy_file( logPath, saveLogPath );

    //
    // create mojoseg file
    //
    boost::filesystem::path saveAbsPath       = boost::filesystem::path( savePath );
    boost::filesystem::path saveParentPath    = saveAbsPath.parent_path();
    boost::filesystem::path saveLeaf          = saveAbsPath.leaf();    
    size_t                  rindex            = saveLeaf.native_directory_string().rfind( mSegmentationTiledDatasetDescription.paths.Get( "SegmentationRootSuffix" ) );
    boost::filesystem::path mojosegPath       = saveParentPath / boost::filesystem::path( saveLeaf.native_directory_string().substr( 0, rindex ) +
                                                "." +
                                                mSegmentationTiledDatasetDescription.paths.Get( "SegmentationFileExtension" ) );
    std::string             mojosegPathString = mojosegPath.native_file_string();

    std::ofstream mojosegFile;
    mojosegFile.open( mojosegPathString );
    mojosegFile << boost::to_upper_copy( mSegmentationTiledDatasetDescription.paths.Get( "SegmentationFileExtension" ) );
    mojosegFile.close();

    //
    // finished successfully
    //

    Printf( "Segmentation saved to: \"", savePath, "\"" );

    mLogger.Log( ToString( "Segmentation saved to: \"", savePath, "\"" ) );
}

void FileSystemTileServer::AutosaveSegmentation()
{
    RELEASE_ASSERT( 0 && "Mojo::Native::FileSystemTileServer::AutosaveSegmentation() is currently unsupported" );
}

void FileSystemTileServer::DeleteTempFiles()
{
    //
    // Delete tile files
    //

    Printf( "Deleting temp files." );

    //
    // Note that boost::remove_all() throws an exception in the case that the user is browsing in the directory being deleted
    // in Windows Explorer. Through repeated calls, we manage to delete all the folders we're trying to get rid of. Windows
    // Explorer eventually gets the hint and allows the folders to be deleted, and returns the user to the parent folder
    // of the top-level folder being deleted.
    //
    for ( int numAttempts = 0; numAttempts < 3; numAttempts++ )
    {
        try
        {
            if ( boost::filesystem::exists( mSegmentationTiledDatasetDescription.paths.Get( "TempRoot" ) ) )
            {
                boost::filesystem::remove_all( mSegmentationTiledDatasetDescription.paths.Get( "TempRoot" ) );
            }
        }
        catch ( std::exception e )
        {
            Printf( "Exception thrown when deleting temp files, which is expected if you were browsing through the Mojo temp files in Windows Explorer." );
        }
    }

    //
    // If the folder is still there after all our attempts to delete it, call once more to throw the exception.
    //
    if ( boost::filesystem::exists( mSegmentationTiledDatasetDescription.paths.Get( "TempRoot" ) ) )
    {
        boost::filesystem::remove_all( mSegmentationTiledDatasetDescription.paths.Get( "TempRoot" ) );
    }
}

int FileSystemTileServer::GetTileCountForId( unsigned int segId )
{
    return (int) mSegmentInfoManager.GetTileCount( segId );
}

Int3 FileSystemTileServer::GetSegmentCentralTileLocation( unsigned int segId )
{
    if ( mIsSegmentationLoaded )
    {
        Int4 numTiles = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

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
            return Int3( 0, 0, 0 );
        }

        return Int3( minTileX + ( ( maxTileX - minTileX ) / 2 ), minTileY + ( ( maxTileY - minTileY ) / 2 ), minTileZ + ( ( maxTileZ - minTileZ ) / 2 ) );

    }

    return Int3( 0, 0, 0 );

}

Int4 FileSystemTileServer::GetSegmentZTileBounds( unsigned int segId, int zIndex )
{
    if ( mIsSegmentationLoaded )
    {
        Int4 numTiles = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;

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
            return Int4( 0, 0, 0, 0 );
        }

        return Int4( minTileX, minTileY, maxTileX, maxTileY );

    }

    return Int4( 0, 0, 0, 0 );

}


//
// Edit Methods
//

void FileSystemTileServer::RemapSegmentLabels( std::set< unsigned int > fromSegIds, unsigned int toSegId )
{
    if ( mIsSegmentationLoaded && toSegId != 0 && mSegmentInfoManager.GetConfidence( toSegId ) < 100 )
    {
        Printf( "\nRemapping ", (int)fromSegIds.size(), " segmentation labels to segmentation label ", toSegId, "...\n" );
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
                //Printf( "\nRemapping segmentation label ", fromSegId, " to segmentation label ", toSegId, "...\n" );

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

                //Printf( "Finished remapping from segmentation label ", fromSegId, " to segmentation label ", toSegId, ".\n" );
            }
        }
        Printf( "Finished remapping to segmentation label ", toSegId, ".\n" );

        mLogger.Log( ToString( "RemapSegmentLabels: fromId=", fromSegIds, " toId=", toSegId ) );

    }
}

void FileSystemTileServer::ReplaceSegmentationLabel( unsigned int oldId, unsigned int newId )
{
    if ( oldId != newId && mIsSegmentationLoaded && mSegmentInfoManager.GetConfidence( newId ) < 100 && mSegmentInfoManager.GetConfidence( oldId ) < 100 )
    {
        long voxelChangeCount = 0;

        Printf( "\nReplacing segmentation label ", oldId, " with segmentation label ", newId, "...\n" );

        Int4 numTiles = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;
        Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

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

            Int4 tileIndex = *tileIndexi;
            if ( tileIndex.w == 0 )
            {
                //
                // load tile
                //
                HashMap< std::string, VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
                unsigned int* currentIdVolume = (unsigned int*)volumeDescriptions.Get( "IdMap" ).data;

                //
                // Get or create the change record for this tile
                //
                TileChangeIdMap *tileChange = &mNextUndoItem->tileChangeMap[ tileIndex ];

                //
                // replace the old id and color with the new id and color...
                //
                Int3 numVoxels = volumeDescriptions.Get( "IdMap" ).numVoxels;
                for ( int zv = 0; zv < numVoxels.z; zv++ )
                {
                    for ( int yv = 0; yv < numVoxels.y; yv++ )
                    {
                        for ( int xv = 0; xv < numVoxels.x; xv++ )
                        {
                            Int3 index3D = Int3( xv, yv, zv );
                            int  index1D = Index3DToIndex1D( index3D, numVoxels );
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
                //Printf(
                //    "    Saving tile ",
                //    "w = ", ToStringZeroPad( tileIndex.w, 8 ), ", ",
                //    "z = ", ToStringZeroPad( tileIndex.z, 8 ), ", ",
                //    "y = ", ToStringZeroPad( tileIndex.y, 8 ), ", ",
                //    "x = ", ToStringZeroPad( tileIndex.x, 8 ) );

                SaveTile( tileIndex );
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
            Printf( "WARNING: Replaced all voxels belonging segment ", oldId, " but segment size is not zero (", mSegmentInfoManager.GetVoxelCount( oldId ), "). Tile index and segment database should be regenerated." );
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

        Printf( "\nFinished replacing segmentation label ", oldId, " with segmentation label ", newId, ".\n" );

        mLogger.Log( ToString( "ReplaceSegmentationLabel: oldId=", oldId, ", newId=", newId ) );

        mCurrentOperationProgress = 1;

    }
}

bool FileSystemTileServer::TileContainsId ( Int3 numVoxelsPerTile, Int3 currentIdNumVoxels, unsigned int* currentIdVolume, unsigned int segId )
{
    bool found = false;
    int maxIndex3D = Index3DToIndex1D( Int3( numVoxelsPerTile.x-1, numVoxelsPerTile.y-1, 0 ), currentIdNumVoxels );
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

void FileSystemTileServer::ReplaceSegmentationLabelCurrentSlice( unsigned int oldId, unsigned int newId, Float3 pointTileSpace )
{
    if ( oldId != newId && mIsSegmentationLoaded && mSegmentInfoManager.GetConfidence( newId ) < 100 && mSegmentInfoManager.GetConfidence( oldId ) < 100 )
    {
        long voxelChangeCount = 0;

        TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        Int3 numVoxels = tiledVolumeDescription.numVoxels;
        Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
        Int4 numTiles = tiledVolumeDescription.numTiles;

        Int3 pVoxelSpace = 
            Int3 (
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

        Printf( "\nReplacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, "...\n" );

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( oldId );
        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

        HashMap< std::string, VolumeDescription > volumeDescriptions;

        PrepForNextUndoRedoChange();
        mNextUndoItem->newId = newId;

        TileChangeIdMap *tileChange;

        Int4 previousTileIndex;
        bool tileLoaded = false;
        bool tileChanged = false;

        int currentW = 0;
        std::queue< Int4 > tileQueue;
        std::multimap< Int4, Int4, Int4Comparator > sliceQueue;
        //std::queue< Int4 > wQueue;

        unsigned int* currentIdVolume;
        Int3 currentIdNumVoxels;
        Int4 thisVoxel;

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

        tileQueue.push( Int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

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
                Int4 tileIndex = Int4( thisVoxel.x / numVoxelsPerTile.x,
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
                            SaveTile( previousTileIndex );
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


                Int3 index3D = Int3( tileX, tileY, 0 );
                int  index1D = Index3DToIndex1D( index3D, currentIdNumVoxels );
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
                                tileQueue.push( Int4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< Int4, Int4 > (
                                    Int4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    Int4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.x < numVoxels.x - 1)
                        {
                            if (tileX < numVoxelsPerTile.x - 1)
                            {
                                tileQueue.push( Int4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< Int4, Int4 > (
                                    Int4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    Int4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }
                        if (thisVoxel.y > 0)
                        {
                            if (tileY > 0)
                            {
                                tileQueue.push( Int4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< Int4, Int4 > (
                                    Int4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                                    Int4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW ) ) );
                            }
                        }
                        if (thisVoxel.y < numVoxels.y - 1)
                        {
                            if (tileY < numVoxelsPerTile.y - 1)
                            {
                                tileQueue.push( Int4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< Int4, Int4 > (
                                    Int4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                                    Int4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW ) ) );
                            }
                        }
                    }

                    //
                    // Add a scaled-down w to the queue
                    //
                    //if (currentW < numTiles.w-1) wQueue.push( Int4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
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
                SaveTile( previousTileIndex );
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

        Printf( "\nFinished replacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " for zslice ", pVoxelSpace.z, ".\n" );

        mLogger.Log( ToString( "ReplaceSegmentationLabelCurrentSlice: oldId=", oldId, ", newId=", newId, ", x=", pVoxelSpace.x, ", y=", pVoxelSpace.y, ", z=", pVoxelSpace.z ) );

        mCurrentOperationProgress = 1;

    }
}

void FileSystemTileServer::ReplaceSegmentationLabelCurrentConnectedComponent( unsigned int oldId, unsigned int newId, Float3 pointTileSpace )
{
    if ( oldId != newId && mIsSegmentationLoaded && mSegmentInfoManager.GetConfidence( newId ) < 100 && mSegmentInfoManager.GetConfidence( oldId ) < 100 )
    {
        long voxelChangeCount = 0;

        TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        Int3 numVoxels = tiledVolumeDescription.numVoxels;
        Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
        Int4 numTiles = tiledVolumeDescription.numTiles;
        Int3 pVoxelSpace = 
            Int3(
            (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
            (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
            (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

        Printf( "\nReplacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " in 3D from zslice ", pVoxelSpace.z, "...\n" );

        FileSystemTileSet tilesContainingOldId = mSegmentInfoManager.GetTiles( oldId );
        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( newId );

        HashMap< std::string, VolumeDescription > volumeDescriptions;

        PrepForNextUndoRedoChange();
        mNextUndoItem->newId = newId;
        
        TileChangeIdMap *tileChange;

        Int4 previousTileIndex;
        bool tileLoaded = false;
        bool tileChanged = false;

        int currentW = 0;
        std::queue< Int4 > tileQueue;
        std::multimap< Int4, Int4, Int4Comparator > sliceQueue;
        //std::queue< Int4 > wQueue;

        unsigned int* currentIdVolume;
        Int3 currentIdNumVoxels;
        Int4 thisVoxel;

        int maxProgress = 0;
        int currentProgress = 0;

        //
        // Determine the (approximate) max amount of work to be done
        //
        maxProgress = (int)tilesContainingOldId.size();

        tileQueue.push( Int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

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
                Int4 tileIndex = Int4( thisVoxel.x / numVoxelsPerTile.x,
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
                            SaveTile( previousTileIndex );
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

                Int3 index3D = Int3( tileX, tileY, 0 );
                int  index1D = Index3DToIndex1D( index3D, currentIdNumVoxels );
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
                                tileQueue.push( Int4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< Int4, Int4 > (
                                    Int4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    Int4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.x < numVoxels.x - 1)
                        {
                            if (tileX < numVoxelsPerTile.x - 1)
                            {
                                tileQueue.push( Int4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< Int4, Int4 > (
                                    Int4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                    Int4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.y > 0)
                        {
                            if (tileY > 0)
                            {
                                tileQueue.push( Int4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< Int4, Int4 > (
                                    Int4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                                    Int4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.y < numVoxels.y - 1)
                        {
                            if (tileY < numVoxelsPerTile.y - 1)
                            {
                                tileQueue.push( Int4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW) );
                            }
                            else
                            {
                                sliceQueue.insert( std::pair< Int4, Int4 > (
                                    Int4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                                    Int4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, currentW ) ) );
                            }
                        }

                        if (thisVoxel.z > 0)
                        {
                            sliceQueue.insert( std::pair< Int4, Int4 > (
                                Int4( tileIndex.x, tileIndex.y, tileIndex.z - 1, tileIndex.w ),
                                Int4( thisVoxel.x, thisVoxel.y, thisVoxel.z - 1, currentW ) ) );
                        }

                        if (thisVoxel.z < numVoxels.z - 1)
                        {
                            sliceQueue.insert( std::pair< Int4, Int4 > (
                                Int4( tileIndex.x, tileIndex.y, tileIndex.z + 1, tileIndex.w ),
                                Int4( thisVoxel.x, thisVoxel.y, thisVoxel.z + 1, currentW ) ) );
                        }
                    }

                    //
                    // Add a scaled-down w to the queue
                    //
                    //if (currentW < numTiles.w-1) wQueue.push( Int4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
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
                SaveTile( previousTileIndex );
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

        Printf( "\nFinished replacing segmentation label ", oldId, " conencted to voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") with segmentation label ", newId, " in 3D from zslice ", pVoxelSpace.z, ".\n" );

        mLogger.Log( ToString( "ReplaceSegmentationLabelCurrentConnectedComponent: oldId=", oldId, ", newId=", newId, ", x=", pVoxelSpace.x, ", y=", pVoxelSpace.y, ", z=", pVoxelSpace.z ) );

        mCurrentOperationProgress = 1;

    }
}

//
// 2D Split Methods
//


void FileSystemTileServer::DrawSplit( Float3 pointTileSpace, float radius )
{
    DrawRegionValue( pointTileSpace, radius, REGION_SPLIT );
}

void FileSystemTileServer::DrawErase( Float3 pointTileSpace, float radius )
{
    DrawRegionValue( pointTileSpace, radius, 0 );
}

void FileSystemTileServer::DrawRegionA( Float3 pointTileSpace, float radius )
{
    DrawRegionValue( pointTileSpace, radius, REGION_A );
}

void FileSystemTileServer::DrawRegionB( Float3 pointTileSpace, float radius )
{
    DrawRegionValue( pointTileSpace, radius, REGION_B );
}

void FileSystemTileServer::DrawRegionValue( Float3 pointTileSpace, float radius, int value )
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

    Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

    Int3 pVoxelSpace = 
        Int3(
        (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
        (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
        (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    int areaX = pVoxelSpace.x - mSplitWindowStart.x * numVoxelsPerTile.x;
    int areaY = pVoxelSpace.y - mSplitWindowStart.y * numVoxelsPerTile.y;
    int areaIndex = areaX + areaY * mSplitWindowWidth;

    SimpleSplitTools::ApplyCircleMask( areaIndex, mSplitWindowWidth, mSplitWindowHeight, value, radius, mSplitDrawArea );

    int irad = (int) ( radius + 0.5 );
    Int2 upperLeft = Int2( areaX - irad, areaY - irad );
    Int2 lowerRight = Int2( areaX + irad, areaY + irad );
    UpdateOverlayTilesBoundingBox( upperLeft, lowerRight );

    //Printf( "\nDrew inside bounding box (", upperLeft.x, ",", upperLeft.y, "x", lowerRight.x, ",", lowerRight.y, ").\n" );
    //Printf( "\nDrew split circle voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, "), with radius ", radius, ".\n" );
}

void FileSystemTileServer::AddSplitSource( Float3 pointTileSpace )
{
    Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

    Int3 pVoxelSpace = 
        Int3(
        (int)floor( pointTileSpace.x * numVoxelsPerTile.x ),
        (int)floor( pointTileSpace.y * numVoxelsPerTile.y ),
        (int)floor( pointTileSpace.z * numVoxelsPerTile.z ) );

    //
    // Check for duplicates
    //
    bool duplicate = false;

    for ( unsigned int si = 0; si < mSplitSourcePoints.size(); ++si )
    {
        Int3 existingPoint = mSplitSourcePoints[ si ];
        if ( existingPoint.x == pVoxelSpace.x && existingPoint.y == pVoxelSpace.y ) //ignore z
        {
            duplicate = true;
            break;
        }
    }

    if ( !duplicate )
    {
        mSplitSourcePoints.push_back( pVoxelSpace );
        Printf( "\nAdded split point at voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ").\n" );
    }

}

void FileSystemTileServer::RemoveSplitSource()
{
    mSplitSourcePoints.pop_back();
    Printf( "\nRemoved split point.\n" );
}

void FileSystemTileServer::UpdateOverlayTiles()
{
    Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

    Int2 upperLeft = Int2( 0, 0 );
    Int2 lowerRight = Int2( mSplitWindowNTiles.x * numVoxelsPerTile.x, mSplitWindowNTiles.y * numVoxelsPerTile.y );

    UpdateOverlayTilesBoundingBox( upperLeft, lowerRight );
}

void FileSystemTileServer::UpdateOverlayTilesBoundingBox( Int2 upperLeft, Int2 lowerRight )
{
    //
    // Export result to OverlayMap tiles
    //
    TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

    Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
    Int4 numTiles = tiledVolumeDescription.numTiles;

    HashMap< std::string, VolumeDescription > volumeDescriptions;

    //std::map< Int4, int, Int4Comparator > wQueue;

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
                //Printf( "Copying result tile ", tileCount, "." );

                Int4 tileIndex = Int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
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
                                //wQueue[ Int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitResultArea[ areaIndex1D ];
                            }
                            else if ( mSplitResultArea[ areaIndex1D ] == 0 && mSplitDrawArea[ areaIndex1D ] && splitData[ tileIndex1D ] != mSplitDrawArea[ areaIndex1D ] )
                            {
                                splitData[ tileIndex1D ] = mSplitDrawArea[ areaIndex1D ];
                                anyChanges = true;
                                //wQueue[ Int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitDrawArea[ areaIndex1D ];
                            }
                            else if ( splitData[ tileIndex1D ] && mSplitResultArea[ areaIndex1D ] == 0 && mSplitDrawArea[ areaIndex1D ] == 0)
                            {
                                splitData[ tileIndex1D ] = 0;
                                anyChanges = true;
                                //wQueue[ Int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = 0;
                            }
                        }
                        else if ( mSplitDrawArea != 0 )
                        {
                            if ( mSplitDrawArea[ areaIndex1D ] && splitData[ tileIndex1D ] != mSplitDrawArea[ areaIndex1D ] )
                            {
                                splitData[ tileIndex1D ] = mSplitDrawArea[ areaIndex1D ];
                                anyChanges = true;
                                //wQueue[ Int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = mSplitDrawArea[ areaIndex1D ];
                            }
                            else if ( splitData[ tileIndex1D ] && mSplitDrawArea[ areaIndex1D ] == 0)
                            {
                                splitData[ tileIndex1D ] = 0;
                                anyChanges = true;
                                //wQueue[ Int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + xOffset + tileX ) / 2, ( mSplitWindowStart.y * numVoxelsPerTile.y + yOffset + tileY ) / 2, mSplitWindowStart.z, 1) ] = 0;
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

                        Int4 strideTileIndex = Int4( tileIndex.x / stride, tileIndex.y / stride, tileIndex.z, wIndex );

                        int strideOffsetX = ( tileIndex.x % stride ) * ( numVoxelsPerTile.x / stride );
                        int strideOffsetY = ( tileIndex.y % stride ) * ( numVoxelsPerTile.y / stride );

                        HashMap< std::string, VolumeDescription > strideVolumeDescriptions = LoadTile( strideTileIndex );
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

void FileSystemTileServer::StrideUpIdTileChange( Int4 numTiles, Int3 numVoxelsPerTile, Int4 tileIndex, unsigned int* data )
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

        Int4 strideTileIndex = Int4( tileIndex.x / stride, tileIndex.y / stride, tileIndex.z, wIndex );

        int strideOffsetX = ( tileIndex.x % stride ) * ( numVoxelsPerTile.x / stride );
        int strideOffsetY = ( tileIndex.y % stride ) * ( numVoxelsPerTile.y / stride );

        HashMap< std::string, VolumeDescription > strideVolumeDescriptions = LoadTile( strideTileIndex );
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

        SaveTile( strideTileIndex );
        UnloadTile( strideTileIndex );

    }
}

void FileSystemTileServer::ResetOverlayTiles()
{
    //
    // Reset the overlay layer
    //
    TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );
    Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
    Int4 numTiles = tiledVolumeDescription.numTiles;

    std::set< Int4, Int4Comparator > clearedTiles;

    HashMap< std::string, VolumeDescription > volumeDescriptions;

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
                Int4 thisTile = Int4(
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

    Printf( "Reset Split State.\n");

}

void FileSystemTileServer::ResetAdjustState()
{

    for ( int i = 0; i < mSplitWindowNPix; ++i )
    {
        mSplitDrawArea[ i ] = 0;
    }

    ResetOverlayTiles();

    Printf( "Reset Adjust State.\n");

}

void FileSystemTileServer::ResetDrawMergeState()
{

    for ( int i = 0; i < mSplitWindowNPix; ++i )
    {
        mSplitDrawArea[ i ] = 0;
    }

    ResetOverlayTiles();

    Printf( "Reset Draw Merge State.\n");

}

void FileSystemTileServer::LoadSplitDistances( unsigned int segId )
{
    //
    // Load distances
    //
    HashMap< std::string, VolumeDescription > volumeDescriptions;

    Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
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
            //Printf( "Loading distance tile ", tileCount, "." );

            Int4 tileIndex = Int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
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
    Printf( "Segment centroid at ", mCentroid.x, "x", mCentroid.y, "." );


    //Printf( "Loaded: areaCount=", areaCount );

}

void FileSystemTileServer::PrepForSplit( unsigned int segId, Float3 pointTileSpace )
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

                Printf( "\nPreparing for split of segment ", segId, " at x=", pointTileSpace.x, ", y=", pointTileSpace.y, ", z=", pointTileSpace.z, ".\n" );

                TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

                Int3 numVoxels = tiledVolumeDescription.numVoxels;
                Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
                Int4 numTiles = tiledVolumeDescription.numTiles;

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
                mSplitWindowStart = Int3( minTileX, minTileY, (int) pointTileSpace.z );
                mSplitWindowNTiles = Int3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

                //Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
                //Printf( "mSplitWindowSize=", mSplitWindowNTiles.x, ":", mSplitWindowNTiles.y, ":", mSplitWindowNTiles.z, ".\n" );

                mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowNTiles.x;
                mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowNTiles.y;
                mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

                Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

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

                Printf( "\nFinished preparing for split of segment ", segId, " at z=", pointTileSpace.z, ".\n" );

                success = true;
            }
            catch ( std::bad_alloc e )
            {
                Printf( "WARNING: std::bad_alloc Error while preparing for split - reducing tile cache size." );
                ReduceCacheSize();
            }
            catch ( ... )
            {
                Printf( "WARNING: Unexpected error while preparing for split - attempting to continue." );
                ReduceCacheSize();
            }
        }

        if ( !success )
        {
            Printf( "ERROR: Unable to prep for split - possibly out of memory." );
        }
    }

}

void FileSystemTileServer::PrepForAdjust( unsigned int segId, Float3 pointTileSpace )
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

                Printf( "\nPreparing for adjustment of segment ", segId, " at x=", pointTileSpace.x, ", y=", pointTileSpace.y, ", z=", pointTileSpace.z, ".\n" );

                TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

                Int3 numVoxels = tiledVolumeDescription.numVoxels;
                Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
                Int4 numTiles = tiledVolumeDescription.numTiles;

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
                mSplitWindowStart = Int3( minTileX, minTileY, (int) pointTileSpace.z );
                mSplitWindowNTiles = Int3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

                //Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
                //Printf( "mSplitWindowSize=", mSplitWindowNTiles.x, ":", mSplitWindowNTiles.y, ":", mSplitWindowNTiles.z, ".\n" );

                mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowNTiles.x;
                mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowNTiles.y;
                mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

                int maxBufferSize = ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * TILE_PIXELS * TILE_PIXELS;
                RELEASE_ASSERT( mSplitWindowNPix <= maxBufferSize );

                //Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

                mSplitLabelCount = 0;

                //
                // Allocate working space (if necessary)
                //

                if ( mSplitDrawArea == 0 )
                    mSplitDrawArea = new char[ maxBufferSize ];

                ResetAdjustState();

                Printf( "\nFinished preparing for adjustment of segment ", segId, " at z=", pointTileSpace.z, ".\n" );

                success = true;
            }
            catch ( std::bad_alloc e )
            {
                Printf( "WARNING: std::bad_alloc Error while preparing for adjust - reducing tile cache size." );
                ReduceCacheSize();
            }
            catch ( ... )
            {
                Printf( "WARNING: Unexpected error while preparing for adjust - attempting to continue." );
                ReduceCacheSize();
            }
        }

        if ( !success )
        {
            Printf( "ERROR: Unable to prep for adjust - possibly out of memory." );
        }
    }

}

void FileSystemTileServer::PrepForDrawMerge( Float3 pointTileSpace )
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

                Printf( "\nPreparing for draw merge at x=", pointTileSpace.x, ", y=", pointTileSpace.y, ", z=", pointTileSpace.z, ".\n" );

                TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

                Int3 numVoxels = tiledVolumeDescription.numVoxels;
                Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
                Int4 numTiles = tiledVolumeDescription.numTiles;

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
                mSplitWindowStart = Int3( minTileX, minTileY, (int) pointTileSpace.z );
                mSplitWindowNTiles = Int3( ( maxTileX - minTileX + 1 ), ( maxTileY - minTileY + 1 ), 1 );

                //Printf( "mSplitWindowStart=", mSplitWindowStart.x, ":", mSplitWindowStart.y, ":", mSplitWindowStart.z, ".\n" );
                //Printf( "mSplitWindowSize=", mSplitWindowNTiles.x, ":", mSplitWindowNTiles.y, ":", mSplitWindowNTiles.z, ".\n" );

                mSplitWindowWidth = numVoxelsPerTile.x * mSplitWindowNTiles.x;
                mSplitWindowHeight = numVoxelsPerTile.y * mSplitWindowNTiles.y;
                mSplitWindowNPix = mSplitWindowWidth * mSplitWindowHeight;

                int maxBufferSize = ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * ( SPLIT_ADJUST_BUFFER_TILE_HALO * 2 + 1 ) * TILE_PIXELS * TILE_PIXELS;
                RELEASE_ASSERT( mSplitWindowNPix <= maxBufferSize );

                //Printf( "mSplitWindowWidth=", mSplitWindowWidth, ", mSplitWindowHeight=", mSplitWindowHeight, ", nPix=", mSplitWindowNPix, ".\n" );

                mSplitLabelCount = 0;

                //
                // Allocate working space (if necessary)
                //

                if ( mSplitDrawArea == 0 )
                {
                    mSplitDrawArea = new char[ maxBufferSize ];
                    ResetDrawMergeState();
                }

                Printf( "\nFinished preparing for draw merge at z=", pointTileSpace.z, ".\n" );

                success = true;
            }
            catch ( std::bad_alloc e )
            {
                Printf( "WARNING: std::bad_alloc Error while preparing for draw merge - reducing tile cache size." );
                ReduceCacheSize();
            }
            catch ( ... )
            {
                Printf( "WARNING: Unexpected error while preparing for draw merge - attempting to continue." );
                ReduceCacheSize();
            }
        }

        if ( !success )
        {
            Printf( "ERROR: Unable to prep for draw merge - possibly out of memory." );
        }
    }

}

void FileSystemTileServer::RecordSplitState( unsigned int segId, Float3 pointTileSpace )
{
    if ( mIsSegmentationLoaded )
    {
        Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
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
                    currentState.splitLine.push_back( Float2(
                        (float) ( mSplitWindowStart.x * numVoxelsPerTile.x + areaX ) / (float) numVoxelsPerTile.x,
                        (float) ( mSplitWindowStart.y * numVoxelsPerTile.y + areaY ) / (float) numVoxelsPerTile.y ) );
                    anyLinePixels = true;
                }
                if ( mSplitDrawArea[ areaIndex1D ] != 0 )
                {
                    currentState.splitDrawPoints.push_back( std::pair< Float2, char >( Float2(
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

void FileSystemTileServer::PredictSplit( unsigned int segId, Float3 pointTileSpace, float radius )
{
    if ( mIsSegmentationLoaded )
    {
        Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
        int currentZ = (int) floor( pointTileSpace.z * numVoxelsPerTile.z );

        std::vector< Float2 > *restoreLine = NULL;

        //
        // Try to restore from stored state
        //
        if ( mSplitStates.find( currentZ ) != mSplitStates.end() )
        {
            std::vector< std::pair< Float2, char >> *restorePoints = &mSplitStates.find( currentZ )->second.splitDrawPoints;
            for ( std::vector< std::pair< Float2, char >>::iterator drawIt = restorePoints->begin(); drawIt != restorePoints->end(); ++drawIt )
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
            Printf( "Restored draw points from stored state at z=", currentZ, "." );
        }
        else if ( mPrevSplitId == segId && ( mPrevSplitZ == currentZ - 1 || mPrevSplitZ == currentZ + 1 ) )
        {
            restoreLine = &mPrevSplitLine;
            Printf( "Predicing split at z=", currentZ, ", from previous split." );
        }

        //
        // Restrict prediction to 1 z-step
        //
        //else if ( mSplitStates.find( currentZ - 1 ) != mSplitStates.end() )
        //{
        //    restoreLine = &mSplitStates.find( currentZ - 1 )->second.splitLine;
        //    Printf( "Predicting split at z=", currentZ, ", from neighbour -1." );
        //}
        //else if ( mSplitStates.find( currentZ + 1 ) != mSplitStates.end() )
        //{
        //    restoreLine = &mSplitStates.find( currentZ + 1 )->second.splitLine;
        //    Printf( "Predicting split at z=", currentZ, ", from neighbour +1." );
        //}
        //else
        //{
        //    Printf( "No neighbours for split prediction at z=", currentZ, " (prev=", mPrevSplitZ, ")." );
        //}

        if ( restoreLine != NULL )
        {
            //
            // Draw a line where the previous split was
            //
            for ( std::vector< Float2 >::iterator splitIt = restoreLine->begin(); splitIt != restoreLine->end(); ++splitIt )
            {
                DrawSplit( Float3( splitIt->x, splitIt->y, (float) currentZ ), radius );
            }

            //
            // Find another split line here
            //
            FindBoundaryWithinRegion2D( segId );

        }
    }
}

unsigned int FileSystemTileServer::CompletePointSplit( unsigned int segId, Float3 pointTileSpace )
{
    unsigned int newId = 0;

    if ( mIsSegmentationLoaded && mSplitSourcePoints.size() > 0 && mSegmentInfoManager.GetConfidence( segId ) < 100 )
    {
        long voxelChangeCount = 0;

        Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

        Int3 pMouseOverVoxelSpace = 
            Int3(
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
                        //Printf( "Seed found at:", seedIndex1D, "." );
                    }
                }
            }
        }

        if ( !seedFound )
        {
            Printf( "WARNING: Could not find seed point - aborting." );
            ResetSplitState();
            return 0;
        }

        //
        // Perform a 2D Flood-fill ( no changes yet, just record bits in the UndoItem )
        //

        Int3 pVoxelSpace = 
            Int3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
            seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
            mSplitWindowStart.z );

        TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        Int3 numVoxels = tiledVolumeDescription.numVoxels;
        Int4 numTiles = tiledVolumeDescription.numTiles;

        Printf( "\nSplitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ")...\n" );

        HashMap< std::string, VolumeDescription > volumeDescriptions;

        PrepForNextUndoRedoChange();

        TileChangeIdMap *tileChange;

        Int4 previousTileIndex;
        bool tileLoaded = false;

        std::queue< Int4 > tileQueue;
        std::multimap< Int4, Int4, Int4Comparator > sliceQueue;
        std::queue< Int4 > wQueue;

        unsigned int* currentIdVolume;
        Int3 currentIdNumVoxels;
        Int4 thisVoxel;

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
        
        tileQueue.push( Int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

        Printf( "Filling at w=0." );

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
                //Printf( "Found mouseover pixel - inverting." );
            }

            //
            // Find the tile for this pixel
            //
            Int4 tileIndex = Int4( thisVoxel.x / numVoxelsPerTile.x,
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

            Int3 index3D = Int3( tileX, tileY, 0 );
            int  index1D = Index3DToIndex1D( index3D, currentIdNumVoxels );
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
                        tileQueue.push( Int4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< Int4, Int4 > (
                            Int4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                            Int4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0 ) ) );
                    }
                }

                if (thisVoxel.x < numVoxels.x - 1)
                {
                    if (tileX < numVoxelsPerTile.x - 1)
                    {
                        tileQueue.push( Int4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< Int4, Int4 > (
                            Int4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                            Int4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0 ) ) );
                    }
                }
                if (thisVoxel.y > 0)
                {
                    if (tileY > 0)
                    {
                        tileQueue.push( Int4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< Int4, Int4 > (
                            Int4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                            Int4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0 ) ) );
                    }
                }
                if (thisVoxel.y < numVoxels.y - 1)
                {
                    if (tileY < numVoxelsPerTile.y - 1)
                    {
                        tileQueue.push( Int4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0) );
                    }
                    else
                    {
                        sliceQueue.insert( std::pair< Int4, Int4 > (
                            Int4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                            Int4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0 ) ) );
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

                wQueue = std::queue< Int4 >();

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
                        tileIndex = Int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
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
                    Printf( "WARNING: Could not find (inverted) seed point - aborting." );
                    ResetSplitState();
                    return 0;
                }
                else
                {
                    //Printf( "Seed found at:", seedIndex1D, "." );
                }

                //
                // Use this new seed point
                //
                pVoxelSpace = 
                    Int3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
                    seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
                    mSplitWindowStart.z );

                nPixChanged = 0;
                invert = true;

                tileQueue.push( Int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

                Printf( "Filling (invert) at w=0." );

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
        mSegmentationTiledDatasetDescription.maxLabelId = newId;

        FileSystemTileSet tilesContainingNewId;

        mNextUndoItem->newId = newId;

        bool tileChanged;

        //Printf( "Splitting (invert=", invert, ")." );

        //
        // Do the split and fill up in w
        //

        std::swap(tileQueue, wQueue);
        int currentW = 0;

        tileLoaded = false;
        tileChanged = false;

        //while ( currentW < numTiles.w )
        //{
            Printf( "Splitting at w=", currentW, "." );
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
                Int4 tileIndex = Int4( thisVoxel.x / numVoxelsPerTile.x,
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
                            SaveTile( previousTileIndex );
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

                Int3 index3D = Int3( tileX, tileY, 0 );
                int  index1D = Index3DToIndex1D( index3D, currentIdNumVoxels );
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
                    //if (currentW < numTiles.w-1) wQueue.push( Int4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
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
                SaveTile( previousTileIndex );
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

        Printf( "\nFinished Splitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") to new segmentation label ", newId, "...\n" );

        mLogger.Log( ToString( "CompletePointSplit: segId=", segId, ", newId=", newId, ", z=", pVoxelSpace.z, ", npixels=", voxelChangeCount ) );

        mCurrentOperationProgress = 1;

    }

    //
    // Prep for more splitting
    //
    LoadSplitDistances( segId );
    ResetSplitState();

    return newId;

}

unsigned int FileSystemTileServer::CompleteDrawSplit( unsigned int segId, Float3 pointTileSpace, bool join3D, int splitStartZ )
{
    unsigned int newId = 0;

    if ( mIsSegmentationLoaded && mSplitNPerimiters > 0 && mSegmentInfoManager.GetConfidence( segId ) < 100 )
    {
        long voxelChangeCount = 0;

        RecordSplitState( segId, pointTileSpace );

        Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;
        Int3 pMouseOverVoxelSpace = 
            Int3(
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
        std::vector< std::pair< Float2, int >> newCentroids;

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
                    Printf( "WARNING: Could not find seed point - aborting." );
                    ResetSplitState();
                    return 0;
                }

                //
                // Perform a 2D Flood-fill ( no changes yet, just record bits in the UndoItem )
                //

                Int3 pVoxelSpace = 
                    Int3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
                    seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
                    mSplitWindowStart.z );

                TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

                Int3 numVoxels = tiledVolumeDescription.numVoxels;
                Int4 numTiles = tiledVolumeDescription.numTiles;

                Printf( "\nSplitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ")...\n" );

                HashMap< std::string, VolumeDescription > volumeDescriptions;

                PrepForNextUndoRedoChange();

                TileChangeIdMap *tileChange;

                Int4 previousTileIndex;
                bool tileLoaded = false;

                std::queue< Int4 > tileQueue;
                std::multimap< Int4, Int4, Int4Comparator > sliceQueue;
                std::queue< Int4 > wQueue;

                unsigned int* currentIdVolume;
                Int3 currentIdNumVoxels;
                Int4 thisVoxel;

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

                tileQueue.push( Int4( pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0 ) );

                Printf( "Filling at w=0." );

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
                        //Printf( "Found mouseover pixel - inverting." );
                    }

                    //
                    // Find the tile for this pixel
                    //
                    Int4 tileIndex = Int4( thisVoxel.x / numVoxelsPerTile.x,
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

                    Int3 index3D = Int3( tileX, tileY, 0 );
                    int  index1D = Index3DToIndex1D( index3D, currentIdNumVoxels );
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
                                    tileQueue.push( Int4(thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0) );
                                }
                                else
                                {
                                    sliceQueue.insert( std::pair< Int4, Int4 > (
                                        Int4( tileIndex.x - 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                        Int4( thisVoxel.x-1, thisVoxel.y, thisVoxel.z, 0 ) ) );
                                }
                            }

                            if (thisVoxel.x < numVoxels.x - 1)
                            {
                                if (tileX < numVoxelsPerTile.x - 1)
                                {
                                    tileQueue.push( Int4(thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0) );
                                }
                                else
                                {
                                    sliceQueue.insert( std::pair< Int4, Int4 > (
                                        Int4( tileIndex.x + 1, tileIndex.y, tileIndex.z, tileIndex.w ),
                                        Int4( thisVoxel.x+1, thisVoxel.y, thisVoxel.z, 0 ) ) );
                                }
                            }
                            if (thisVoxel.y > 0)
                            {
                                if (tileY > 0)
                                {
                                    tileQueue.push( Int4(thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0) );
                                }
                                else
                                {
                                    sliceQueue.insert( std::pair< Int4, Int4 > (
                                        Int4( tileIndex.x, tileIndex.y - 1, tileIndex.z, tileIndex.w ),
                                        Int4( thisVoxel.x, thisVoxel.y-1, thisVoxel.z, 0 ) ) );
                                }
                            }
                            if (thisVoxel.y < numVoxels.y - 1)
                            {
                                if (tileY < numVoxelsPerTile.y - 1)
                                {
                                    tileQueue.push( Int4(thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0) );
                                }
                                else
                                {
                                    sliceQueue.insert( std::pair< Int4, Int4 > (
                                        Int4( tileIndex.x, tileIndex.y + 1, tileIndex.z, tileIndex.w ),
                                        Int4( thisVoxel.x, thisVoxel.y+1, thisVoxel.z, 0 ) ) );
                                }
                            }
                        }
                    }

                    if ( tileQueue.size() == 0 && sliceQueue.size() == 0 && nPixChanged == 0 )
                    {
                        Printf( "WARNING: No fill pixels found for draw split - attempting to find next perimiter pixel." );
                    }
                    else if ( tileQueue.size() == 0 && sliceQueue.size() == 0 )
                    {
                        //
                        // Split is complete
                        //
                        centerX = centerX / (float) nPixChanged;
                        centerY = centerY / (float) nPixChanged;
                        Printf( "Split centroid at ", centerX, "x", centerY, ".");

                        if ( !invert )
                        {
                            if ( foundMouseOverPixel )
                            {
                                //
                                // Check for inversion ( do not re-label the mouse-over segment )
                                //
                                invert = true;
                                Printf( "Inverting (mouse over).");
                            }

                            if ( invert )
                            {
                                //
                                // Invert - find an alternative fill point
                                //

                                wQueue = std::queue< Int4 >();

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
                                        tileIndex = Int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
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
                                    Printf( "WARNING: Could not find (inverted) seed point - ignoring." );
                                    mUndoDeque.pop_front();
                                    break;
                                }
                                else
                                {
                                    //Printf( "Seed found at:", seedIndex1D, "." );
                                }

                                //
                                // Use this new seed point
                                //
                                pVoxelSpace = 
                                    Int3( seedIndex1D % mSplitWindowWidth + mSplitWindowStart.x * numVoxelsPerTile.x,
                                    seedIndex1D / mSplitWindowWidth + mSplitWindowStart.y * numVoxelsPerTile.y,
                                    mSplitWindowStart.z );

                                newId = 0;
                                centerX = 0;
                                centerY = 0;
                                nPixChanged = 0;
                                invert = true;

                                tileQueue.push( Int4(pVoxelSpace.x, pVoxelSpace.y, pVoxelSpace.z, 0) );

                                Printf( "Filling (invert) at w=0." );

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
                mSegmentationTiledDatasetDescription.maxLabelId = newId;

                FileSystemTileSet tilesContainingNewId;

                mNextUndoItem->newId = newId;

                bool tileChanged;

                //Printf( "Splitting (invert=", invert, ")." );

                //
                // Do the split and fill up in w
                //

                std::swap(tileQueue, wQueue);
                int currentW = 0;

                tileLoaded = false;
                tileChanged = false;

                //while ( currentW < numTiles.w )
                //{
                    Printf( "Splitting at w=", currentW, "." );
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
                        Int4 tileIndex = Int4( thisVoxel.x / numVoxelsPerTile.x,
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
                                    SaveTile( previousTileIndex );
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

                        Int3 index3D = Int3( tileX, tileY, 0 );
                        int  index1D = Index3DToIndex1D( index3D, currentIdNumVoxels );
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
                            //if (currentW < numTiles.w-1) wQueue.push( Int4(thisVoxel.x / 2, thisVoxel.y / 2, thisVoxel.z, currentW+1) );
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
                        SaveTile( previousTileIndex );
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
                    newCentroids.push_back( std::pair< Float2, int >( Float2( centerX, centerY ), newId ));

                    //
                    // Recalculate centroid of unchanged label
                    //
                    mSplitLabelCount -= voxelChangeCount;
                    mCentroid.x += ( mCentroid.x - centerX ) * ( (float)voxelChangeCount / (float)mSplitLabelCount );
                    mCentroid.y += ( mCentroid.y - centerY ) * ( (float)voxelChangeCount / (float)mSplitLabelCount );
                    Printf( "Remaining centroid at ", mCentroid.x, "x", mCentroid.y, "." );

                    Printf( "\nFinished Splitting segmentation label ", segId, " from voxel (x", pVoxelSpace.x, ", y", pVoxelSpace.y, " z", pVoxelSpace.z, ") to new segmentation label ", newId, "...\n" );

                    mLogger.Log( ToString( "CompleteDrawSplit: segId=", segId, ", newId=", newId, ", z=", pVoxelSpace.z, ", npixels=", voxelChangeCount ) );

                    mCurrentOperationProgress = 1;

                }

            }

            if ( mSplitLabelCount > 0 )
            {
                newCentroids.push_back( std::pair< Float2, int>( mCentroid, segId ) );
            }

            if ( join3D && segId == mPrevSplitId && ( mPrevSplitZ == mSplitWindowStart.z - 1 || mPrevSplitZ == mSplitWindowStart.z + 1 ) )
            {

                 //
                // Check for 3D links
                //

                Printf( "Checking for 3D links: ", (unsigned int) newCentroids.size(), " segments, ", (unsigned int) mPrevSplitCentroids.size(), " neighbours." );

                int closestId = -1;
                std::vector< std::pair< Float2, int >>::iterator closestIt;
                std::map< int, std::pair< int, float > > matches;

                for ( std::vector< std::pair< Float2, int >>::iterator prevCentroidIt = mPrevSplitCentroids.begin(); prevCentroidIt != mPrevSplitCentroids.end(); ++prevCentroidIt )
                {
                    float minDist = -1;
                    for ( std::vector< std::pair< Float2, int >>::iterator newCentroidIt = newCentroids.begin(); newCentroidIt != newCentroids.end(); ++newCentroidIt )
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
                        Printf( "Found centroid match at distance ", dist, " previd=", prevId, " newid=", thisId, ".");
                        ReplaceSegmentationLabel( thisId, prevId );

                        //
                        // Update centroid info
                        //
                        for ( std::vector< std::pair< Float2, int >>::iterator newCentroidIt = newCentroids.begin(); newCentroidIt != newCentroids.end(); ++newCentroidIt )
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
                        Printf( "WARNING: 3d join attempt to remerge back to original id.");
                        Printf( "Found centroid match at distance ", dist, " previd=", prevId, " newid=", thisId, " (not merged).");
                    }
                    else if ( thisId == segId && prevId != segId )
                    {
                        //
                        // Possible re-merge
                        //
                        Printf( "WARNING: 3d join attempt merge original id to new segment - possible mouse over conflict.");
                        Printf( "Found centroid match at distance ", dist, " previd=", prevId, " newid=", thisId, " (not merged).");
                    }
                    else
                    {
                        //
                        // Both segId
                        //
                        Printf( "Found centroid match at distance ", dist, " previd=", prevId, " newid=", thisId, " (nothing to do).");
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

void FileSystemTileServer::CommitAdjustChange( unsigned int segId, Float3 pointTileSpace )
{
    if ( mIsSegmentationLoaded )
    {
        long voxelChangeCount = 0;
        std::map< unsigned int, long > idChangeCounts;

        HashMap< std::string, VolumeDescription > volumeDescriptions;

        TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
        Int4 numTiles = tiledVolumeDescription.numTiles;
        unsigned int* currentIdVolume;
        int nVoxels = numVoxelsPerTile.x * numVoxelsPerTile.y;

        FileSystemTileSet tilesContainingNewId = mSegmentInfoManager.GetTiles( segId );

        PrepForNextUndoRedoChange();
        mNextUndoItem->newId = segId;

        TileChangeIdMap *tileChange;

        int currentW = 0;
        std::queue< Int4 > tileQueue;
        std::queue< Int4 > wQueue;

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
                //Printf( "Loading distance tile ", tileCount, "." );

                Int4 tileIndex = Int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
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
                        //wQueue.push( Int4( ( mSplitWindowStart.x * numVoxelsPerTile.x + areaX ) / 2,
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

                    SaveTile( tileIndex );
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

        Printf( "\nFinished Adjusting segmentation label ", segId, " in tile z=", pointTileSpace.z, ".\n" );

        mLogger.Log( ToString( "CommitAdjustChange: segId=", segId, ", z=", pointTileSpace.z, ", npixels=", voxelChangeCount ) );

    }
}

std::set< unsigned int > FileSystemTileServer::GetDrawMergeIds( Float3 pointTileSpace )
{
    std::set< unsigned int > mergeIds;

    if ( mIsSegmentationLoaded )
    {

        Printf( "\nFinding Merge Ids.\n" );

        HashMap< std::string, VolumeDescription > volumeDescriptions;

        TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
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
                //Printf( "Loading distance tile ", tileCount, "." );

                Int4 tileIndex = Int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
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

std::map< unsigned int, Float3 > FileSystemTileServer::GetDrawMergeIdsAndPoints( Float3 pointTileSpace )
{
    std::map< unsigned int, Float3 > mergeIdsAndPoints;

    if ( mIsSegmentationLoaded )
    {

        Printf( "\nFinding Merge Ids.\n" );

        HashMap< std::string, VolumeDescription > volumeDescriptions;

        TiledVolumeDescription tiledVolumeDescription = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" );

        Int3 numVoxelsPerTile = tiledVolumeDescription.numVoxelsPerTile;
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
                //Printf( "Loading distance tile ", tileCount, "." );

                Int4 tileIndex = Int4( mSplitWindowStart.x + xd, mSplitWindowStart.y + yd, mSplitWindowStart.z, 0 );
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
                            mergeIdsAndPoints[ idValue ] = Float3(
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

unsigned int FileSystemTileServer::CommitDrawMerge( std::set< unsigned int > mergeIds, Float3 pointTileSpace )
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

unsigned int FileSystemTileServer::CommitDrawMergeCurrentSlice( Float3 pointTileSpace )
{
    std::map< unsigned int, Float3 > mergeIdsAndPoints = GetDrawMergeIdsAndPoints( pointTileSpace );

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

        for ( std::map< unsigned int, Float3 >::iterator mergeIt = mergeIdsAndPoints.begin(); mergeIt != mergeIdsAndPoints.end(); ++mergeIt )
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
            for ( std::map< unsigned int, Float3 >::iterator mergeIt = mergeIdsAndPoints.begin(); mergeIt != mergeIdsAndPoints.end(); ++mergeIt )
            {
                ReplaceSegmentationLabelCurrentSlice( mergeIt->first, largestSegId, mergeIt->second );
            }
        }

    }

    ResetDrawMergeState();

    return largestSegId;

}

unsigned int FileSystemTileServer::CommitDrawMergeCurrentConnectedComponent( Float3 pointTileSpace )
{
    std::map< unsigned int, Float3 > mergeIdsAndPoints = GetDrawMergeIdsAndPoints( pointTileSpace );

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

        for ( std::map< unsigned int, Float3 >::iterator mergeIt = mergeIdsAndPoints.begin(); mergeIt != mergeIdsAndPoints.end(); ++mergeIt )
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
            for ( std::map< unsigned int, Float3 >::iterator mergeIt = mergeIdsAndPoints.begin(); mergeIt != mergeIdsAndPoints.end(); ++mergeIt )
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
        Printf( "\nFinding Split line for segment ", segId, " with ", (unsigned int) mSplitSourcePoints.size(), " split segments.\n" );

        Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

        int nSources = (int) mSplitSourcePoints.size();
        int* sourceLinks = new int[ nSources ];
        int* sourceLocations = new int[ nSources ];

        for ( int si = 0; si < nSources; ++si )
        {
            sourceLinks[ si ] = 0;
            sourceLocations[ si ] = ( mSplitSourcePoints[ si ].x - mSplitWindowStart.x * numVoxelsPerTile.x ) +
                ( mSplitSourcePoints[ si ].y - mSplitWindowStart.y * numVoxelsPerTile.y ) * mSplitWindowWidth;

            //Printf( "Split point at:", mSplitSourcePoints[ si ].x, ":", mSplitSourcePoints[ si ].y, " index=", sourceLocations[ si ], ".\n" );
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
                Printf( "WARNING: Could not find shortest path in FindBoundaryJoinPoints2D." );
                break;
            }

        }

        UpdateOverlayTiles();

        delete[] sourceLinks;
        delete[] sourceLocations;

        Printf( "\nFinished splitting label ", segId, ".\n" );

    }
}

void FileSystemTileServer::FindBoundaryWithinRegion2D( unsigned int segId )
{
    //
    // Watershed drawn region to highest boundaries
    //
    if ( mIsSegmentationLoaded )
    {
        Printf( "\nFindBoundaryWithinRegion2D: Splitting label ", segId, ".\n" );

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
                //Printf( "Added new perimiterLabel=", mSplitNPerimiters, " at ", i, "." );

                while ( perimiterSearchSet.size() > 0 )
                {
                    int index1D = *perimiterSearchSet.begin();
                    perimiterSearchSet.erase( index1D );
                    //Printf( "Searching around perimiterLabel at ", index1D, "." );

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

                        //Printf( "New boundary edge - adding neighbours." );

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
                        //Printf( "New watershed pixel." );
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

            //Printf( "Got new watershed pixel at ", index1D, ", score=", mapIt->first, "." );

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
                                //Printf( "Added watershed neighbour at ", addIndex1D, "." );
                            }
                        }
                    }
                }
            }

        }

        UpdateOverlayTiles();

        Printf( "\nFindBoundaryWithinRegion2D: Finished splitting label ", segId, ".\n" );

    }
}

void FileSystemTileServer::FindBoundaryBetweenRegions2D( unsigned int segId )
{
    //
    // Watershed between drawn regions to highest boundaries
    //
    if ( mIsSegmentationLoaded )
    {
        Printf( "\nFindBoundaryBetweenRegions2D: Splitting label ", segId, ".\n" );

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

            //Printf( "Got new watershed pixel at ", index1D, ", score=", mapIt->first, "." );

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
                                //Printf( "Added watershed neighbour at ", addIndex1D, "." );
                            }
                        }
                    }
                }
            }

        }

        UpdateOverlayTiles();

        Printf( "\nFindBoundaryBetweenRegions2D: Finished splitting label ", segId, ".\n" );

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

        Int4 numTiles = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;
        Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

        unsigned int newId = UndoItem.newId;

        if ( newId != 0 && mIsSegmentationLoaded )
        {
            if ( UndoItem.remapFromIdsAndSizes.size() > 0 )
            {
                Printf( "\nUndo operation: Unmapping ", (int) UndoItem.remapFromIdsAndSizes.size(), " segmentation labels away from ", newId, "...\n" );

                mLogger.Log( ToString( "Undo (remap): newId=", newId, " Remaps=", (int)UndoItem.remapFromIdsAndSizes.size() ) );

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
                Printf( "\nUndo operation: changing segmentation label ", newId, " back to multiple segmentation labels...\n" );

                mLogger.Log( ToString( "Undo (tile change): newId=", newId, " Tiles=", (int)UndoItem.tileChangeMap.size() ) );

                stdext::hash_map< std::string, std::set< Int2, Int2Comparator > >::iterator changeSetIt;

                for ( TileChangeMap::iterator tileChangeIt = UndoItem.tileChangeMap.begin(); tileChangeIt != UndoItem.tileChangeMap.end(); ++tileChangeIt )
                {

                    Int4 tileIndex = tileChangeIt->first;

                    if ( tileIndex.w == 0 )
                    {
                        //
                        // load tile
                        //
                        HashMap< std::string, VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
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
                        SaveTile( tileIndex );
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

            Printf( "\nUndo operation complete.\n" );

            //
            // Make this a redo item
            //
            mUndoDeque.pop_front();
            mRedoDeque.push_front( UndoItem );
            mNextUndoItem = &mUndoDeque.front();
        }
        else
        {
            Printf( "\nWarning - invalid undo item - discarding.\n" );
            mUndoDeque.pop_front();
            mNextUndoItem = &mUndoDeque.front();
        }
    }
    else
    {
        Printf( "\nWarning - invalid undo item - discarding.\n" );
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

        Int4 numTiles = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numTiles;
        Int3 numVoxelsPerTile = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( "IdMap" ).numVoxelsPerTile;

        int newId = RedoItem.newId;

        if ( newId != 0 && mIsSegmentationLoaded )
        {
            if ( RedoItem.remapFromIdsAndSizes.size() > 0 )
            {
                Printf( "\nRedo operation: Remapping ", (int) RedoItem.remapFromIdsAndSizes.size(), " segmentation labels to ", newId, "...\n" );

                mLogger.Log( ToString( "Redo (remap): newId=", newId, " Remaps=", (int)RedoItem.remapFromIdsAndSizes.size() ) );

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

                Printf( "\nRedo operation: changing multiple segmentation labels back to ", newId, "...\n" );

                mLogger.Log( ToString( "Redo (tile change): newId=", newId, " Tiles=", (int)RedoItem.tileChangeMap.size() ) );

                stdext::hash_map< std::string, std::set< Int2, Int2Comparator > >::iterator changeSetIt;

                for ( TileChangeMap::iterator tileChangeIt = RedoItem.tileChangeMap.begin(); tileChangeIt != RedoItem.tileChangeMap.end(); ++tileChangeIt )
                {

                    Int4 tileIndex = tileChangeIt->first;

                    if ( tileIndex.w == 0 )
                    {

                        //
                        // load tile
                        //
                        HashMap< std::string, VolumeDescription > volumeDescriptions = LoadTile( tileIndex );
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
                        SaveTile( tileIndex );
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

            Printf( "\nRedo operation complete.\n" );

            //
            // Make this a redo item
            //
            mRedoDeque.pop_front();
            mUndoDeque.push_front( RedoItem );
            mNextUndoItem = &mUndoDeque.front();
        }
        else
        {
            Printf( "\nWarning - invalid redo item - discarding.\n" );
            mRedoDeque.pop_front();
        }
    }
    else
    {
        Printf( "\nWarning - invalid redo item - discarding.\n" );
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

HashMap< std::string, VolumeDescription > FileSystemTileServer::LoadTile( Int4 tileIndex )
{
    std::string tileIndexString = CreateTileString( tileIndex );
    if ( mFileSystemTileCache.GetHashMap().find( tileIndexString ) != mFileSystemTileCache.GetHashMap().end() )
    {
        stdext::hash_map < std::string, FileSystemTileCacheEntry >::iterator ientry =
            mFileSystemTileCache.GetHashMap().find( tileIndexString );
        ientry->second.inUse++;
        ientry->second.timeStamp = clock();

        //Printf("Got tile from cache (inUse=", ientry->second.inUse, ").");

        return ientry->second.volumeDescriptions;
    }
    else
    {
        HashMap< std::string, VolumeDescription > volumeDescriptions;

        //
        // source map
        //
        volumeDescriptions.Set( "SourceMap", LoadTileLayerFromImage( tileIndex, mSourceImagesTiledDatasetDescription.paths.Get( "SourceMap" ), "SourceMap" ) );

        //
        // id map
        //
        if ( mIsSegmentationLoaded )
        {
            VolumeDescription idMapVolumeDescription;
            bool success = TryLoadTileLayerFromHdf5( tileIndex, mSegmentationTiledDatasetDescription.paths.Get( "TempIdMap" ), "IdMap", "IdMap", idMapVolumeDescription );

            if ( !success )
            {
                idMapVolumeDescription = LoadTileLayerFromHdf5( tileIndex, mSegmentationTiledDatasetDescription.paths.Get( "IdMap" ), "IdMap", "IdMap" );
            }

            volumeDescriptions.Set( "IdMap", idMapVolumeDescription );

            VolumeDescription overlayMapVolumeDescription;
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

        //Printf( "Got tile from disk (cache size is now ", mFileSystemTileCache.GetHashMap().size(), ")." );

        return volumeDescriptions;
    }
}


void FileSystemTileServer::UnloadTile( Int4 tileIndex )
{
    //
    // Keep the tile in the cache, but mark it as unused
    //
    std::string tileIndexString = CreateTileString( tileIndex );

    //
    // Note that the tile might have been flushed from the cache already, so it won't neccesarily be in mFileSystemTileCache. 
    //
    if ( mFileSystemTileCache.GetHashMap().find( tileIndexString ) != mFileSystemTileCache.GetHashMap().end() )
    {
        stdext::hash_map < std::string, FileSystemTileCacheEntry >::iterator ientry =
            mFileSystemTileCache.GetHashMap().find( tileIndexString );
        ientry->second.inUse--;
    }

    ReduceCacheSizeIfNecessary();

}

VolumeDescription FileSystemTileServer::LoadTileLayerFromImage( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName )
{
    VolumeDescription volumeDescription;

    bool success = TryLoadTileLayerFromImage( tileIndex, tileBasePath, tiledVolumeDescriptionName, volumeDescription );

    RELEASE_ASSERT( success );

    return volumeDescription;
}

bool FileSystemTileServer::TryLoadTileLayerFromImage( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, VolumeDescription& volumeDescription )
{
    switch ( mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).dxgiFormat )
    {
    case DXGI_FORMAT_R8_UNORM:
        return TryLoadTileLayerFromImageInternalUChar1( tileIndex, tileBasePath, tiledVolumeDescriptionName, volumeDescription );
        break;

    case DXGI_FORMAT_R8G8B8A8_UNORM:
        return TryLoadTileLayerFromImageInternalUChar4( tileIndex, tileBasePath, tiledVolumeDescriptionName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        return false;
        break;
    }
}

void FileSystemTileServer::UnloadTileLayer( VolumeDescription& volumeDescription )
{
    UnloadTileLayerInternal( volumeDescription );
}

void FileSystemTileServer::UnloadTileLayerInternal( VolumeDescription& volumeDescription )
{
    RELEASE_ASSERT( volumeDescription.data != NULL );

    delete[] volumeDescription.data;

    volumeDescription = VolumeDescription();
}

VolumeDescription FileSystemTileServer::LoadTileLayerFromHdf5( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName )
{
    VolumeDescription volumeDescription;

    bool success = TryLoadTileLayerFromHdf5( tileIndex, tileBasePath, tiledVolumeDescriptionName, hdf5InternalDatasetName, volumeDescription );

    RELEASE_ASSERT( success );

    return volumeDescription;
}

bool FileSystemTileServer::TryLoadTileLayerFromHdf5( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName, VolumeDescription& volumeDescription )
{
    switch ( mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).dxgiFormat )
    {
    case DXGI_FORMAT_R32_UINT:
        return TryLoadTileLayerFromHdf5Internal( tileIndex, tileBasePath, tiledVolumeDescriptionName, hdf5InternalDatasetName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        return false;
        break;
    }
}

void FileSystemTileServer::SaveTile( Int4 tileIndex )
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

void FileSystemTileServer::SaveTileLayerToImage( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, const VolumeDescription& volumeDescription )
{
    switch ( mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).dxgiFormat )
    {
    case DXGI_FORMAT_R8_UNORM:
        SaveTileLayerToImageInternalUChar1( tileIndex, tileBasePath, tiledVolumeDescriptionName, volumeDescription );
        break;

    case DXGI_FORMAT_R8G8B8A8_UNORM:
        SaveTileLayerToImageInternalUChar4( tileIndex, tileBasePath, tiledVolumeDescriptionName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        break;
    }
}

void FileSystemTileServer::SaveTileLayerToHdf5( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName, const VolumeDescription& volumeDescription )
{
    switch ( mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).dxgiFormat )
    {
    case DXGI_FORMAT_R32_UINT:
        return SaveTileLayerToHdf5Internal( tileIndex, tileBasePath, tiledVolumeDescriptionName, hdf5InternalDatasetName, volumeDescription );
        break;

    default:
        RELEASE_ASSERT( 0 );
        break;
    }
}


//
// Cache Methods
//

std::string FileSystemTileServer::CreateTileString( Int4 tileIndex )
{
    return ToString("W", tileIndex.w, "X", tileIndex.x, "Y", tileIndex.y, "Z", tileIndex.z);
    
    //std::string tileString = ToString(
 //       "w=", ToStringZeroPad( tileIndex.w, 8 ), ",",
 //       "z=", ToStringZeroPad( tileIndex.z, 8 ), ",",
 //       "y=", ToStringZeroPad( tileIndex.y, 8 ), ",",
 //       "x=", ToStringZeroPad( tileIndex.x, 8 ) );
    //Printf( "Made tile string:", tileString );
    //return tileString;
}

Int4 FileSystemTileServer::CreateTileIndex( std::string tileString )
{
    int w, x, y, z;
    sscanf_s( tileString.c_str(), "W%dX%dY%dZ%d", &w, &x, &y, &z );
    //sscanf_s( tileString.c_str(), "w=%dz=%dy=%dx=%d", &w, &z, &y, &x );
    return Int4( x, y, z, w );
}

void FileSystemTileServer::ReduceCacheSize()
{
    Printf("Flushing Cache...");
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
                //Printf("Saving tile ", i->first, ".");
                SaveTileLayerToHdf5( i->second.tileIndex, mSegmentationTiledDatasetDescription.paths.Get( "TempIdMap" ), "IdMap", "IdMap", i->second.volumeDescriptions.Get( "IdMap" ) );
                i->second.needsSaving = false;
            }

            if ( i->second.volumeDescriptions.GetHashMap().find( "IdMap" ) != i->second.volumeDescriptions.GetHashMap().end() )
            {
                UnloadTileLayer( i->second.volumeDescriptions.Get( "IdMap" ) );
            }

            //
            // CODE QUALITY ISSUE:
            // Asymmetrical new and delete. The array is allocated by calling LoadTileLayer, so it would improve the code symettry to deallocate by calling UnloadTileLayer. -MR
            //
            if ( i->second.volumeDescriptions.GetHashMap().find( "OverlayMap" ) != i->second.volumeDescriptions.GetHashMap().end() )
            {
                delete[] i->second.volumeDescriptions.Get( "OverlayMap" ).data;
            }

            UnloadTileLayer( i->second.volumeDescriptions.Get( "SourceMap" ) );
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
    Printf("Removed ", removed, " tiles from the cache.");
}


void FileSystemTileServer::ReduceCacheSizeIfNecessary()
{
    //
    // CODE QUALITY ISSUE:
    // Notice that if this cache was implemented as a page table with fixed-size array to store the cache entries,
    // you could avoid a lot of this complexity. -MR
    //
    if ( mFileSystemTileCache.GetHashMap().size() >= FILE_SYSTEM_TILE_CACHE_SIZE )
    {
        ReduceCacheSize();
    }

}

void FileSystemTileServer::TempSaveFileSystemTileCacheChanges()
{
    //
    // CODE QUALITY ISSUE:
    // These for loops (and others like them) are better expressed using the MOJO_FOR_EACH_KEY, MOJO_FOR_EACH_VALUE, and MOJO_FOR_EACH_KEY_VALUE macros. -MR
    //
    //MOJO_FOR_EACH_VALUE( FileSystemTileCacheEntry f, mFileSystemTileCache.GetHashMap() )
    //{
    //    if ( f.needsSaving )
    //    {
    //        SaveTileLayerToHdf5( f.tileIndex, mSegmentationTiledDatasetDescription.paths.Get( "TempIdMap" ), "IdMap", "IdMap", f.volumeDescriptions.Get( "IdMap" ) );
    //        f.needsSaving = false;
    //    }
    //}

    for( stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator i = mFileSystemTileCache.GetHashMap().begin(); i != mFileSystemTileCache.GetHashMap().end(); i++ )
    {
        //
        // Save this tile if necessary
        //
        if ( i->second.needsSaving )
        {
            //Printf("Saving tile ", i->first, ".");
            SaveTileLayerToHdf5( i->second.tileIndex, mSegmentationTiledDatasetDescription.paths.Get( "TempIdMap" ), "IdMap", "IdMap", i->second.volumeDescriptions.Get( "IdMap" ) );

            i->second.needsSaving = false;
        }
    }
}

void FileSystemTileServer::ClearFileSystemTileCache()
{
    stdext::hash_map < std::string, FileSystemTileCacheEntry > :: iterator i = mFileSystemTileCache.GetHashMap().begin();
    while( i != mFileSystemTileCache.GetHashMap().end() )
    {
        MOJO_FOR_EACH_VALUE( VolumeDescription volumeDescription, i->second.volumeDescriptions.GetHashMap() )
        {
            UnloadTileLayer( volumeDescription );
        }

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
    mLogger.Log( ToString( "LockSegmentLabel: segId=", segId ) );
}

void FileSystemTileServer::UnlockSegmentLabel( unsigned int segId )
{
    mSegmentInfoManager.UnlockSegmentLabel( segId );
    mLogger.Log( ToString( "UnlockSegmentLabel: segId=", segId ) );
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
// CODE QUALITY ISSUE:
// What linker errors? - MR
//

//
// CODE QUALITY ISSUE:
// Note that the name TryLoad... is misleading here, since there are still asserts in this function if OpenCV can't open the image.
// In other words, this code will assert, rather than return false, if OpenCV can't open the image. Also, why is there a TryLoad...
// system in place at all? Why would I want to do anything other than assert if an image doesn't load properly? Finally, note that
// at the only call site of this function, you immediately assert that the image loaded successfully anyway, thereby undoing the
// flexibility that might have been gained by having a working TryLoad... method. -MR
//
bool FileSystemTileServer::TryLoadTileLayerFromImageInternalUChar1( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, VolumeDescription& volumeDescription )
{
    std::string tilePath = ToString(
        tileBasePath, "\\",
        "w=", ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", ToStringZeroPad( tileIndex.x, 8 ), ".",
        mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        TiledVolumeDescription tiledVolumeDescription = mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName );

        volumeDescription.data             = new unsigned char[ tiledVolumeDescription.numVoxelsPerTile.y * tiledVolumeDescription.numVoxelsPerTile.x * tiledVolumeDescription.numBytesPerVoxel ];
        volumeDescription.dxgiFormat       = tiledVolumeDescription.dxgiFormat;
        volumeDescription.isSigned         = tiledVolumeDescription.isSigned;
        volumeDescription.numBytesPerVoxel = tiledVolumeDescription.numBytesPerVoxel;
        volumeDescription.numVoxels        = tiledVolumeDescription.numVoxelsPerTile;

        int flags = 0; // force greyscale
        cv::Mat tileImage = cv::imread( tilePath, flags );

        RELEASE_ASSERT( tileImage.cols        == tiledVolumeDescription.numVoxelsPerTile.x );
        RELEASE_ASSERT( tileImage.rows        == tiledVolumeDescription.numVoxelsPerTile.y );
        RELEASE_ASSERT( tileImage.elemSize()  == 1 );
        RELEASE_ASSERT( tileImage.elemSize1() == 1 );
        RELEASE_ASSERT( tileImage.channels()  == 1 );
        RELEASE_ASSERT( tileImage.type()      == CV_8UC1 );
        RELEASE_ASSERT( tileImage.depth()     == CV_8U );
        RELEASE_ASSERT( tileImage.isContinuous() );

        memcpy( volumeDescription.data, tileImage.ptr(), tiledVolumeDescription.numVoxelsPerTile.y * tiledVolumeDescription.numVoxelsPerTile.x * tiledVolumeDescription.numBytesPerVoxel );

        return true;
    }
    else
    {
        return false;
    }
}

bool FileSystemTileServer::TryLoadTileLayerFromImageInternalUChar4( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, VolumeDescription& volumeDescription )
{
    std::string tilePath = ToString(
        tileBasePath, "\\",
        "w=", ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", ToStringZeroPad( tileIndex.x, 8 ), ".",
        mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        TiledVolumeDescription tiledVolumeDescription = mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName );
        Int3 numVoxelsPerTile              = tiledVolumeDescription.numVoxelsPerTile;
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

bool FileSystemTileServer::TryLoadTileLayerFromHdf5Internal( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName, VolumeDescription& volumeDescription )
{
    std::string tilePath = ToString(
        tileBasePath, "\\",
        "w=", ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", ToStringZeroPad( tileIndex.x, 8 ), ".",
        mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).fileExtension );

    if ( boost::filesystem::exists( tilePath ) )
    {
        Int3 numVoxelsPerTile          = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).numVoxelsPerTile;
        int  numBytesPerVoxel              = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).numBytesPerVoxel;

        volumeDescription.data             = new unsigned char[ numVoxelsPerTile.y * numVoxelsPerTile.x * numBytesPerVoxel ];
        volumeDescription.dxgiFormat       = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).dxgiFormat;
        volumeDescription.isSigned         = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).isSigned;
        volumeDescription.numBytesPerVoxel = mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).numBytesPerVoxel;
        volumeDescription.numVoxels        = numVoxelsPerTile;

        //Printf( "Loading tile ", tilePath, "...");

        hid_t hdf5FileHandle = marray::hdf5::openFile( tilePath );
        marray::Marray< unsigned int > marray;
        try
        {
            marray::hdf5::load( hdf5FileHandle, hdf5InternalDatasetName, marray );
        }
        catch (...)
        {
            Printf( "Warning - error loading hdf5 tile. Attempting to reduce cache size." );
            ReduceCacheSize();
            marray::hdf5::load( hdf5FileHandle, hdf5InternalDatasetName, marray );
        }
        marray::hdf5::closeFile( hdf5FileHandle );

        //Printf( "Done.");

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

void FileSystemTileServer::SaveTileLayerToImageInternalUChar4( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, const VolumeDescription& volumeDescription )
{
    std::string tilePath = ToString(
        tileBasePath, "\\",
        "w=", ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", ToStringZeroPad( tileIndex.x, 8 ), ".",
        mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).fileExtension );

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

void FileSystemTileServer::SaveTileLayerToImageInternalUChar1( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, const VolumeDescription& volumeDescription )
{
    std::string tilePath = ToString(
        tileBasePath, "\\",
        "w=", ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", ToStringZeroPad( tileIndex.x, 8 ), ".",
        mSourceImagesTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).fileExtension );

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

void FileSystemTileServer::SaveTileLayerToHdf5Internal( Int4 tileIndex, std::string tileBasePath, std::string tiledVolumeDescriptionName, std::string hdf5InternalDatasetName, const VolumeDescription& volumeDescription )
{
    size_t shape[] = { volumeDescription.numVoxels.y, volumeDescription.numVoxels.x };
    marray::Marray< unsigned int > marray( shape, shape + 2 );

    memcpy( &marray( 0 ), volumeDescription.data, volumeDescription.numVoxels.y * volumeDescription.numVoxels.x * volumeDescription.numBytesPerVoxel );

    std::string tilePathString = ToString(
        tileBasePath, "\\",
        "w=", ToStringZeroPad( tileIndex.w, 8 ), "\\",
        "z=", ToStringZeroPad( tileIndex.z, 8 ), "\\",
        "y=", ToStringZeroPad( tileIndex.y, 8 ), ",",
        "x=", ToStringZeroPad( tileIndex.x, 8 ), ".",
        mSegmentationTiledDatasetDescription.tiledVolumeDescriptions.Get( tiledVolumeDescriptionName ).fileExtension );

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