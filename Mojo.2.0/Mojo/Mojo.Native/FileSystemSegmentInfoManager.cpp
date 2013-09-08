#include "FileSystemSegmentInfoManager.hpp"
#include "FileSystemTileServerConstants.hpp"

#include "Boost.hpp"

namespace Mojo
{
namespace Native
{

FileSystemSegmentInfoManager::FileSystemSegmentInfoManager()
{
    mIsDBOpen = false;
}

FileSystemSegmentInfoManager::FileSystemSegmentInfoManager( std::string colorMapFilePath, std::string idTileIndexDBFilePath )
{
    mColorMapPath = colorMapFilePath;
    mIdTileIndexDBPath = idTileIndexDBFilePath;
    mIsDBOpen = false;

    mColorMapHdf5FileHandle = marray::hdf5::openFile( mColorMapPath );
    marray::hdf5::load( mColorMapHdf5FileHandle, "idColorMap", mIdColorMap );
    marray::hdf5::closeFile( mColorMapHdf5FileHandle );
}

void FileSystemSegmentInfoManager::CloseDB()
{
    if ( mIsDBOpen )
    {
        sqlite3_close( mIdTileIndexDB );
        Printf( "Closed SQLite database ", mIdTileIndexDBPath, "." );
        mIsDBOpen = false;
    }
}

void FileSystemSegmentInfoManager::OpenDB()
{
    //
    // Open the SQLite database
    //
    if ( !mIsDBOpen )
    {
        int sqlReturn = sqlite3_open( mIdTileIndexDBPath.c_str(), &mIdTileIndexDB );
        if ( sqlReturn )
        {
            Printf( "Error opening SQLite database: ", std::string( sqlite3_errmsg( mIdTileIndexDB ) ) );
            sqlite3_close( mIdTileIndexDB );
        }
        else
        {

            Printf( "Opened SQLite database ", mIdTileIndexDBPath, "." );
            mIsDBOpen = true;

            //
            // SQLite PRAGMAS for faster operation
            //
            std::string query;
            std::ostringstream converter;

            converter << "PRAGMA main.cache_size=10000;\n";
            converter << "PRAGMA main.locking_mode=EXCLUSIVE;\n";
            converter << "PRAGMA main.synchronous=OFF;\n";
            converter << "PRAGMA main.journal_mode=WAL;\n";
            converter << "PRAGMA count_changes=OFF;\n";
            converter << "PRAGMA main.temp_store=MEMORY;\n";

            int sqlReturn;
            char *sqlError = NULL;

            query = converter.str();
            sqlReturn = sqlite3_exec( mIdTileIndexDB, query.c_str(), NULL, NULL, &sqlError); 

            if ( sqlReturn != SQLITE_OK )
            {
                Printf( "ERROR: Unable to execute PRAGMA statements in database (", sqlReturn, "): ", std::string( sqlError ) );
            }

            //
            // Get mIdMax
            //
            sqlite3_stmt* statement = NULL;
            converter.str("");
            converter << "SELECT MAX(id) FROM segmentInfo;";
            query = converter.str();

            sqlReturn = sqlite3_prepare_v2(mIdTileIndexDB, query.c_str(), (int)query.size(), &statement, NULL); 

            if ( sqlReturn )
            {
                Printf( "Error preparing SQLite3 query (", sqlReturn, "): ", sqlite3_errmsg( mIdTileIndexDB ) );
            }
            else
            {
                if ( sqlite3_step( statement ) == SQLITE_ROW )
                {
                    mIdMax = sqlite3_column_int(statement, 0);
                }
                Printf( "Found max id of ", mIdMax, "." );
            }

            sqlite3_finalize(statement);

            //
            // Round label id / confidence map size up to the nearest (+1) EXTRA_SEGMENTS_PER_SESSION
            //
            size_t shape[] = { ( mIdMax / EXTRA_SEGMENTS_PER_SESSION + 2 ) * EXTRA_SEGMENTS_PER_SESSION };
            unsigned char defaultConfidence = 0;
            mIdConfidenceMap = marray::Marray< unsigned char >( shape, shape + 1, defaultConfidence );

            mLabelIdMap = marray::Marray< unsigned char >( shape, shape + 1 );

            for ( unsigned int i = 0; i < shape[0]; ++i )
            {
                mLabelIdMap( i ) = i;
            }

            //
            // Load the Segment Info 
            //
            converter.str("");
            converter << "SELECT id, name, size, confidence FROM segmentInfo ORDER BY id;";
            query = converter.str();

            sqlReturn = sqlite3_prepare_v2(mIdTileIndexDB, query.c_str(), (int)query.size(), &statement, NULL); 

            if ( sqlReturn )
            {
                Printf( "Error preparing SQLite3 query (", sqlReturn, "): ", sqlite3_errmsg( mIdTileIndexDB ) );
            }
            else
            {
                while ( sqlite3_step( statement ) == SQLITE_ROW )
                {
                    mSegmentMultiIndex.push_back( SegmentInfo(
                        sqlite3_column_int(statement, 0),
                        std::string( reinterpret_cast<const char*>( sqlite3_column_text(statement, 1) ) ),
                        sqlite3_column_int(statement, 2),
                        sqlite3_column_int(statement, 3) ) );
                    mIdConfidenceMap( sqlite3_column_int(statement, 0) ) = sqlite3_column_int(statement, 3);

                    /*if ( ( sqlite3_column_int(statement, 0) ) < 500 )
                    {
                        Printf( "Segment debug: ", sqlite3_column_int(statement, 0), "=", std::string( reinterpret_cast<const char*>( sqlite3_column_text(statement, 1) ) ), "." );
                    }*/

                }
                Printf( "Read ", (int)mSegmentMultiIndex.size(), " segment info entries from db." );
            }

            sqlite3_finalize(statement);

            //
            // Load the label id map
            //
            converter.str("");
            converter << "SELECT fromId, toId FROM relabelMap WHERE fromId != toId ORDER BY fromId;";
            query = converter.str();

            sqlReturn = sqlite3_prepare_v2(mIdTileIndexDB, query.c_str(), (int)query.size(), &statement, NULL); 

            if ( sqlReturn )
            {
                Printf( "Didn't find any label remap entries in the db, which is expected when proofreading a dataset for the first time." );
            }
            else
            {
                unsigned int relabelCount = 0;
                while ( sqlite3_step( statement ) == SQLITE_ROW )
                {
                    mLabelIdMap( sqlite3_column_int(statement, 0) ) = sqlite3_column_int(statement, 1);
                    ++relabelCount;
                }
                Printf( "Read ", relabelCount, " label remap entries from db." );
            }

            sqlite3_finalize(statement);

        }

    }
}


static int callback(void *unused, int argc, char **argv, char **colName)
{
  return 0;
}


void FileSystemSegmentInfoManager::Save()
{
    SaveHelper( mColorMapPath );
}

void FileSystemSegmentInfoManager::SaveAs( std::string colorMapFilePath, std::string idTileIndexDBFilePath )
{
    SaveHelper( colorMapFilePath );

    boost::filesystem::path oldIdTileIndexDBPath = boost::filesystem::path( mIdTileIndexDBPath );
    boost::filesystem::path newIdTileIndexDBPath = boost::filesystem::path( idTileIndexDBFilePath );

    if ( boost::filesystem::exists( newIdTileIndexDBPath ) )
    {
        boost::filesystem::remove( newIdTileIndexDBPath );
    }
    else
    {
        boost::filesystem::create_directories( newIdTileIndexDBPath.parent_path() );
    }

    boost::filesystem::copy_file( oldIdTileIndexDBPath, newIdTileIndexDBPath );
}

void FileSystemSegmentInfoManager::SaveHelper( std::string colorMapFilePath )
{
    //
    // Save  colorMap.hdf5
    //

    Printf( "Saving colorMap.hdf5" );

    std::string newPath = colorMapFilePath + ".temp";

    boost::filesystem::path tempidInfoPath = boost::filesystem::path( newPath );
    if ( !boost::filesystem::exists( tempidInfoPath ) )
    {
        boost::filesystem::create_directories( tempidInfoPath.parent_path() );
    }

    hid_t newIdMapsFile = marray::hdf5::createFile( newPath );
    marray::hdf5::save( newIdMapsFile, "idColorMap", mIdColorMap );
    marray::hdf5::closeFile( newIdMapsFile );


    boost::filesystem::path idColorMapPath = boost::filesystem::path( colorMapFilePath );

    boost::filesystem::remove( idColorMapPath );
    boost::filesystem::rename( tempidInfoPath, idColorMapPath );

    //
    // Save colorMap.db
    //

    Printf( "Saving colorMap.db" );

    std::string query;
    std::ostringstream converter;

    int numDeletes = 0;
    int numInserts = 0;

    converter << "BEGIN TRANSACTION;\n";

    //
    // Update tiles map
    //

    for ( FileSystemIdTileMapDirect::iterator idIt = mCacheIdTileMap.GetHashMap().begin(); idIt != mCacheIdTileMap.GetHashMap().end(); ++idIt )
    {
        FileSystemTileSet oldTiles = LoadTileSet( idIt->first );

        for ( FileSystemTileSet::iterator oldTileIt = oldTiles.begin(); oldTileIt != oldTiles.end(); ++oldTileIt )
        {
            if ( idIt->second.find( *oldTileIt ) == idIt->second.end() )
            {
                converter << "DELETE FROM idTileIndex WHERE id = " << idIt->first << " AND w = " << oldTileIt->w << " AND z = " << oldTileIt->z << " AND y = " << oldTileIt->y << " AND x = " << oldTileIt->x << ";\n";
                ++numDeletes;
            }
        }

        for ( FileSystemTileSet::iterator newTileIt = idIt->second.begin(); newTileIt != idIt->second.end(); ++newTileIt )
        {
            if ( oldTiles.find( *newTileIt ) == oldTiles.end() )
            {
                converter << "INSERT INTO idTileIndex VALUES (" << idIt->first << "," << newTileIt->w << "," << newTileIt->z << "," << newTileIt->y << "," << newTileIt->x << ");\n";
                ++numInserts;
            }
        }

    }

    //
    // Update tile info
    //

    for ( SegmentMultiIndex::iterator infoIt = mSegmentMultiIndex.begin(); infoIt != mSegmentMultiIndex.end(); ++infoIt )
    {
        if ( infoIt->changed )
        {
            converter << "INSERT or REPLACE INTO segmentInfo (id, name, size, confidence) VALUES (" << infoIt->id << ",\"" << infoIt->name << "\"," << infoIt->size << "," << infoIt->confidence << ");\n";
        }
    }

    //
    // Update remap info
    //

    converter << "CREATE TABLE IF NOT EXISTS relabelMap ( fromId int PRIMARY KEY, toId int);\n";

    for ( unsigned int i = 0; i < mIdMax; ++i )
    {
        if ( mLabelIdMap( i ) != i )
        {
            converter << "INSERT or REPLACE INTO relabelMap (fromId, toId) VALUES (" << i << "," << mLabelIdMap( i ) << ");\n";
        }
    }

    converter << "END TRANSACTION;\n";

    Printf( "Removing ", numDeletes, " and adding ", numInserts, " tile index entries." );

    int sqlReturn;
    char *sqlError = NULL;

    query = converter.str();
    //Printf( query );
    sqlReturn = sqlite3_exec( mIdTileIndexDB, query.c_str(), NULL, NULL, &sqlError); 

    if ( sqlReturn != SQLITE_OK )
    {
        Printf( "ERROR: Unable to update tile index database (", sqlReturn, "): ", std::string( sqlError ) );
    }
    else
    {
        mCacheIdTileMap.GetHashMap().clear();
    }

    //
    // Call CloseDB and OpenDB so that changes to the database are flushed to disk.
    //
    CloseDB();
    OpenDB();
}


marray::Marray< unsigned char >* FileSystemSegmentInfoManager::GetIdColorMap()
{
    return &mIdColorMap;
}

marray::Marray< unsigned int >* FileSystemSegmentInfoManager::GetLabelIdMap()
{
    return &mLabelIdMap;
}

marray::Marray< unsigned char >* FileSystemSegmentInfoManager::GetIdConfidenceMap()
{
    return &mIdConfidenceMap;
}

FileSystemTileSet FileSystemSegmentInfoManager::GetTiles( unsigned int segid )
{
    if ( mCacheIdTileMap.GetHashMap().find( segid ) != mCacheIdTileMap.GetHashMap().end() )
    {
        return mCacheIdTileMap.Get( segid );
    }
    else
    {
        return LoadTileSet( segid );
    }
}

FileSystemTileSet FileSystemSegmentInfoManager::LoadTileSet( unsigned int segid )
{
    //
    // Load the tile id map from the SQLite DB
    //

    FileSystemTileSet tileSet;

    std::string query;
    std::ostringstream converter;

    int w, x, y, z, sqlReturn;
    sqlite3_stmt* statement = NULL; 

    OpenDB();

    converter << "SELECT w, z, y, x FROM idTileIndex WHERE id = " << segid << ";";
    query = converter.str();

    sqlReturn = sqlite3_prepare_v2(mIdTileIndexDB, query.c_str(), (int)query.size(), &statement, NULL); 

    if ( sqlReturn )
    {
        Printf( "Error preparing SQLite3 query (", sqlReturn, "): ", sqlite3_errmsg( mIdTileIndexDB ) );
    }
    else
    {
        while ( sqlite3_step( statement ) == SQLITE_ROW )
        {
            w = sqlite3_column_int(statement, 0);
            z = sqlite3_column_int(statement, 1);
            y = sqlite3_column_int(statement, 2);
            x = sqlite3_column_int(statement, 3);

            tileSet.insert( Int4( x, y, z, w ) );
        }
        //Printf( "Read ", tileSet.size(), " entries from db for segment ", segid, "." );
    }

    sqlite3_finalize(statement); 

    return tileSet;
}

unsigned int FileSystemSegmentInfoManager::GetTileCount ( unsigned int segid )
{
    return (unsigned int)GetTiles( segid ).size();
}

long FileSystemSegmentInfoManager::GetVoxelCount ( unsigned int segid )
{
    return mSegmentMultiIndex.get<id>().find( segid )->size;
}
                                               
int FileSystemSegmentInfoManager::GetConfidence ( unsigned int segid )
{
    return mSegmentMultiIndex.get<id>().find( segid )->confidence;
}
                                               
unsigned int FileSystemSegmentInfoManager::GetMaxId()
{
    return mIdMax;
}

unsigned int FileSystemSegmentInfoManager::AddNewId()
{
    ++mIdMax;

    std::string name;
    std::ostringstream converter;

    converter << "segment" << mIdMax;
    name = converter.str();

    mSegmentMultiIndex.push_back( SegmentInfo( mIdMax, name, 0, 0, true ) );

    //
    // Resize the label id map if necessary
    //
    if ( mIdMax > mLabelIdMap.shape( 0 ) )
    {
        size_t shape[] = { ( mIdMax / EXTRA_SEGMENTS_PER_SESSION + 2 ) * EXTRA_SEGMENTS_PER_SESSION };
        mLabelIdMap.resize( shape, shape + 1 );
        for ( unsigned int i = mIdMax; i < shape[0]; ++i )
        {
            mLabelIdMap( i ) = i;
        }
    }

    //
    // Resize the confidence map if necessary
    //
    if ( mIdMax > mIdConfidenceMap.shape( 0 ) )
    {
        size_t shape[] = { ( mIdMax / EXTRA_SEGMENTS_PER_SESSION + 2 ) * EXTRA_SEGMENTS_PER_SESSION };
        unsigned char defaultConfidence = 0;
        mIdConfidenceMap.resize( shape, shape + 1, defaultConfidence );
    }

    return mIdMax;
}
                                               
void FileSystemSegmentInfoManager::SetTiles( unsigned int segid, FileSystemTileSet tiles )
{
    mCacheIdTileMap.GetHashMap()[ segid ] = tiles;
}

void FileSystemSegmentInfoManager::SetVoxelCount ( unsigned int segid, long voxelCount )
{
    SegmentMultiIndexById& idIndex = mSegmentMultiIndex.get<id>();
    SegmentMultiIndexById::iterator segIt = idIndex.find( segid );

    if ( segIt == idIndex.end() )
    {
        Printf( "WARNING: Could not update voxel count for id  ", segid, " to size ", voxelCount, " - id not found in multi index." );
    }
    else
    {    
        SegmentInfo segInfo = *segIt;

        segInfo.size = voxelCount;
        segInfo.changed = true;
    
        bool success = idIndex.replace( segIt, segInfo );

        if ( !success )
        {
            Printf( "WARNING: Could not update voxel count for id  ", segid, " to size ", voxelCount, "." );
        }
    }

}

void FileSystemSegmentInfoManager::SortSegmentInfoById( bool reverse )
{
    mSegmentMultiIndex.get<0>().rearrange( mSegmentMultiIndex.get<id>().begin() );
    if ( reverse )
    {
        mSegmentMultiIndex.get<0>().reverse();
    }
}

void FileSystemSegmentInfoManager::SortSegmentInfoByName( bool reverse )
{
    mSegmentMultiIndex.get<0>().rearrange( mSegmentMultiIndex.get<name>().begin() );
    if ( reverse )
    {
        mSegmentMultiIndex.get<0>().reverse();
    }
}

void FileSystemSegmentInfoManager::SortSegmentInfoBySize( bool reverse )
{
    mSegmentMultiIndex.get<0>().rearrange( mSegmentMultiIndex.get<size>().begin() );
    if ( reverse )
    {
        mSegmentMultiIndex.get<0>().reverse();
    }
}

void FileSystemSegmentInfoManager::SortSegmentInfoByConfidence( bool reverse )
{
    mSegmentMultiIndex.get<0>().rearrange( mSegmentMultiIndex.get<confidence>().begin() );
    if ( reverse )
    {
        mSegmentMultiIndex.get<0>().reverse();
    }
}

void FileSystemSegmentInfoManager::RemapSegmentLabel( unsigned int fromSegId, unsigned int toSegId )
{
    if ( fromSegId <= mIdMax && toSegId <= mIdMax )
    {
        mLabelIdMap( fromSegId ) = toSegId;
    }
}

unsigned int FileSystemSegmentInfoManager::GetIdForLabel( unsigned int label )
{
    if ( label == mPreviousLabelQuery && mPreviousIdResult != 0 )
        return mPreviousIdResult;

    mPreviousLabelQuery = label;
    mPreviousIdResult = mLabelIdMap( label );

    while ( mPreviousIdResult != label )
    {
        label = mPreviousIdResult;
        mPreviousIdResult = mLabelIdMap( label );
    }
    
    return mPreviousIdResult;

}

void FileSystemSegmentInfoManager::LockSegmentLabel( unsigned int segId )
{
    if ( segId <= mIdMax )
    {
        mIdConfidenceMap( segId ) = 100;
        SegmentMultiIndexById::iterator segmentInfoIt = mSegmentMultiIndex.get<id>().find( segId );
        SegmentInfo changeSeg = *segmentInfoIt;
        changeSeg.confidence = 100;
        changeSeg.changed = true;
        mSegmentMultiIndex.get<id>().replace( segmentInfoIt, changeSeg );
    }
}

void FileSystemSegmentInfoManager::UnlockSegmentLabel( unsigned int segId )
{
    if ( segId <= mIdMax )
    {
        mIdConfidenceMap( segId ) = 0;
        SegmentMultiIndexById::iterator segmentInfoIt = mSegmentMultiIndex.get<id>().find( segId );
        SegmentInfo changeSeg = *segmentInfoIt;
        changeSeg.confidence = 0;
        changeSeg.changed = true;
        mSegmentMultiIndex.get<id>().replace( segmentInfoIt, changeSeg );
    }
}

unsigned int FileSystemSegmentInfoManager::GetSegmentInfoCount()
{
    return (unsigned int)mSegmentMultiIndex.size();
}

unsigned int FileSystemSegmentInfoManager::GetSegmentInfoCurrentListLocation( unsigned int segId )
{
    return (unsigned int) ( mSegmentMultiIndex.project<0> ( mSegmentMultiIndex.get<id>().find( segId ) ) - mSegmentMultiIndex.get<0>().begin() );
}

std::list< SegmentInfo > FileSystemSegmentInfoManager::GetSegmentInfoRange( unsigned int startIndex, unsigned int endIndex )
{
    std::list< SegmentInfo > segmentInfoPage;

    if ( endIndex > mSegmentMultiIndex.size() )
    {
        endIndex = (unsigned int)mSegmentMultiIndex.size();
    }

    for ( unsigned int i = startIndex; i < endIndex; ++i )
    {
        segmentInfoPage.push_back( mSegmentMultiIndex[ i ] );
    }

    return segmentInfoPage;
}

SegmentInfo FileSystemSegmentInfoManager::GetSegmentInfo( unsigned int segId )
{

    SegmentMultiIndexById& idIndex = mSegmentMultiIndex.get<id>();
    SegmentMultiIndexById::iterator segIt = idIndex.find( segId );

    if ( segIt == idIndex.end() )
    {
        Printf( "WARNING: Could not find segment for id  ", segId, " - id not found in multi index." );
        return SegmentInfo( 0, "", 0, 0, false );
    }
    else
    {
        return *segIt;
    }

}


}
}
