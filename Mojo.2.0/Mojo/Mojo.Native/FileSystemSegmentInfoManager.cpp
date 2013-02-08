#include "FileSystemSegmentInfoManager.hpp"

#include "Mojo.Core/Boost.hpp"

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
        Core::Printf( "Closed SQLite database ", mIdTileIndexDBPath, "." );
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
            Core::Printf( "Error opening SQLite database: ", std::string( sqlite3_errmsg( mIdTileIndexDB ) ) );
            sqlite3_close( mIdTileIndexDB );
        }
        else
        {

			Core::Printf( "Opened SQLite database ", mIdTileIndexDBPath, "." );
            mIsDBOpen = true;

			//
			// SQLite PRAGMAS for faster operation
			//
			std::string query;
			std::stringstream converter;

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
				Core::Printf( "ERROR: Unable to execute PRAGMA statements in database (", sqlReturn, "): ", std::string( sqlError ) );
			}

			//
			// Get mIdMax
			//
            sqlite3_stmt* statement = NULL;
            converter.str("");
            converter << "SELECT MAX(id) FROM segmentInfo;";
            query = converter.str();

            sqlReturn = sqlite3_prepare_v2(mIdTileIndexDB, query.c_str(), query.size(), &statement, NULL); 

            if ( sqlReturn )
            {
                Core::Printf( "Error preparing SQLite3 query (", sqlReturn, "): ", sqlite3_errmsg( mIdTileIndexDB ) );
            }
            else
            {
                if ( sqlite3_step( statement ) == SQLITE_ROW )
                {
                    mIdMax = sqlite3_column_int(statement, 0);
                }
                Core::Printf( "Found max id of ", mIdMax, "." );
            }

            sqlite3_finalize(statement);

			//
			// Load the Segment Info
			//
            converter.str("");
            converter << "SELECT id, name, size, confidence FROM segmentInfo WHERE size > 0 ORDER BY id;";
            query = converter.str();

            sqlReturn = sqlite3_prepare_v2(mIdTileIndexDB, query.c_str(), query.size(), &statement, NULL); 

            if ( sqlReturn )
            {
                Core::Printf( "Error preparing SQLite3 query (", sqlReturn, "): ", sqlite3_errmsg( mIdTileIndexDB ) );
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
                }
                Core::Printf( "Read ", mSegmentMultiIndex.size(), " segment info entries from db." );
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

    std::string newPath = mColorMapPath + ".temp";

    boost::filesystem::path tempidInfoPath = boost::filesystem::path( newPath );
    if ( !boost::filesystem::exists( tempidInfoPath ) )
    {
        boost::filesystem::create_directories( tempidInfoPath.parent_path() );
    }

    Core::Printf("Saving idInfo (temp).");

    hid_t newIdMapsFile = marray::hdf5::createFile( newPath );
    marray::hdf5::save( newIdMapsFile, "idColorMap", mIdColorMap );
    marray::hdf5::closeFile( newIdMapsFile );

    Core::Printf( "Saving idTileIndex." );

    OpenDB();

    std::string query;
    std::stringstream converter;

    int numDeletes = 0;
    int numInserts = 0;

    SegmentMultiIndexById& idIndex = mSegmentMultiIndex.get<id>();

	converter << "BEGIN TRANSACTION;\n";

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

        SegmentInfo changedInfo = *idIndex.find( idIt->first );

        converter << "INSERT or REPLACE INTO segmentInfo (id, name, size, confidence) VALUES (" << changedInfo.id << ",\"" << changedInfo.name << "\"," << changedInfo.size << "," << changedInfo.confidence << ");\n";

    }

	converter << "END TRANSACTION;\n";

    Core::Printf( "Removing ", numDeletes, " and adding ", numInserts, " tile index entries." );

    int sqlReturn;
    char *sqlError = NULL;

    query = converter.str();
    sqlReturn = sqlite3_exec( mIdTileIndexDB, query.c_str(), NULL, NULL, &sqlError); 

    if ( sqlReturn != SQLITE_OK )
    {
        Core::Printf( "ERROR: Unable to update tile index database (", sqlReturn, "): ", std::string( sqlError ) );
    }
    else
    {
        mCacheIdTileMap.GetHashMap().clear();
    }

    Core::Printf( "Replacing idInfo file." );

    boost::filesystem::path idColorMapPath = boost::filesystem::path( mColorMapPath );

    boost::filesystem::remove( idColorMapPath );
    boost::filesystem::rename( tempidInfoPath, idColorMapPath );

}

marray::Marray< unsigned char > FileSystemSegmentInfoManager::GetIdColorMap()
{
    return mIdColorMap;
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
    std::stringstream converter;

    int w, x, y, z, sqlReturn;
    sqlite3_stmt* statement = NULL; 

    OpenDB();

    converter << "SELECT w, z, y, x FROM idTileIndex WHERE id = " << segid << ";";
    query = converter.str();

    sqlReturn = sqlite3_prepare_v2(mIdTileIndexDB, query.c_str(), query.size(), &statement, NULL); 

    if ( sqlReturn )
    {
        Core::Printf( "Error preparing SQLite3 query (", sqlReturn, "): ", sqlite3_errmsg( mIdTileIndexDB ) );
    }
    else
    {
        while ( sqlite3_step( statement ) == SQLITE_ROW )
        {
            w = sqlite3_column_int(statement, 0);
            z = sqlite3_column_int(statement, 1);
            y = sqlite3_column_int(statement, 2);
            x = sqlite3_column_int(statement, 3);

            tileSet.insert( make_int4( x, y, z, w ) );
        }
        //Core::Printf( "Read ", tileSet.size(), " entries from db for segment ", segid, "." );
    }

    sqlite3_finalize(statement); 

    return tileSet;
}

unsigned int FileSystemSegmentInfoManager::GetTileCount ( unsigned int segid )
{
    return GetTiles( segid ).size();
}

unsigned int FileSystemSegmentInfoManager::GetVoxelCount ( unsigned int segid )
{
    return mSegmentMultiIndex.get<id>().find( segid )->size;
}
                                               
unsigned int FileSystemSegmentInfoManager::GetMaxId()
{
    return mIdMax;
}

unsigned int FileSystemSegmentInfoManager::AddNewId()
{
    ++mIdMax;

    std::string name;
    std::stringstream converter;

    converter << "segment" << mIdMax;
    name = converter.str();

    mSegmentMultiIndex.push_back( SegmentInfo( mIdMax, name, 0, 0 ) );

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
    SegmentInfo segInfo = *segIt;

    segInfo.size = voxelCount;
    
    bool success = idIndex.replace( segIt, segInfo );

    if ( !success )
    {
        Core::Printf( "WARNING: Could not update voxel count for id  ", segid, " to size ", voxelCount, "." );
    }

}

void FileSystemSegmentInfoManager::SortById( bool reverse )
{
    mSegmentMultiIndex.get<0>().rearrange( mSegmentMultiIndex.get<id>().begin() );
    if ( reverse )
    {
        mSegmentMultiIndex.get<0>().reverse();
    }
}

void FileSystemSegmentInfoManager::SortByName( bool reverse )
{
    mSegmentMultiIndex.get<0>().rearrange( mSegmentMultiIndex.get<name>().begin() );
    if ( reverse )
    {
        mSegmentMultiIndex.get<0>().reverse();
    }
}

void FileSystemSegmentInfoManager::SortBySize( bool reverse )
{
    mSegmentMultiIndex.get<0>().rearrange( mSegmentMultiIndex.get<size>().begin() );
    if ( reverse )
    {
        mSegmentMultiIndex.get<0>().reverse();
    }
}

void FileSystemSegmentInfoManager::SortByConfidence( bool reverse )
{
    mSegmentMultiIndex.get<0>().rearrange( mSegmentMultiIndex.get<confidence>().begin() );
    if ( reverse )
    {
        mSegmentMultiIndex.get<0>().reverse();
    }
}

std::list< SegmentInfo > FileSystemSegmentInfoManager::GetSegmentInfoRange( unsigned int startIndex, unsigned int endIndex )
{
    std::list< SegmentInfo > segmentInfoPage;

    if ( endIndex > mSegmentMultiIndex.size() )
    {
        endIndex = mSegmentMultiIndex.size();
    }

    for ( unsigned int i = startIndex; i < endIndex; ++i )
    {
        segmentInfoPage.push_back( mSegmentMultiIndex[ i ] );
    }

    return segmentInfoPage;
}


}
}