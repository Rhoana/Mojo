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
    marray::hdf5::load( mColorMapHdf5FileHandle, "idMax", mIdMax );
    marray::hdf5::load( mColorMapHdf5FileHandle, "idColorMap", mIdColorMap );
    marray::hdf5::load( mColorMapHdf5FileHandle, "idVoxelCountMap", mIdVoxelCountMap );
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
			converter << "PRAGMA main.temp_store=MEMORY";

			int sqlReturn;
			char *sqlError = NULL;

			query = converter.str();
			sqlReturn = sqlite3_exec( mIdTileIndexDB, query.c_str(), NULL, NULL, &sqlError); 

			if ( sqlReturn != SQLITE_OK )
			{
				Core::Printf( "ERROR: Unable to execute PRAGMA statements in database (", sqlReturn, "): ", std::string( sqlError ) );
			}

			//
			// TODO:Load the Segment Info
			//
			mSegmentMultiIndex.

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
    marray::hdf5::save( newIdMapsFile, "idMax", mIdMax );
    marray::hdf5::save( newIdMapsFile, "idColorMap", mIdColorMap );
    marray::hdf5::save( newIdMapsFile, "idVoxelCountMap", mIdVoxelCountMap );
    marray::hdf5::closeFile( newIdMapsFile );

    Core::Printf( "Saving idTileIndex." );

    OpenDB();

    std::string query;
    std::stringstream converter;

    int numDeletes = 0;
    int numInserts = 0;

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

    converter << "SELECT w, z, y, x FROM idTileIndex WHERE id = " << segid;
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
    return mIdVoxelCountMap( segid );
}
                                               
unsigned int FileSystemSegmentInfoManager::GetMaxId()
{
    return mIdMax( 0 );
}

unsigned int FileSystemSegmentInfoManager::AddNewId()
{
    mIdMax( 0 ) = mIdMax( 0 ) + 1;

    size_t shape[] = { mIdMax( 0 ) + 1 };
    mIdVoxelCountMap.resize( shape, shape + 1 );
    mIdVoxelCountMap( mIdMax(0) ) = 0;

    return mIdMax( 0 );
}
                                               
void FileSystemSegmentInfoManager::SetTiles( unsigned int segid, FileSystemTileSet tiles )
{
    mCacheIdTileMap.GetHashMap()[ segid ] = tiles;
}

void FileSystemSegmentInfoManager::SetVoxelCount ( unsigned int segid, unsigned long voxelCount )
{
    mIdVoxelCountMap( segid ) = voxelCount;
}


}
}
