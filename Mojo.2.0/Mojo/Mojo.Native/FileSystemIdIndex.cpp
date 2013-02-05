#include "FileSystemIdIndex.hpp"

#include "Mojo.Core/Boost.hpp"

namespace Mojo
{
namespace Native
{

FileSystemIdIndex::FileSystemIdIndex()
{
    mIsDBOpen = false;
}

FileSystemIdIndex::FileSystemIdIndex( std::string idInfoFilePath, std::string idTileIndexDBFilePath )
{
    mIdInfoPath = idInfoFilePath;
    mIdTileIndexDBPath = idTileIndexDBFilePath;
    mIsDBOpen = false;

    mIdIndexHdf5FileHandle = marray::hdf5::openFile( mIdInfoPath );
    marray::hdf5::load( mIdIndexHdf5FileHandle, "idMax", mIdMax );
    marray::hdf5::load( mIdIndexHdf5FileHandle, "idColorMap", mIdColorMap );
    marray::hdf5::load( mIdIndexHdf5FileHandle, "idVoxelCountMap", mIdVoxelCountMap );
    marray::hdf5::closeFile( mIdIndexHdf5FileHandle );
}

void FileSystemIdIndex::CloseDB()
{
    if ( mIsDBOpen )
    {
        sqlite3_close( mIdTileIndexDB );
        Core::Printf( "Closed SQLite database ", mIdTileIndexDBPath, "." );
        mIsDBOpen = false;
    }
}

void FileSystemIdIndex::OpenDB()
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
        }
    }
}


static int callback(void *unused, int argc, char **argv, char **colName)
{
  return 0;
}


void FileSystemIdIndex::Save()
{

    std::string newPath = mIdInfoPath + ".temp";

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

    for ( FileSystemIdTileMapDirect::iterator idIt = mCacheIdTileMap.GetHashMap().begin(); idIt != mCacheIdTileMap.GetHashMap().end(); ++idIt )
    {
        FileSystemTileSet oldTiles = LoadTiles( idIt->first );

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

    boost::filesystem::path idInfoPath = boost::filesystem::path( mIdInfoPath );

    boost::filesystem::remove( idInfoPath );
    boost::filesystem::rename( tempidInfoPath, idInfoPath );

}

marray::Marray< unsigned char > FileSystemIdIndex::GetIdColorMap()
{
    return mIdColorMap;
}
                                               
FileSystemTileSet FileSystemIdIndex::GetTiles( unsigned int segid )
{

    if ( mCacheIdTileMap.GetHashMap().find( segid ) != mCacheIdTileMap.GetHashMap().end() )
    {
        return mCacheIdTileMap.Get( segid );
    }
    else
    {
        return LoadTiles( segid );
    }
}

FileSystemTileSet FileSystemIdIndex::LoadTiles( unsigned int segid )
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

unsigned int FileSystemIdIndex::GetTileCount ( unsigned int segid )
{
    return GetTiles( segid ).size();
}

unsigned int FileSystemIdIndex::GetVoxelCount ( unsigned int segid )
{
    return mIdVoxelCountMap( segid );
}
                                               
unsigned int FileSystemIdIndex::GetMaxId()
{
    return mIdMax( 0 );
}

unsigned int FileSystemIdIndex::AddNewId()
{
    mIdMax( 0 ) = mIdMax( 0 ) + 1;

    size_t shape[] = { mIdMax( 0 ) + 1 };
    mIdVoxelCountMap.resize( shape, shape + 1 );
    mIdVoxelCountMap( mIdMax(0) ) = 0;

    return mIdMax( 0 );
}
                                               
void FileSystemIdIndex::SetTiles( unsigned int segid, FileSystemTileSet tiles )
{
    mCacheIdTileMap.GetHashMap()[ segid ] = tiles;
}

void FileSystemIdIndex::SetVoxelCount ( unsigned int segid, unsigned long voxelCount )
{
    mIdVoxelCountMap( segid ) = voxelCount;
}


}
}
