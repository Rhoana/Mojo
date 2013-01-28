#include "FileSystemIdMaps.hpp"

namespace Mojo
{
namespace Native
{

FileSystemIdMaps::FileSystemIdMaps()
{
}

FileSystemIdMaps::FileSystemIdMaps( std::string idMapsPath )
{
    mIdMapsPath = idMapsPath;
    mIdMapsHdf5FileHandle = marray::hdf5::openFile( mIdMapsPath );

    marray::hdf5::load( mIdMapsHdf5FileHandle, "idMax", mIdMax );
    //marray::hdf5::load( mIdMapsHdf5FileHandle, "idColorMap", mIdColorMap );
    marray::hdf5::load( mIdMapsHdf5FileHandle, "idVoxelCountMap", mIdVoxelCountMap );

    mIdTileMapGroupHandle = marray::hdf5::openGroup( mIdMapsHdf5FileHandle, "idTileMap" );
}

FileSystemIdMaps::~FileSystemIdMaps()
{
    Close();
}

void FileSystemIdMaps::Close()
{
    if ( mIdTileMapGroupHandle != 0 )
    {
        marray::hdf5::closeGroup( mIdTileMapGroupHandle );
        mIdTileMapGroupHandle = 0;
    }
    if ( mIdMapsHdf5FileHandle != 0 )
    {
        marray::hdf5::closeFile( mIdMapsHdf5FileHandle );
        mIdMapsHdf5FileHandle = 0;
    }
}

void FileSystemIdMaps::Save()
{
    marray::hdf5::save( mIdMapsHdf5FileHandle, "idMax", mIdMax );
    //marray::hdf5::save( mIdMapsHdf5FileHandle, "idColorMap", mIdColorMap );
    marray::hdf5::save( mIdMapsHdf5FileHandle, "idVoxelCountMap", mIdVoxelCountMap );

    for ( FileSystemIdTileMapDirect::iterator idIt = mCacheIdTileMap.GetHashMap().begin();
        idIt != mCacheIdTileMap.GetHashMap().end(); ++idIt )
    {
        size_t shape[] = { idIt->second.size(), 4 };
        marray::Marray< unsigned int > rawIdTileMap = marray::Marray< unsigned int >( shape, shape + 2 );

        unsigned int i = 0;
        for ( FileSystemTileSet::iterator tileIt = idIt->second.begin(); tileIt != idIt->second.end(); ++tileIt )
        {
            rawIdTileMap( i, 0 ) = tileIt->w;
            rawIdTileMap( i, 1 ) = tileIt->z;
            rawIdTileMap( i, 2 ) = tileIt->y;
            rawIdTileMap( i, 3 ) = tileIt->x;
        }

        std::string idString;
        std::stringstream converter;

        converter << idIt->first;
        idString = converter.str();

        marray::hdf5::save( mIdTileMapGroupHandle, idString, rawIdTileMap );

    }

}

void FileSystemIdMaps::SaveAs( std::string newIdMapsPath )
{
}

marray::Marray< unsigned char > FileSystemIdMaps::GetIdColorMap()
{
    marray::Marray< unsigned char > idColorMap;
    //marray::hdf5::load( mIdMapsHdf5FileHandle, "idColorMap", mIdColorMap );
    return idColorMap;
}
                                               
FileSystemTileSet FileSystemIdMaps::GetTiles( unsigned int segid )
{

    if ( mCacheIdTileMap.GetHashMap().find( segid ) == mCacheIdTileMap.GetHashMap().end() )
    {
        //
        // Load the tile id map from the hdf5
        //
        std::string idString;
        std::stringstream converter;
        marray::Marray< unsigned int > rawIdTileMap;

        converter << segid;
        idString = converter.str();

        marray::hdf5::load( mIdTileMapGroupHandle, idString, rawIdTileMap );

        unsigned int i;
        for ( i = 0; i < rawIdTileMap.shape( 0 ); i++ )
        {
            mCacheIdTileMap.GetHashMap()[ segid ].insert( make_int4( rawIdTileMap( i, 3 ), rawIdTileMap( i, 2 ), rawIdTileMap( i, 1 ), rawIdTileMap( i, 0 ) ) );
        }
    }

    return mCacheIdTileMap.Get( segid );

}

unsigned int FileSystemIdMaps::GetTileCount ( unsigned int segid )
{
    return GetTiles( segid ).size();
}

unsigned int FileSystemIdMaps::GetVoxelCount ( unsigned int segid )
{
    return mIdVoxelCountMap( segid );
}
                                               
unsigned int FileSystemIdMaps::GetMaxId()
{
    return mIdMax( 0 );
}

unsigned int FileSystemIdMaps::AddNewId()
{
    mIdMax( 0 ) = mIdMax( 0 ) + 1;
    return mIdMax( 0 );
}
                                               
void FileSystemIdMaps::SetTiles( unsigned int segid, FileSystemTileSet tiles )
{
    mCacheIdTileMap.GetHashMap()[ segid ] = tiles;
}

void FileSystemIdMaps::SetVoxelCount ( unsigned int segid, unsigned long voxelCount )
{
    mIdVoxelCountMap( segid ) = voxelCount;
}


}
}
