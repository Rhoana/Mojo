#pragma once

#include "Mojo.Native/TiledDatasetDescription.hpp"

#include "NotifyPropertyChanged.hpp"
#include "ObservableDictionary.hpp"
#include "TiledVolumeDescription.hpp"

using namespace System;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class TiledDatasetDescription : public NotifyPropertyChanged
{
public:
    Native::TiledDatasetDescription ToNative();

    property ObservableDictionary< String^, TiledVolumeDescription^ >^ TiledVolumeDescriptions
    {
        ObservableDictionary< String^, TiledVolumeDescription^ >^ get()
        {
            return mTiledVolumeDescriptions;
        }
        
        void set( ObservableDictionary< String^, TiledVolumeDescription^ >^ value )
        {
            mTiledVolumeDescriptions = value;
            OnPropertyChanged( "TiledVolumeDescriptions" );
        }
    }

    property Collections::Generic::IDictionary< unsigned int, Collections::Generic::IList< Vector4 >^ >^ IdTileMap
    {
        Collections::Generic::IDictionary< unsigned int, Collections::Generic::IList< Vector4 >^ >^ get()
        {
            return mIdTileMap;
        }
        
        void set( Collections::Generic::IDictionary< unsigned int, Collections::Generic::IList< Vector4 >^ >^ value )
        {
            mIdTileMap = value;
            OnPropertyChanged( "IdTileMap" );
        }
    }

    property ObservableDictionary< String^, String^ >^ Paths
    {
        ObservableDictionary< String^, String^ >^ get()
        {
            return mPaths;
        }
        
        void set( ObservableDictionary< String^, String^ >^ value )
        {
            mPaths = value;
            OnPropertyChanged( "Paths" );
        }
    }

    property unsigned int MaxLabelId
    {
        unsigned int get()
        {
            return mMaxLabelId;
        }
        
        void set( unsigned int value )
        {
            mMaxLabelId = value;
            OnPropertyChanged( "MaxLabelId" );
        }
    }

private:
    ObservableDictionary< String^, TiledVolumeDescription^ >^                                   mTiledVolumeDescriptions;
    Collections::Generic::IDictionary< unsigned int, Collections::Generic::IList< Vector4 >^ >^ mIdTileMap;
    ObservableDictionary< String^, String^ >^                                                   mPaths;
	unsigned int                                                                                mMaxLabelId;
};

}
}
