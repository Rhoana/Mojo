#include "PrimitiveMap.hpp"

#include <msclr/marshal_cppstd.h>

#include "Mojo.Native/ForEach.hpp"
#include "Mojo.Native/ID3D11Texture.hpp"

#using <SlimDX.dll>

using namespace System;
using namespace SlimDX;

namespace Mojo
{
namespace Interop
{

PrimitiveMap::PrimitiveMap()
{
    mFloat4 = gcnew Collections::Generic::Dictionary< String^, Vector4 >();
    mFloat2 = gcnew Collections::Generic::Dictionary< String^, Vector2 >();
    mUChar4 = gcnew Collections::Generic::Dictionary< String^, Vector4 >();
    mFloat  = gcnew Collections::Generic::Dictionary< String^, float >();
    mInt    = gcnew Collections::Generic::Dictionary< String^, int >();
    mBool   = gcnew Collections::Generic::Dictionary< String^, bool >();
}

Collections::IEnumerator^ PrimitiveMap::GetEnumerator()
{
    RELEASE_ASSERT( 0 && "Iterating through instances of the PrimitiveMap class is not supported." );
    return nullptr;
}

void PrimitiveMap::Add( PrimitiveType type, String^ key, Vector4 value )
{
    switch( type )
    {
    case PrimitiveType::Float4:
        SetFloat4( key, value );
        break;
        
    case PrimitiveType::UChar4:
        SetUChar4( key, value );
        break;

    default:
        RELEASE_ASSERT( 0 );
    }
}

void PrimitiveMap::Add( PrimitiveType type, String^ key, Vector2 value )
{
    RELEASE_ASSERT( type == PrimitiveType::Float2 );
    SetFloat2( key, value );
}

void PrimitiveMap::Add( PrimitiveType type, String^ key, float value )
{
    RELEASE_ASSERT( type == PrimitiveType::Float );
    SetFloat( key, value );
}

void PrimitiveMap::Add( PrimitiveType type, String^ key, int value )
{
    RELEASE_ASSERT( type == PrimitiveType::Int );
    SetInt( key, value );
}

void PrimitiveMap::Add( PrimitiveType type, String^ key, bool value )
{
    RELEASE_ASSERT( type == PrimitiveType::Bool );
    SetBool( key, value );
}

Native::PrimitiveMap PrimitiveMap::ToCore()
{
    Native::PrimitiveMap primitiveDictionary;

    //for each ( Collections::Generic::KeyValuePair< String^, Vector4 > keyValuePair in mFloat4 )
    //{
    //    primitiveDictionary.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value.X, keyValuePair.Value.Y, keyValuePair.Value.Z, keyValuePair.Value.W );
    //}

    //for each ( Collections::Generic::KeyValuePair< String^, Vector2 > keyValuePair in mFloat2 )
    //{
    //    primitiveDictionary.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value.X, keyValuePair.Value.Y );
    //}

    //for each ( Collections::Generic::KeyValuePair< String^, Vector4 > keyValuePair in mUChar4 )
    //{
    //    primitiveDictionary.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), (unsigned char)keyValuePair.Value.X, (unsigned char)keyValuePair.Value.Y, (unsigned char)keyValuePair.Value.Z, (unsigned char)keyValuePair.Value.W );
    //}

    for each ( Collections::Generic::KeyValuePair< String^, float > keyValuePair in mFloat )
    {
        primitiveDictionary.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value );
    }

    for each ( Collections::Generic::KeyValuePair< String^, int > keyValuePair in mInt )
    {
        primitiveDictionary.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value );
    }

    for each ( Collections::Generic::KeyValuePair< String^, bool > keyValuePair in mBool )
    {
        primitiveDictionary.Set( msclr::interop::marshal_as< std::string >( keyValuePair.Key ), keyValuePair.Value );
    }

    return primitiveDictionary;
}

Vector4 PrimitiveMap::GetFloat4( String^ key )
{
    RELEASE_ASSERT( mFloat4->ContainsKey( key ) );
    return mFloat4[ key ];
}

Vector2 PrimitiveMap::GetFloat2( String^ key )
{
    RELEASE_ASSERT( mFloat2->ContainsKey( key ) );
    return mFloat2[ key ];
}

Vector4 PrimitiveMap::GetUChar4( String^ key )
{
    RELEASE_ASSERT( mUChar4->ContainsKey( key ) );
    return mUChar4[ key ];
}

float PrimitiveMap::GetFloat( String^ key )
{
    RELEASE_ASSERT( mFloat->ContainsKey( key ) );
    return mFloat[ key ];
}

int PrimitiveMap::GetInt( String^ key )
{
    RELEASE_ASSERT( mInt->ContainsKey( key ) );
    return mInt[ key ];
}

bool PrimitiveMap::GetBool( String^ key )
{
    RELEASE_ASSERT( mBool->ContainsKey( key ) );
    return mBool[ key ];
}

void PrimitiveMap::SetFloat4( String^ key, Vector4 value )
{
    mFloat4[ key ] = value;
}

void PrimitiveMap::SetFloat2( String^ key, Vector2 value )
{
    mFloat2[ key ] = value;
}

void PrimitiveMap::SetUChar4( String^ key, Vector4 value )
{
    mUChar4[ key ] = value;
}

void PrimitiveMap::SetFloat( String^ key, float value )
{
    mFloat[ key ] = value;
}

void PrimitiveMap::SetInt( String^ key, int value )
{
    mInt[ key ] = value;
}

void PrimitiveMap::SetBool( String^ key, bool value )
{
    mBool[ key ] = value;
}

Collections::Generic::IDictionary< String^, Vector4 >^ PrimitiveMap::GetDictionaryFloat4()
{
    return mFloat4;
}

Collections::Generic::IDictionary< String^, Vector2 >^ PrimitiveMap::GetDictionaryFloat2()
{
    return mFloat2;
}

Collections::Generic::IDictionary< String^, Vector4 >^ PrimitiveMap::GetDictionaryUChar4()
{
    return mUChar4;
}

Collections::Generic::IDictionary< String^, float >^ PrimitiveMap::GetDictionaryFloat()
{
    return mFloat;
}

Collections::Generic::IDictionary< String^, int >^ PrimitiveMap::GetDictionaryInt()
{
    return mInt;
}

Collections::Generic::IDictionary< String^, bool >^ PrimitiveMap::GetDictionaryBool()
{
    return mBool;
}

}
}