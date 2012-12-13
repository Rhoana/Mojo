#pragma once

#include "Mojo.Core/PrimitiveMap.hpp"

#using <SlimDX.dll> 

using namespace System;
using namespace SlimDX;
using namespace SlimDX::Direct3D11;

namespace Mojo
{
namespace Interop
{

#pragma managed
public enum class PrimitiveType : Int32
{
    Float4,
    Float2,
    UChar4,
    Float,
    Int,
    Bool
};

#pragma managed
public ref class PrimitiveMap : public Collections::IEnumerable
{
public:
    PrimitiveMap();

    virtual Collections::IEnumerator^ GetEnumerator();
    void                              Add( PrimitiveType type, String^ key, SlimDX::Vector4 value );
    void                              Add( PrimitiveType type, String^ key, SlimDX::Vector2 value );
    void                              Add( PrimitiveType type, String^ key, float           value );
    void                              Add( PrimitiveType type, String^ key, int             value );
    void                              Add( PrimitiveType type, String^ key, bool            value );

    Core::PrimitiveMap ToCore();

    SlimDX::Vector4 GetFloat4( String^ key );
    SlimDX::Vector2 GetFloat2( String^ key );
    SlimDX::Vector4 GetUChar4( String^ key );
    float           GetFloat ( String^ key );
    int             GetInt   ( String^ key );
    bool            GetBool  ( String^ key );

    void SetFloat4( String^ key, SlimDX::Vector4 value );
    void SetFloat2( String^ key, SlimDX::Vector2 value );
    void SetUChar4( String^ key, SlimDX::Vector4 value );
    void SetFloat ( String^ key, float           value );
    void SetInt   ( String^ key, int             value );
    void SetBool  ( String^ key, bool            value );

    Collections::Generic::IDictionary< String^, SlimDX::Vector4 >^ GetDictionaryFloat4();
    Collections::Generic::IDictionary< String^, SlimDX::Vector2 >^ GetDictionaryFloat2();
    Collections::Generic::IDictionary< String^, SlimDX::Vector4 >^ GetDictionaryUChar4();
    Collections::Generic::IDictionary< String^, float           >^ GetDictionaryFloat();
    Collections::Generic::IDictionary< String^, int             >^ GetDictionaryInt();
    Collections::Generic::IDictionary< String^, bool            >^ GetDictionaryBool();

private:
    Collections::Generic::IDictionary< String^, SlimDX::Vector4 >^ mFloat4;
    Collections::Generic::IDictionary< String^, SlimDX::Vector2 >^ mFloat2;
    Collections::Generic::IDictionary< String^, SlimDX::Vector4 >^ mUChar4;
    Collections::Generic::IDictionary< String^, float >^           mFloat;
    Collections::Generic::IDictionary< String^, int >^             mInt;
    Collections::Generic::IDictionary< String^, bool >^            mBool;
};

}
}
