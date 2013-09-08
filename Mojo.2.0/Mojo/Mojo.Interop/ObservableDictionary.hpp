#pragma once

#include "Mojo.Native/Stl.hpp"

#include "NotifyPropertyChanged.hpp"

#using <ObservableDictionary.dll>

using namespace System;
using namespace DrWPF::Windows::Data;

namespace Mojo
{
namespace Interop
{

void ObservableDictionaryError();

#pragma managed
generic < typename K, typename V >
public ref class ObservableDictionary : public NotifyPropertyChanged, Collections::IEnumerable
{
public:
    ObservableDictionary();

    //
    // these methods enfore error-checking get and set semantics
    //
    V    Get( K key );
    void Set( K key, V value );

    //
    // these methods and properties are required for C# interop
    //
    void                              Add( K key, V value );
    virtual Collections::IEnumerator^ GetEnumerator();

    //
    // this property allows us to access the internal Observable Dictionary
    //
    property DrWPF::Windows::Data::ObservableDictionary< K, V >^ Internal
    {
        DrWPF::Windows::Data::ObservableDictionary< K, V >^ get()
        {
            return mInternal;
        }
        
        void set( DrWPF::Windows::Data::ObservableDictionary< K, V >^ value )
        {
            mInternal = value;
            OnPropertyChanged( "Internal" );
        }
    }

private:
    DrWPF::Windows::Data::ObservableDictionary< K, V >^ mInternal;

};

generic < typename K, typename V >
inline ObservableDictionary< K, V >::ObservableDictionary()
{
    Internal = gcnew DrWPF::Windows::Data::ObservableDictionary< K, V >();
}

generic < typename K, typename V >
inline void ObservableDictionary< K, V >::Add( K key, V value )
{
    Set( key, value );
}

generic < typename K, typename V >
inline Collections::IEnumerator^ ObservableDictionary< K, V >::GetEnumerator()
{
    return Internal->GetEnumerator();
}

generic < typename K, typename V >
inline V ObservableDictionary< K, V >::Get( K key )
{
    if ( !Internal->ContainsKey( key ) )
    {
        ObservableDictionaryError();                                                                                                                                                      
    }

    return Internal[ key ];
}

generic < typename K, typename V >
inline void ObservableDictionary< K, V >::Set( K key, V value )
{
    Internal[ key ] = value;
}

}
}
