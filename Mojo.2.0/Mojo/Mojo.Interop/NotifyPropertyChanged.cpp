#include "NotifyPropertyChanged.hpp"

#include "Mojo.Native/Assert.hpp"

namespace Mojo
{
namespace Interop
{

void NotifyPropertyChanged::OnPropertyChanged( String^ propertyName )
{
    //
    // CODE QUALITY ISSUE:
    // Again, commenting out this assert is the wrong approach. This assert has caught
    // lots of bugs. If you call OnPropertyChanged and pass the name of a property that
    // doesn't exist, this assert is the only protection you have against a difficult-to-isolate
    // WPF bug. -MR
    //
    //RELEASE_ASSERT( TypeDescriptor::GetProperties( this )[ propertyName ] != nullptr );

    RELEASE_ASSERT( TypeDescriptor::GetProperties( this )[ propertyName ] != nullptr );
    PropertyChanged( this, gcnew PropertyChangedEventArgs( propertyName ) );
}

}
}