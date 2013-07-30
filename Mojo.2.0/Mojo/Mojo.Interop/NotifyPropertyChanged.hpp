#pragma once

#include <Windows.h>

#using <System.dll>

using namespace System;
using namespace ComponentModel;

namespace Mojo
{
namespace Interop
{

#pragma managed
public ref class NotifyPropertyChanged : public INotifyPropertyChanged
{
private:
    PropertyChangedEventHandler^ mPropertyChanged;

protected:
    void OnPropertyChanged( String^ propertyName );

public:
    virtual event PropertyChangedEventHandler^ PropertyChanged
    {
    public:
        virtual void add ( PropertyChangedEventHandler^ propertyChangedEventHandler )
        {
            mPropertyChanged += propertyChangedEventHandler;
        }

        virtual void remove ( PropertyChangedEventHandler^ propertyChangedEventHandler )
        {
            mPropertyChanged -= propertyChangedEventHandler;
        }

    protected:
        void raise( Object^, ComponentModel::PropertyChangedEventArgs^ e )
        { 
            PropertyChangedEventHandler^ tmp = mPropertyChanged;

            if ( tmp != nullptr )
            {
                tmp->Invoke( this, e );
            }
        }
    }
};

}
}