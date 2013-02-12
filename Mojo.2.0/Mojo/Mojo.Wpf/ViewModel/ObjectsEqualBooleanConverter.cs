using System;
using System.Windows;
using System.Windows.Data;

namespace Mojo.Wpf.ViewModel
{
    public class ObjectsEqualBooleanConverter : IMultiValueConverter
    {
        public object Convert( object[] values, Type targetType, object parameter, System.Globalization.CultureInfo culture )
        {
            for ( int i = 1; i < values.Length; ++i )
                if ( !values[0].Equals( values[i] ) )
                    return false;

            return true;

        }

        public object[] ConvertBack( object value, Type[] targetTypes, object parameter, System.Globalization.CultureInfo culture )
        {
            return null;
        }
    }
}