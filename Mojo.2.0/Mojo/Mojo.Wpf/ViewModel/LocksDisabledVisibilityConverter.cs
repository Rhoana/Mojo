using System;
using System.Windows;
using System.Windows.Data;

namespace Mojo.Wpf.ViewModel
{
    public class LocksDisabledVisibilityConverter: IMultiValueConverter
    {
        public object Convert( object[] values, Type targetType, object parameter, System.Globalization.CultureInfo culture )
        {
            if ( values == null || values.Length < 2 || values[ 0 ] == null || values[ 1 ] == null || values[ 0 ] == DependencyProperty.UnsetValue || values[ 1 ] == DependencyProperty.UnsetValue )
            {
                return DependencyProperty.UnsetValue;
            }

            if ( (bool)values[ 0 ] && !(bool)values[ 1 ] )
            {
                return Visibility.Visible;
            }
            return Visibility.Hidden;
        }

        public object[] ConvertBack( object value, Type[] targetTypes, object parameter, System.Globalization.CultureInfo culture )
        {
            return null;
        }
    }
}
