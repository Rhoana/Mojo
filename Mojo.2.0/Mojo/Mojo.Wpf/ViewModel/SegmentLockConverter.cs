using System;
using System.Windows;
using System.Windows.Data;

namespace Mojo.Wpf.ViewModel
{
    public class SegmentLockConverter : IValueConverter
    {
        public object Convert( object value, Type targetType, object parameter, System.Globalization.CultureInfo culture )
        {
            if ( value == null )
            {
                return DependencyProperty.UnsetValue;
            }

            var confidence = (int) value;

            return confidence >= 100;
        }

        public object ConvertBack( object value, Type targetType, object parameter, System.Globalization.CultureInfo culture )
        {
            return null;
        }
    }
}
