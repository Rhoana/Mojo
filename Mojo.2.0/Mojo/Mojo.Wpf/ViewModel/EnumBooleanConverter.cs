using System;
using System.Windows;
using System.Windows.Data;

namespace Mojo.Wpf.ViewModel
{
    public class EnumBooleanConverter : IValueConverter
    {
        public object Convert( object value, Type targetType, object parameter, System.Globalization.CultureInfo culture )
        {
            var parameterString = parameter as string;

            if ( parameterString == null )
            {
                return DependencyProperty.UnsetValue;                
            }

            if ( value == null )
            {
                return DependencyProperty.UnsetValue;
            }

            if ( !Enum.IsDefined( value.GetType(), value ) )
            {
                return DependencyProperty.UnsetValue;                
            }

            if ( ((ToolMode)value) == ToolMode.DrawMergeSegmentation )
            {
                value = ToolMode.MergeSegmentation;
            }

            var parameterValue = Enum.Parse( value.GetType(), parameterString );

            return parameterValue.Equals( value );
        }

        public object ConvertBack( object value, Type targetType, object parameter, System.Globalization.CultureInfo culture )
        {
            var parameterString = parameter as string;

            if ( parameterString == null )
            {
                return DependencyProperty.UnsetValue;                
            }

            return Enum.Parse( targetType, parameterString );                
        }
    }

}
