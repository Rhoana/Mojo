using System;
using System.Windows.Input;
using System.Windows.Controls;

using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Windows;
using System.Windows.Data;

namespace Mojo.Wpf.View
{
    public partial class MainWindow : IDisposable
    {

        public MainWindow()
        {
            InitializeComponent();
        }

        public void Dispose()
        {
        }

        private void SegmentList_SelectionChanged( object sender, SelectionChangedEventArgs e )
        {
            var selectedSegment = (Interop.SegmentInfo)( (ListView)sender ).SelectedItem;
            ( (Mojo.Wpf.ViewModel.EngineDataContext) DataContext ).Engine.SelectSegment( selectedSegment.Id );
        }

        public void SegmentLockCheckBox_OnUnchecked( object sender, RoutedEventArgs e )
        {
            var changedSegment = (Interop.SegmentInfo) ( (CheckBox) sender ).DataContext;
            if ( changedSegment.Confidence != 0 )
            {
                changedSegment.Confidence = 0;
                ( (Mojo.Wpf.ViewModel.EngineDataContext) DataContext ).Engine.UnlockSegmentLabel( changedSegment.Id );
            }
        }

        public void SegmentLockCheckBox_OnChecked( object sender, RoutedEventArgs e )
        {
            var changedSegment = (Interop.SegmentInfo)( (CheckBox)sender ).DataContext;
            if ( changedSegment.Confidence != 100 )
            {
                changedSegment.Confidence = 100;
                ( (Mojo.Wpf.ViewModel.EngineDataContext) DataContext ).Engine.LockSegmentLabel( changedSegment.Id );
            }
        }

        private void SegmentList_MouseWheel( object sender, System.Windows.Input.MouseWheelEventArgs e )
        {
            //
            // Take the focus so that the main window does not zoom.
            //
            Keyboard.Focus( (ListView)sender );
        }
    }


}
