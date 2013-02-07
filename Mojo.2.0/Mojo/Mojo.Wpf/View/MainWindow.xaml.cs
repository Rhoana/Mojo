using System;
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

        }

        private void SegmentList_SizeChanged( object sender, SizeChangedEventArgs e )
        {
            ( (Mojo.Wpf.ViewModel.EngineDataContext)DataContext ).SegmentListSizeChanged( sender, e );
        }

    }


}
