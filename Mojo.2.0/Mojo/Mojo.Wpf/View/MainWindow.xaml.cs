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

        private void SegmentList_OnSelectionChanged( object sender, SelectionChangedEventArgs e )
        {
            var selectedSegment = (Interop.SegmentInfo)( (ListView)sender ).SelectedItem;
            if ( selectedSegment != null && DataContext != null )
            {
                var engine = ( (ViewModel.EngineDataContext)DataContext ).Engine;
                if ( engine.TileManager.SelectedSegmentId != selectedSegment.Id )
                {
                    engine.SelectSegment( selectedSegment.Id );
                }
                ( (ListView)sender ).ScrollIntoView( selectedSegment );
            }
        }

        public void SegmentLockCheckBox_OnUnchecked( object sender, RoutedEventArgs e )
        {
            var changedSegment = (Interop.SegmentInfo) ( (CheckBox) sender ).DataContext;
            if ( changedSegment != null && changedSegment.Confidence != 0 )
            {
                changedSegment.Confidence = 0;
                ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.UnlockSegmentLabel( changedSegment.Id );
            }
        }

        public void SegmentLockCheckBox_OnChecked( object sender, RoutedEventArgs e )
        {
            var changedSegment = (Interop.SegmentInfo)( (CheckBox)sender ).DataContext;
            if ( changedSegment != null && changedSegment.Confidence != 100 )
            {
                changedSegment.Confidence = 100;
                ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.LockSegmentLabel( changedSegment.Id );
            }
        }

        private void SegmentList_OnMouseWheel( object sender, System.Windows.Input.MouseWheelEventArgs e )
        {
            //
            // Take the focus so that the main window does not zoom.
            //
            Keyboard.Focus( (ListView)sender );
        }

        private void SegmentListViewItem_OnMouseEnter( object sender, MouseEventArgs e )
        {
            var segmentViewItem = ( (ListViewItem)sender );
            ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.MouseOverSegmentId = (uint)segmentViewItem.Tag;
        }

        private void SegmentListViewItem_OnMouseLeave( object sender, MouseEventArgs e )
        {
            ( (ViewModel.EngineDataContext) DataContext ).Engine.TileManager.MouseOverSegmentId = 0;
        }

        private void SegmentListViewItem_OnMouseDoubleClick( object sender, MouseEventArgs e )
        {
            var segmentViewItem = ( (ListViewItem)sender );
            ( (ViewModel.EngineDataContext)DataContext ).Engine.CenterAndZoomToSegmentCentroid3D( (uint)segmentViewItem.Tag );
        }

        private GridViewColumnHeader mCurrentSortColumHeader = null;
        private bool mSordDescending = true;

        private void SegmentListColumnHeader_OnClick( object sender, RoutedEventArgs e )
        {
            var header = sender as GridViewColumnHeader;
            if ( header != null )
            {
                //
                // Determine sort direction
                //
                if ( header.Equals( mCurrentSortColumHeader ) )
                {
                    mSordDescending = !mSordDescending;
                }
                else
                {
                    //
                    // Sort size and Lock descending on first click (other colums ascending)
                    //
                    mSordDescending = ( header.Tag.Equals( "Size" ) && mCurrentSortColumHeader != null ) || header.Tag.Equals( "Lock" );

                    //
                    // Reset previous sort column arrow
                    //
                    if ( mCurrentSortColumHeader != null )
                    {
                        mCurrentSortColumHeader.Column.HeaderTemplate = null;
                    }
                }

                //
                // Add sort column arrow
                //
                if ( mSordDescending )
                {
                    header.Column.HeaderTemplate = Resources["HeaderTemplateArrowUp"] as DataTemplate;
                }
                else
                {
                    header.Column.HeaderTemplate = Resources["HeaderTemplateArrowDown"] as DataTemplate;
                }
                
                mCurrentSortColumHeader = header;

                //
                // Perform sort
                //
                var fieldName = header.Tag as String;
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SortSegmentListBy( fieldName, mSordDescending );

                SegmentList.SelectedValue = ( (ViewModel.EngineDataContext) DataContext ).Engine.TileManager.SelectedSegmentId;

            }
        }

    }


}
