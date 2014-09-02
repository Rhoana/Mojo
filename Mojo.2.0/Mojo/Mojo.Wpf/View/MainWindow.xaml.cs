using System;
using System.Windows.Controls.Primitives;
using System.Windows.Forms.Integration;
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

        public void SelectedSegmentLockCheckBox_OnUnchecked( object sender, RoutedEventArgs e )
        {
            if ( DataContext != null && ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentConfidence != 0 )
            {
                var segId = ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.SelectedSegmentId;
                if (segId == 0) return;
                ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.UnlockSegmentLabel( segId );
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentConfidence = 0;
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.UpdateSegmentInfoList( segId );
            }
        }

        public void SelectedSegmentLockCheckBox_OnChecked( object sender, RoutedEventArgs e )
        {
            if ( DataContext != null && ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentConfidence != 100 )
            {
                var segId = ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.SelectedSegmentId;
                if (segId == 0) return;
                ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.LockSegmentLabel( segId );
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentConfidence = 100;
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.UpdateSegmentInfoList( segId );
            }
        }

        public void SegmentLockCheckBox_OnUnchecked( object sender, RoutedEventArgs e )
        {
            var changedSegment = (Interop.SegmentInfo)( (CheckBox)sender ).DataContext;
            if ( changedSegment != null && changedSegment.Confidence != 0 )
            {
                changedSegment.Confidence = 0;
                ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.UnlockSegmentLabel( changedSegment.Id );
                if ( changedSegment.Id == ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.SelectedSegmentId )
                {
                    ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentConfidence = 0;
                }
            }
        }

        public void SegmentLockCheckBox_OnChecked( object sender, RoutedEventArgs e )
        {
            var changedSegment = (Interop.SegmentInfo)( (CheckBox)sender ).DataContext;
            if ( changedSegment != null && changedSegment.Confidence != 100 )
            {
                changedSegment.Confidence = 100;
                ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.LockSegmentLabel( changedSegment.Id );
                if ( changedSegment.Id == ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.SelectedSegmentId )
                {
                    ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentConfidence = 100;
                }
            }
        }

        public void SegmentTypeRadioButton_Checked( object sender, RoutedEventArgs e )
        {
            var newType = ( (RadioButton)sender ).CommandParameter.ToString();
            ViewModel.TileManagerDataContext.SegmentType newEnumType = (ViewModel.TileManagerDataContext.SegmentType)Enum.Parse( typeof( ViewModel.TileManagerDataContext.SegmentType ), newType );

            if ( DataContext != null && ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentType != newEnumType )
            {
                var segId = ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.SelectedSegmentId;
                ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.SetSegmentType( segId, newType );
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentType = newEnumType;
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.UpdateSegmentInfoList( segId );
            }
        }

        public void SegmentSubTypeRadioButton_Checked( object sender, RoutedEventArgs e )
        {
            var newSubType = ( (RadioButton)sender ).CommandParameter.ToString();
            ViewModel.TileManagerDataContext.SegmentSubType newEnumSubType = (ViewModel.TileManagerDataContext.SegmentSubType)Enum.Parse( typeof( ViewModel.TileManagerDataContext.SegmentSubType ), newSubType );

            if ( DataContext != null && ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentSubType != newEnumSubType )
            {
                var segId = ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.SelectedSegmentId;
                ( (ViewModel.EngineDataContext)DataContext ).Engine.TileManager.SetSegmentSubType( segId, newSubType );
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.SelectedSegmentSubType = newEnumSubType;
                ( (ViewModel.EngineDataContext)DataContext ).TileManagerDataContext.UpdateSegmentInfoList( segId );
            }
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
