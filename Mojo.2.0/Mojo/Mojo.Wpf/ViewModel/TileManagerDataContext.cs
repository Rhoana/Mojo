﻿using System;
using System.ComponentModel;
using System.Windows.Media;
using Mojo.Interop;

namespace Mojo.Wpf.ViewModel
{
    public class TileManagerDataContext : NotifyPropertyChanged, IDisposable
    {
        private TileManager mTileManager;

        public event EventHandler StateChanged;

        public TileManagerDataContext( TileManager tileManager )
        {
            mTileManager = tileManager;

            mTileManager.PropertyChanged += PropertyChangedHandler;
            mTileManager.Internal.PropertyChanged += PropertyChangedHandler;
        }

        public void Dispose()
        {
            if ( mTileManager != null )
            {
                mTileManager.Internal.PropertyChanged -= PropertyChangedHandler;
                mTileManager.PropertyChanged -= PropertyChangedHandler;

                mTileManager.Dispose();
                mTileManager = null;
            }
        }

        public void Refresh()
        {
        }

        protected void OnStateChanged( EventArgs e )
        {
            var handler = StateChanged;
            if ( handler != null ) handler( this, e );
        }

        private void PropertyChangedHandler( object sender, PropertyChangedEventArgs e )
        {
            switch ( e.PropertyName )
            {
                case "TiledDatasetView":
                    UpdateCurrentZLocationString();
                    break;
                case "SelectedSegmentId":
                    UpdateSelectedSegmentBrush();
                    break;
                case "MouseOverSegmentId":
                    UpdateMouseOverSegmentBrush();
                    break;
            }

            OnStateChanged( e );

        }

        private SolidColorBrush mSelectedSegmentBrush = new SolidColorBrush();
        public SolidColorBrush SelectedSegmentBrush
        {
            get
            {
                return mSelectedSegmentBrush;
            }
            set
            {
                mSelectedSegmentBrush = value;
                OnPropertyChanged("SelectedSegmentBrush");
            }
        }

        public void UpdateSelectedSegmentBrush( )
        {
            if ( mTileManager.SelectedSegmentId == 0 )
            {
                SelectedSegmentBrush = new SolidColorBrush();
            }
            else
            {
                SlimDX.Vector3 segmentColor = mTileManager.Internal.GetSegmentationLabelColor( mTileManager.SelectedSegmentId );
                SelectedSegmentBrush = new SolidColorBrush( Color.FromRgb( (byte)( segmentColor.X ), (byte)( segmentColor.Y ), (byte)( segmentColor.Z ) ) );
            }
        }

        private SolidColorBrush mMouseOverSegmentBrush = new SolidColorBrush();
        public SolidColorBrush MouseOverSegmentBrush
        {
            get
            {
                return mMouseOverSegmentBrush;
            }
            set
            {
                mMouseOverSegmentBrush = value;
                OnPropertyChanged("MouseOverSegmentBrush");
            }
        }

        public void UpdateMouseOverSegmentBrush( )
        {
            if ( mTileManager.MouseOverSegmentId == 0 )
            {
                MouseOverSegmentBrush = new SolidColorBrush();
            }
            else
            {
                SlimDX.Vector3 segmentColor = mTileManager.Internal.GetSegmentationLabelColor( mTileManager.MouseOverSegmentId );
                MouseOverSegmentBrush = new SolidColorBrush( Color.FromRgb( (byte)( segmentColor.X ), (byte)( segmentColor.Y ), (byte)( segmentColor.Z ) ) );
            }

        }

        private string mCurrentZLocationString = "";
        public string CurrentZLocationString
        {
            get
            {
                return mCurrentZLocationString;
            }
            set
            {
                mCurrentZLocationString = value;
                OnPropertyChanged("CurrentZLocationString");
            }
        }

        public void UpdateCurrentZLocationString()
        {
            if ( mTileManager.TiledDatasetLoaded )
            {
                CurrentZLocationString = "Image " + ( mTileManager.TiledDatasetView.CenterDataSpace.Z + 1 ) + " of " + mTileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get("SourceMap").NumVoxelsZ;
            }
            else
            {
                CurrentZLocationString = "";
            }
        }
    }
}