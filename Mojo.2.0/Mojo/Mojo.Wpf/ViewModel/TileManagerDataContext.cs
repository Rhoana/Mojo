using System;
using System.ComponentModel;
using System.Windows.Media;
using System.Collections.Generic;
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
                    JumpToSelectedSegmentInfoPage();
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

        private string mSegmentListCurrentPageString;
        public string SegmentListCurrentPageString
        {
            get
            {
                return mSegmentListCurrentPageString;
            }
            set
            {
                mSegmentListCurrentPageString = value;
                OnPropertyChanged( "SegmentListCurrentPageString" );
            }
        }

        public void UpdateSegmentListCurrentPageString()
        {
            if ( mTileManager.TiledDatasetLoaded )
            {
                SegmentListCurrentPageString = "Page " + ( mSegmentInfoCurrentPageIndex + 1 ) + " of " + mSegmentInfoPageCount;
            }
            else
            {
                SegmentListCurrentPageString = "";
            }
        }

        private IList<SegmentInfo> mSegmentInfoList;

        public IList<SegmentInfo> SegmentInfoList
        {
            get
            {
                return mSegmentInfoList;
            }
            set
            {
                mSegmentInfoList = value;
                OnPropertyChanged( "SegmentInfoList" );
            }
        }

        private SegmentInfo mSelectedSegmentInfo;
        public SegmentInfo SelectedSegmentInfo
        {
            get
            {
                return mSelectedSegmentInfo;
            }
            set
            {
                if ( SelectedSegmentInfo.Id != value.Id )
                {
                    mTileManager.SelectedSegmentId = value.Id;
                }
                mSelectedSegmentInfo = value;
                OnPropertyChanged( "SelectedSegmentInfo" );
            }
        }

        private int mSegmentInfoCurrentPageIndex = 0;
        private int mSegmentInfoPageCount = 0;
        private readonly int mItemsPerPage = Settings.Default.SegmentListItemsPerPage;

        public void UpdateSegmentInfoList()
        {
            var totalSegments = mTileManager.Internal.GetSegmentInfoCount();
            mSegmentInfoPageCount = (int) ( 1 + ( totalSegments - 1 ) / mItemsPerPage );

            SegmentInfoList = mTileManager.Internal.GetSegmentInfoRange( mSegmentInfoCurrentPageIndex * mItemsPerPage, ( mSegmentInfoCurrentPageIndex + 1 ) * mItemsPerPage );
            UpdateSegmentListCurrentPageString();
        }

        public void SortSegmentListBy( String fieldName )
        {
            SortSegmentListBy( fieldName, false );
        }

        public void SortSegmentListBy( String fieldName, bool sordDescending )
        {
            switch ( fieldName )
            {
                case "Id":
                    mTileManager.Internal.SortSegmentInfoById( sordDescending );
                    break;
                case "Name":
                    mTileManager.Internal.SortSegmentInfoByName( sordDescending );
                    break;
                case "Size":
                    mTileManager.Internal.SortSegmentInfoBySize( sordDescending );
                    break;
                case "Lock":
                    mTileManager.Internal.SortSegmentInfoByConfidence( sordDescending );
                    break;
            }

            SegmentInfoList = mTileManager.Internal.GetSegmentInfoRange( 0, mItemsPerPage );
            mSegmentInfoCurrentPageIndex = 0;

            UpdateSegmentInfoList();
        }

        public void MoveToFirstSegmentInfoPage()
        {
            mSegmentInfoCurrentPageIndex = 0;
            UpdateSegmentInfoList();
        }

        public void MoveToPreviousSegmentInfoPage()
        {
            if ( mSegmentInfoCurrentPageIndex > 0 )
            {
                --mSegmentInfoCurrentPageIndex;
                UpdateSegmentInfoList();
            }
        }

        public void MoveToNextSegmentInfoPage()
        {
            if ( mSegmentInfoCurrentPageIndex < mSegmentInfoPageCount - 1 )
            {
                ++mSegmentInfoCurrentPageIndex;
                UpdateSegmentInfoList();
            }
        }

        public void MoveToLastSegmentInfoPage()
        {
            mSegmentInfoCurrentPageIndex = mSegmentInfoPageCount - 1;
            UpdateSegmentInfoList();
        }

        public void JumpToSelectedSegmentInfoPage()
        {
            uint segmentIndex = mTileManager.Internal.GetSegmentInfoCurrentListLocation( mTileManager.SelectedSegmentId );
            mSegmentInfoCurrentPageIndex = (int) (segmentIndex / mItemsPerPage);
            UpdateSegmentInfoList();
        }

    }
}
