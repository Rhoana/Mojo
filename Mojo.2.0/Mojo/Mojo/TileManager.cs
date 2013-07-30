using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.ComponentModel;
using Mojo.Interop;
using SlimDX;
using SlimDX.DXGI;
using System.Xml.Serialization;
using System.Xml;

namespace Mojo
{
    public class TileManager : NotifyPropertyChanged, IDisposable
    {
        public Interop.TileManager Internal { get; private set; }

        public bool SourceImagesLoaded
        {
            get
            {
                if ( Internal != null )
                {
                    return Internal.AreSourceImagesLoaded();
                }
                else
                {
                    return false;
                }
            }
        }

        public bool SegmentationLoaded
        {
            get
            {
                if ( Internal != null )
                {
                    return Internal.IsSegmentationLoaded();
                }
                else
                {
                    return false;
                }
            }
        }

        public bool TiledDatasetViewInitialized
        {
            get
            {
                return TiledDatasetView.WidthNumPixels >= 32;
            }
        }

        //
        // CODE QUALITY ISSUE:
        // Many of the properties below are UI concerns that do not belong in TileManager.cs.
        // Also, the ones with trivial implementations should be replaced with implicit properties. -MR
        //
        private bool mSegmentationChangeInProgress = false;
        public bool SegmentationChangeInProgress
        {
            get
            {
                return mSegmentationChangeInProgress;
            }
            set
            {
                mSegmentationChangeInProgress = value;
            }
        }

        private bool mBackgroundSegmentationChangeInProgress = false;
        public bool BackgroundSegmentationChangeInProgress
        {
            get
            {
                return mBackgroundSegmentationChangeInProgress;
            }
            set
            {
                mBackgroundSegmentationChangeInProgress = value;
            }
        }

        private bool mSegmentationControlsEnabled = false;
        public bool SegmentationControlsEnabled
        {
            get
            {
                return mSegmentationControlsEnabled;
            }
            set
            {
                mSegmentationControlsEnabled = value;
                OnPropertyChanged( "SegmentationControlsEnabled" );
            }
        }

        private bool mNavigationControlsEnabled = false;
        public bool NavigationControlsEnabled
        {
            get
            {
                return mNavigationControlsEnabled;
            }
            set
            {
                mNavigationControlsEnabled = value;
                OnPropertyChanged( "NavigationControlsEnabled" );
            }
        }

        private TiledDatasetDescription mSourceImagesTiledDatasetDescription;
        public TiledDatasetDescription SourceImagesTiledDatasetDescription
        {
            get
            {
                return mSourceImagesTiledDatasetDescription;
            }
            set
            {
                mSourceImagesTiledDatasetDescription = value;
                OnPropertyChanged( "SourceImagesTiledDatasetDescription" );
            }
        }

        private TiledDatasetDescription mSegmentationTiledDatasetDescription;
        public TiledDatasetDescription SegmentationTiledDatasetDescription
        {
            get
            {
                return mSegmentationTiledDatasetDescription;
            }
            set
            {
                mSegmentationTiledDatasetDescription = value;
                OnPropertyChanged( "SegmentationTiledDatasetDescription" );
            }
        }

        private TiledDatasetView mTiledDatasetView;
        public TiledDatasetView TiledDatasetView
        {
            get
            {
                return mTiledDatasetView;
            }
            set
            {
                mTiledDatasetView = value;
                OnPropertyChanged( "TiledDatasetView" );
            }
        }

        private bool mShowSegmentation = true;
        public bool ShowSegmentation
        {
            get
            {
                return SegmentationLoaded && mShowSegmentation;
            }
            set
            {
                if ( SegmentationLoaded )
                {
                    mShowSegmentation = value;
                    OnPropertyChanged( "ShowSegmentation" );
                    OnPropertyChanged( "SegmentationVisibilityRatio" );
                }
            }
        }

        public void ToggleShowSegmentation()
        {
            if ( SegmentationLoaded )
            {
                ShowSegmentation = !ShowSegmentation;
            }
        }

        private float mSegmentationVisibilityRatio = 0.5f;
        public float SegmentationVisibilityRatio
        {
            get
            {
                return ( SegmentationLoaded && mShowSegmentation ) ? mSegmentationVisibilityRatio : 0f;
            }
            set
            {
                if ( SegmentationLoaded )
                {
                    mSegmentationVisibilityRatio = value;
                    OnPropertyChanged( "SegmentationVisibilityRatio" );

                    //Someone is trying to change the segmentation visibility - make sure they can see it
                    if ( !mShowSegmentation )
                    {
                        mShowSegmentation = true;
                        OnPropertyChanged( "ShowSegmentation" );
                    }
                }
            }
        }

        public void IncreaseSegmentationVisibility()
        {
            SegmentationVisibilityRatio = System.Math.Min( SegmentationVisibilityRatio + 0.1f, 1.0f );
        }

        public void DecreaseSegmentationVisibility()
        {
            SegmentationVisibilityRatio = System.Math.Max( SegmentationVisibilityRatio - 0.1f, 0f );
        }

        private bool mShowBoundaryLines = true;
        public bool ShowBoundaryLines
        {
            get
            {
                return SegmentationLoaded && mShowBoundaryLines;
            }
            set
            {
                if ( SegmentationLoaded )
                {
                    mShowBoundaryLines = value;
                    OnPropertyChanged( "ShowBoundaryLines" );
                }
            }
        }

        public void ToggleShowBoundaryLines()
        {
            if ( SegmentationLoaded )
            {
                ShowBoundaryLines = !ShowBoundaryLines;
            }
        }

        private bool mJoinSplits3D = true;
        public bool JoinSplits3D
        {
            get
            {
                return SegmentationLoaded && mJoinSplits3D;
            }
            set
            {
                mJoinSplits3D = value;
                OnPropertyChanged( "JoinSplits3D" );
            }
        }

        public void ToggleJoinSplits3D()
        {
            if ( SegmentationLoaded )
            {
                JoinSplits3D = !JoinSplits3D;
            }
        }

        private float mSplitStartZ = 0;
        public float SplitStartZ
        {
            get
            {
                return mSplitStartZ;
            }
            set
            {
                mSplitStartZ = value;
                OnPropertyChanged( "SplitStartZ" );
            }
        }

        private uint mSelectedSegmentId = 0;
        public uint SelectedSegmentId
        {
            get
            {
                return mSelectedSegmentId;
            }
            set
            {
                if ( mSelectedSegmentId != value )
                {
                    SplitStartZ = TiledDatasetView.CenterDataSpace.Z;
                }
                mSelectedSegmentId = value;
                OnPropertyChanged( "SelectedSegmentId" );
            }
        }

        private uint mMouseOverSegmentId = 0;
        public uint MouseOverSegmentId
        {
            get
            {
                return mMouseOverSegmentId;
            }
            set
            {
                mMouseOverSegmentId = value;
                OnPropertyChanged( "MouseOverSegmentId" );
            }
        }

        private float mMouseOverX = 0;
        public float MouseOverX
        {
            get
            {
                return mMouseOverX;
            }
            set
            {
                mMouseOverX = value;
                OnPropertyChanged( "MouseOverX" );
            }
        }

        private float mMouseOverY = 0;
        public float MouseOverY
        {
            get
            {
                return mMouseOverY;
            }
            set
            {
                mMouseOverY = value;
                OnPropertyChanged( "MouseOverY" );
            }
        }

        private float mBrushSize = 10;
        public float BrushSize
        {
            get
            {
                return mBrushSize;
            }
            set
            {
                mBrushSize = value;
                OnPropertyChanged( "BrushSize" );
            }
        }

        public void IncreaseBrushSize()
        {
            if ( BrushSize < 32 )
            {
                BrushSize += 2;
            }
        }

        public void DecreaseBrushSize()
        {
            if ( BrushSize > 2 )
            {
                BrushSize -= 2;
            }
        }

        private float mMergeBrushSize = 20;
        public float MergeBrushSize
        {
            get
            {
                return mMergeBrushSize;
            }
            set
            {
                mMergeBrushSize = value;
                OnPropertyChanged( "MergeBrushSize" );
            }
        }

        public void IncreaseMergeBrushSize()
        {
            if ( MergeBrushSize < 64 )
            {
                MergeBrushSize += 4;
            }
        }

        public void DecreaseMergeBrushSize()
        {
            if ( MergeBrushSize > 4 )
            {
                MergeBrushSize -= 4;
            }
        }

        private MergeMode mCurrentMergeMode = MergeMode.GlobalReplace;
        public MergeMode CurrentMergeMode
        {
            get { return mCurrentMergeMode; }
            set
            {
                if ( mCurrentMergeMode != value )
                {
                    mCurrentMergeMode = value;
                    OnPropertyChanged( "CurrentMergeMode" );
                }
            }
        }

        private SplitMode mCurrentSplitMode = SplitMode.DrawSplit;
        public SplitMode CurrentSplitMode
        {
            get { return mCurrentSplitMode; }
            set
            {
                if ( mCurrentSplitMode != value )
                {
                    mCurrentSplitMode = value;
                    OnPropertyChanged( "CurrentSplitMode" );
                }
            }
        }

        private bool mAutoChangesMade = false;
        public bool AutoChangesMade
        {
            get { return mAutoChangesMade; }
            set
            {
                if ( mAutoChangesMade != value )
                {
                    mAutoChangesMade = value;
                    OnPropertyChanged( "AutoChangesMade" );
                }
            }
        }

        private bool mChangesMade = false;
        public bool ChangesMade
        {
            get { return mChangesMade; }
            set
            {
                if ( mChangesMade != value )
                {
                    mChangesMade = value;
                    OnPropertyChanged( "ChangesMade" );
                }
                AutoChangesMade = value;
            }
        }

        private Vector3 GetPointDataSpace( Vector2 relativeScreenOffset )
        {
            var topLeftDataSpaceX = mTiledDatasetView.CenterDataSpace.X - ( mTiledDatasetView.ExtentDataSpace.X / 2f );
            var topLeftDataSpaceY = mTiledDatasetView.CenterDataSpace.Y - ( mTiledDatasetView.ExtentDataSpace.Y / 2f );

            var offsetDataSpaceX = ( relativeScreenOffset.X ) * mTiledDatasetView.ExtentDataSpace.X;
            var offsetDataSpaceY = ( relativeScreenOffset.Y ) * mTiledDatasetView.ExtentDataSpace.Y;

            return new Vector3( topLeftDataSpaceX + offsetDataSpaceX, topLeftDataSpaceY + offsetDataSpaceY, mTiledDatasetView.CenterDataSpace.Z );
        }

        public uint GetSegmentationLabelId( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return 0;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return 0;
                return Internal.GetSegmentationLabelId( mTiledDatasetView, GetPointDataSpace( p ) );
            }
        }

        public void MouseOver( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                Vector3 pointDataSpace = GetPointDataSpace( p );
                MouseOverSegmentId = Internal.GetSegmentationLabelId( mTiledDatasetView, pointDataSpace );
                MouseOverX = pointDataSpace.X;
                MouseOverY = pointDataSpace.Y;
            }
        }

        public void PrepForAdjust( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.PrepForAdjust( mSelectedSegmentId, GetPointDataSpace( p ) );
            }
        }

        public void PrepForDrawMerge( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.PrepForDrawMerge( GetPointDataSpace( p ) );
            }
        }

        public void ResetDrawMergeState( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.ResetDrawMergeState( GetPointDataSpace( p ) );
            }
        }

        public void PrepForSplit( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.PrepForSplit( mSelectedSegmentId, GetPointDataSpace( p ) );
            }
        }

        public void AddSplitSource( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.AddSplitSource( mTiledDatasetView, GetPointDataSpace( p ) );
            }
        }

        public void DrawRegionA( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.DrawRegionA( mTiledDatasetView, GetPointDataSpace( p ), mBrushSize );
            }
        }

        public void DrawRegionA( Vector2 p, float brushSize )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.DrawRegionA( mTiledDatasetView, GetPointDataSpace( p ), brushSize );
            }
        }

        public void DrawRegionB( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.DrawRegionB( mTiledDatasetView, GetPointDataSpace( p ), mBrushSize );
            }
        }

        public void DrawRegionB( Vector2 p, float brushSize)
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.DrawRegionB( mTiledDatasetView, GetPointDataSpace( p ), brushSize );
            }
        }

        public void DrawSplit( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.DrawSplit( mTiledDatasetView, GetPointDataSpace( p ), mBrushSize );
            }
        }

        public void DrawErase( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.DrawErase( mTiledDatasetView, GetPointDataSpace( p ), mBrushSize );
            }
        }

        public void FindBoundaryWithinRegion2D( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.FindBoundaryWithinRegion2D( mSelectedSegmentId, GetPointDataSpace( p ) );
            }
        }

        public void FindBoundaryBetweenRegions2D( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.FindBoundaryBetweenRegions2D( mSelectedSegmentId, GetPointDataSpace( p ) );
            }
        }

        public void FindBoundaryJoinPoints2D( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.FindBoundaryJoinPoints2D( mSelectedSegmentId, GetPointDataSpace( p ) );
            }
        }

        public void PredictSplit( Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.PredictSplit( mSelectedSegmentId, GetPointDataSpace( p ), mBrushSize );
            }
        }

        private void DoWorkAndUpdateProgressBlocking( DoWorkEventHandler workDelegate )
        {
            SegmentationChangeProgress = 10;
            BackgroundSegmentationChangeInProgress = true;

            //
            // BackgroundWorker to do the change
            //
            BackgroundWorker worker = new BackgroundWorker();
            worker.DoWork += workDelegate;
            worker.RunWorkerAsync();

            //
            // Monitor progress here (blocking)
            //
            System.Threading.Thread.Sleep( TimeSpan.FromSeconds( 0.2 ) );
            while ( worker.IsBusy )
            {
                SegmentationChangeProgress = Internal.GetCurrentOperationProgress() * 80 + 10;
                System.Threading.Thread.Sleep( TimeSpan.FromSeconds( 0.2 ) );
            }

            BackgroundSegmentationChangeInProgress = false;
            SegmentationChangeProgress = 100;

            UpdateView();
        }

        //private void DoWorkAndUpdateProgressAsync( DoWorkEventHandler workDelegate )
        //{
        //    SegmentationChangeProgress = 10;

        //    //
        //    // BackgroundWorker to do the change
        //    //
        //    BackgroundWorker worker = new BackgroundWorker();
        //    worker.DoWork += workDelegate;
        //    worker.RunWorkerAsync();

        //    //
        //    // BackgroundWorker to monitor the progress
        //    // TODO: Requires interface to be disabled so that no tile loads occur.
        //    //
        //    BackgroundWorker progressWorker = new BackgroundWorker();
        //    progressWorker.WorkerReportsProgress = true;

        //    progressWorker.DoWork += delegate( object s, DoWorkEventArgs args )
        //    {
        //        System.Threading.Thread.Sleep( TimeSpan.FromSeconds( 0.2 ) );
        //        while ( worker.IsBusy )
        //        {
        //            float progress = Internal.GetCurrentOperationProgress();
        //            progressWorker.ReportProgress( (int)( progress * 80 + 10 ) );
        //            System.Threading.Thread.Sleep( TimeSpan.FromSeconds( 0.2 ) );
        //        }
        //        progressWorker.ReportProgress( 100 );
        //    };

        //    progressWorker.ProgressChanged += delegate( object s, ProgressChangedEventArgs args )
        //    {
        //        SegmentationChangeProgress = (float)args.ProgressPercentage;
        //        if ( args.ProgressPercentage == 100 )
        //        {
        //            SegmentationChangeInProgress = false;
        //            UpdateView();
        //        }
        //    };

        //    progressWorker.RunWorkerAsync(); 

        //}

        public void CommmitSplitChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                if ( CurrentSplitMode == SplitMode.JoinPoints )
                {
                    DoWorkAndUpdateProgressBlocking(
                        delegate( object s, DoWorkEventArgs args )
                        {
                            Internal.CompletePointSplit( mSelectedSegmentId, new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                        });
                }
                else
                {
                    DoWorkAndUpdateProgressBlocking(
                        delegate( object s, DoWorkEventArgs args )
                        {
                            Internal.CompleteDrawSplit( mSelectedSegmentId, new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ), mJoinSplits3D, (int)mSplitStartZ );
                        } );
                }

                SplitStartZ = TiledDatasetView.CenterDataSpace.Z;
                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void CancelSplitChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.ResetSplitState( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
            }
        }

        public void CommmitAdjustChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        Internal.CommitAdjustChange( SelectedSegmentId, new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void CancelAdjustChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                Internal.ResetAdjustState( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
            }
        }

        public void CommitDrawMerge()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        mSelectedSegmentId = Internal.CommitDrawMerge( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void CommitDrawMergeCurrentSlice()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        mSelectedSegmentId = Internal.CommitDrawMergeCurrentSlice( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void CommitDrawMergeCurrentConnectedComponent()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        mSelectedSegmentId = Internal.CommitDrawMergeCurrentConnectedComponent( new Vector3( mMouseOverX, mMouseOverY, mTiledDatasetView.CenterDataSpace.Z ) );
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
            return;
        }

        public void RemapSegmentLabel( uint clickedId )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        Internal.RemapSegmentLabel( clickedId, mSelectedSegmentId );
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void ReplaceSegmentationLabelCurrentSlice( uint clickedId, Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        Internal.ReplaceSegmentationLabelCurrentSlice( clickedId, mSelectedSegmentId, mTiledDatasetView, GetPointDataSpace( p ) );
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void ReplaceSegmentationLabelCurrentConnectedComponent( uint clickedId, Vector2 p )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        Internal.ReplaceSegmentationLabelCurrentConnectedComponent( clickedId, mSelectedSegmentId, mTiledDatasetView, GetPointDataSpace( p ) );
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void UndoChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        Internal.UndoChange();
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void RedoChange()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                DoWorkAndUpdateProgressBlocking(
                    delegate( object s, DoWorkEventArgs args )
                    {
                        Internal.RedoChange();
                    } );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void SelectNewId()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;
                SelectedSegmentId = Internal.GetNewId();
            }
        }

        public void LockSegmentLabel( uint segId )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                Internal.LockSegmentLabel( segId );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        public void UnlockSegmentLabel( uint segId )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                Internal.UnlockSegmentLabel( segId );

                ChangesMade = true;

                SegmentationChangeInProgress = false;
            }
        }

        private float mSegmentationChangeProgress = 100;
        public float SegmentationChangeProgress
        {
            get
            {
                return mSegmentationChangeProgress;
            }
            set
            {
                mSegmentationChangeProgress = value;
                OnPropertyChanged( "SegmentationChangeProgress" );
            }
        }


        public TileManager( Interop.TileManager tileManager )
        {
            Internal = tileManager;
            TiledDatasetView = new TiledDatasetView();
        }

        public void Dispose()
        {
            UnloadSegmentation();
            UnloadSourceImages();

            if (Internal != null)
            {
                Internal.Dispose();
                Internal = null;
            }
        }

        public void Update()
        {
            if ( SourceImagesLoaded && TiledDatasetViewInitialized )
            {
                Internal.LoadTiles( TiledDatasetView );
            }
        }

        public void UpdateOneTile()
        {
            if ( SourceImagesLoaded )
            {
                Internal.LoadOverTile( TiledDatasetView );
            }
        }

        public void UpdateView()
        {
            OnPropertyChanged( "TiledDatasetView" );
            OnPropertyChanged( "SegmentationControlsEnabled" );
            OnPropertyChanged( "NavigationControlsEnabled" );
            OnPropertyChanged( "ShowSegmentation" );
            OnPropertyChanged( "SegmentationVisibilityRatio" );
            OnPropertyChanged( "ShowBoundaryLines" );
            OnPropertyChanged( "SelectedSegmentId" );
            OnPropertyChanged( "MouseOverSegmentId" );
            OnPropertyChanged( "CurrentMergeMode" );
            OnPropertyChanged( "CurrentSplitMode" );
            OnPropertyChanged( "JoinSplits3D" );
        }

        public void UpdateZ()
        {
            OnPropertyChanged( "TiledDatasetView" );
        }

        public void UpdateXYZ()
        {
            OnPropertyChanged( "TiledDatasetView" );
        }

        public void UpdateSegmentListFocus()
        {
            OnPropertyChanged( "SegmentListFocus" );
            OnPropertyChanged( "SelectedSegmentId" );
        }

        public void LoadSourceImages( string tiledDatasetRootDirectory )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                UnloadSegmentation();
                UnloadSourceImages();

                if ( !Directory.Exists( tiledDatasetRootDirectory ) )
                {
                    throw new Exception( "Image subdirectory " + tiledDatasetRootDirectory + " not found." );
                }

                if ( !Directory.Exists( Path.Combine( tiledDatasetRootDirectory, Constants.SOURCE_MAP_ROOT_DIRECTORY_NAME ) ) )
                {
                    throw new Exception( "Image subdirectory " + Path.Combine( tiledDatasetRootDirectory, Constants.SOURCE_MAP_ROOT_DIRECTORY_NAME ) + " not found." );
                }

                var tiledDatasetDescription = new TiledDatasetDescription
                                              {
                                                  TiledVolumeDescriptions =
                                                      new ObservableDictionary<string, TiledVolumeDescription>
                                                      {
                                                          { "SourceMap", GetTiledVolumeDescription( Path.Combine( tiledDatasetRootDirectory, Constants.SOURCE_MAP_TILED_VOLUME_DESCRIPTION_NAME ) ) }
                                                      },
                                                      Paths = new ObservableDictionary<string, string>
                                                      {
                                                          { "SourceMap", Path.Combine( tiledDatasetRootDirectory, Constants.SOURCE_MAP_ROOT_DIRECTORY_NAME ) }
                                                      },
                                                      MaxLabelId = 0
                                              };

                LoadSourceImages( tiledDatasetDescription );

                //
                // CODE QUALITY ISSUE:
                // The tile manager shouldn't have to care about these UI concerns. -MR
                //
                NavigationControlsEnabled = true;
                SegmentationChangeInProgress = false;
                ChangesMade = false;

                UpdateView();
            }
        }

        public void LoadSegmentation( string segmentationRootDirectory )
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                UnloadSegmentation();

                if ( !Directory.Exists( segmentationRootDirectory ) )
                {
                    throw new Exception( "Dataset directory not found." );
                }

                if ( !Directory.Exists( Path.Combine( segmentationRootDirectory, Constants.ID_MAP_ROOT_DIRECTORY_NAME ) ) )
                {
                    throw new Exception( "Id subdirectory not found." );
                }

                var tiledDatasetDescription = new TiledDatasetDescription
                                              {
                                                  TiledVolumeDescriptions =
                                                      new ObservableDictionary<string, TiledVolumeDescription>
                                                      {
                                                          { "IdMap", GetTiledVolumeDescription( Path.Combine( segmentationRootDirectory, Constants.ID_MAP_TILED_VOLUME_DESCRIPTION_NAME ) ) }
                                                      },
                                                  Paths =
                                                      new ObservableDictionary<string, string>
                                                      {
                                                          { "SegmentationRoot", segmentationRootDirectory },
                                                          { "SegmentationRootSuffix", Constants.SEGMENTATION_ROOT_DIRECTORY_NAME_SUFFIX },
                                                          { "SegmentationFileExtension", Constants.SEGMENTATION_FILE_NAME_EXTENSION },

                                                          { "IdMap", Path.Combine( segmentationRootDirectory, Constants.ID_MAP_ROOT_DIRECTORY_NAME ) },
                                                          { "IdMapTiledVolumeDescription", Path.Combine( segmentationRootDirectory, Constants.ID_MAP_TILED_VOLUME_DESCRIPTION_NAME ) },
                                                          { "ColorMap", Path.Combine( segmentationRootDirectory, Constants.COLOR_MAP_PATH ) },
                                                          { "SegmentInfo", Path.Combine( segmentationRootDirectory, Constants.SEGMENT_INFO_PATH ) },
                                                          { "Log", Path.Combine( segmentationRootDirectory, Constants.LOG_PATH ) },

                                                          { "TempRoot", Path.Combine( segmentationRootDirectory, Constants.TEMP_ROOT_DIRECTORY_NAME ) },
                                                          { "TempIdMap", Path.Combine( segmentationRootDirectory, Constants.TEMP_ID_MAP_ROOT_DIRECTORY_NAME ) },
                                                          { "TempSegmentInfo", Path.Combine( segmentationRootDirectory, Constants.TEMP_SEGMENT_INFO_PATH ) },
                                                          { "TempColorMap", Path.Combine( segmentationRootDirectory, Constants.TEMP_COLOR_MAP_PATH ) },

                                                          { "AutosaveIdMap", Path.Combine( segmentationRootDirectory, Constants.AUTOSAVE_ID_MAP_ROOT_DIRECTORY_NAME ) },

                                                          { "IdMapRelativePath", Constants.ID_MAP_ROOT_DIRECTORY_NAME },
                                                          { "IdMapTiledVolumeDescriptionRelativePath", Constants.ID_MAP_TILED_VOLUME_DESCRIPTION_NAME },
                                                          { "ColorMapRelativePath", Constants.COLOR_MAP_PATH },
                                                          { "LogRelativePath", Constants.LOG_PATH },
                                                          { "SegmentInfoRelativePath", Constants.SEGMENT_INFO_PATH },

                                                          { "TempSegmentInfoRelativePath", Constants.TEMP_SEGMENT_INFO_PATH },
                                                      },
                                                  MaxLabelId = 0
                                              };

                LoadSegmentation( tiledDatasetDescription );

                SegmentationControlsEnabled = true;
                ChangesMade = false;

                UpdateView();

                SegmentationChangeInProgress = false;
            }
        }

        private static TiledVolumeDescription GetTiledVolumeDescription( string tiledVolumeDescriptionPath )
        {
            var tiledVolumeDescriptionXml = XmlReader.ReadFromFile< tiledVolumeDescription >( tiledVolumeDescriptionPath );
            var tiledVolumeDescription = new TiledVolumeDescription
                                         {
                                             FileExtension = tiledVolumeDescriptionXml.fileExtension,
                                             NumTilesX = tiledVolumeDescriptionXml.numTilesX,
                                             NumTilesY = tiledVolumeDescriptionXml.numTilesY,
                                             NumTilesZ = tiledVolumeDescriptionXml.numTilesZ,
                                             NumTilesW = tiledVolumeDescriptionXml.numTilesW,
                                             NumVoxelsPerTileX = tiledVolumeDescriptionXml.numVoxelsPerTileX,
                                             NumVoxelsPerTileY = tiledVolumeDescriptionXml.numVoxelsPerTileY,
                                             NumVoxelsPerTileZ = tiledVolumeDescriptionXml.numVoxelsPerTileZ,
                                             NumVoxelsX = tiledVolumeDescriptionXml.numVoxelsX,
                                             NumVoxelsY = tiledVolumeDescriptionXml.numVoxelsY,
                                             NumVoxelsZ = tiledVolumeDescriptionXml.numVoxelsZ,
                                             DxgiFormat = (Format)Enum.Parse( typeof( Format ), tiledVolumeDescriptionXml.dxgiFormat ),
                                             NumBytesPerVoxel = tiledVolumeDescriptionXml.numBytesPerVoxel,
                                             IsSigned = tiledVolumeDescriptionXml.isSigned
                                         };

            return tiledVolumeDescription;
        }

        public void LoadSourceImages( TiledDatasetDescription tiledDatasetDescription )
        {
            Release.Assert( Internal != null );
            Release.Assert( !SourceImagesLoaded );
            Release.Assert( !SegmentationLoaded );

            Internal.LoadSourceImages( tiledDatasetDescription );

            Release.Assert( SourceImagesLoaded );
            Release.Assert( !SegmentationLoaded );

            SourceImagesTiledDatasetDescription = tiledDatasetDescription;
            ChangesMade = false;
        }

        public void LoadSegmentation( TiledDatasetDescription tiledDatasetDescription )
        {
            Release.Assert( Internal != null );
            Release.Assert( SourceImagesLoaded );
            Release.Assert( !SegmentationLoaded );

            Internal.LoadSegmentation( tiledDatasetDescription );

            Release.Assert( SourceImagesLoaded );
            Release.Assert( SegmentationLoaded );

            SegmentationTiledDatasetDescription = tiledDatasetDescription;
            ChangesMade = false;
        }

        public void UnloadSourceImages()
        {
            UnloadSegmentation();

            if ( SourceImagesLoaded && Internal != null )
            {
                Internal.UnloadSourceImages();
                Release.Assert( !SourceImagesLoaded );
            }

            SourceImagesTiledDatasetDescription = new TiledDatasetDescription();
            NavigationControlsEnabled = false;
            ChangesMade = false;
        }

        public void UnloadSegmentation()
        {
            if ( SegmentationLoaded && Internal != null )
            {
                Internal.UnloadSegmentation();
                Release.Assert( !SegmentationLoaded );
            }

            MouseOverSegmentId = 0;
            SegmentationControlsEnabled = false;
            ChangesMade = false;
        }

        public void SaveSegmentation()
        {
            lock ( this )
            {
                Release.Assert( Internal != null );
                Release.Assert( SourceImagesLoaded );
                Release.Assert( SegmentationLoaded );

                SegmentationChangeProgress = 10;
                Internal.SaveSegmentation();
                SegmentationChangeProgress = 100;
                ChangesMade = false;
            }
        }

        public void SaveSegmentationAs( string savePath )
        {
            lock ( this )
            {
                Release.Assert( Internal != null );
                Release.Assert( SourceImagesLoaded );
                Release.Assert( SegmentationLoaded );

                SegmentationChangeProgress = 10;
                Internal.SaveSegmentationAs( savePath );
                SegmentationChangeProgress = 100;
                ChangesMade = false;
            }
        }

        public void AutoSaveSegmentation()
        {
            lock ( this )
            {
                Release.Assert( Internal != null );
                Release.Assert( SourceImagesLoaded );
                Release.Assert( SegmentationLoaded );

                SegmentationChangeProgress = 10;
                Internal.AutosaveSegmentation();
                SegmentationChangeProgress = 100;
            }
        }

        public void DiscardChanges()
        {
            if ( SegmentationChangeInProgress ) return;
            lock ( this )
            {
                if ( SegmentationChangeInProgress ) return;

                SegmentationChangeInProgress = true;

                Release.Assert( Internal != null );
                Release.Assert( SourceImagesLoaded );
                Release.Assert( SegmentationLoaded );

                Internal.DeleteTempFiles();

                SegmentationChangeInProgress = false;
            }
        }

        public IList<TileCacheEntry> GetTileCache()
        {
            return Internal.GetTileCache().Where( tileCacheEntry => tileCacheEntry.Active ).OrderBy( tileCacheEntry => tileCacheEntry.IndexTileSpace.W ).ToList();
        }

    }
}
