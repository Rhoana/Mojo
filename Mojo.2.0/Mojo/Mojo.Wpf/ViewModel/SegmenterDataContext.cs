using System;
using System.Collections.Generic;
using System.ComponentModel;
using Mojo.Interop;

namespace Mojo.Wpf.ViewModel
{
    public class SegmenterDataContext : NotifyPropertyChanged, IDisposable
    {
        private Segmenter mSegmenter;

        public string ToolbarString
        {
            get
            {
                if ( !mSegmenter.DatasetLoaded )
                {
                    return "No dataset loaded.";
                }

                switch ( mSegmenter.CurrentSegmenterToolMode )
                {
                    case SegmenterToolMode.Adjust:
                        return
                            mSegmenter.CurrentSegmentationLabel == null ?
                            "Left mouse button picks a segmentation label" :
                            "Current Segmentation Label: " + mSegmenter.CurrentSegmentationLabel.Name + " (" + mSegmenter.CurrentSegmentationLabel.Color + "). Left mouse button indicates foreground. Right mouse button indicates background.";

                    case SegmenterToolMode.Merge:
                        return
                            mSegmenter.MergeSourceSegmentationLabel == null ?
                            "Left mouse button picks source." :
                            "Merge Source: " + mSegmenter.MergeSourceSegmentationLabel.Name + " (" + mSegmenter.MergeSourceSegmentationLabel.Color + "). Right mouse button picks destination.";

                    case SegmenterToolMode.Split:
                        return
                            mSegmenter.SplitSegmentationLabel == null ?
                            "Left mouse button selects a segmentation label to split." :
                            "Segmentation label to split: " + mSegmenter.SplitSegmentationLabel.Name + " (" + mSegmenter.SplitSegmentationLabel.Color + "). Use mouse to paint within segmentation label.";

                    default:
                        Release.Assert( false );
                        return "";
                }
            } 
        }

        public bool EditMenuIsEnabled
        {
            get
            {
                return UndoLastCommitMenuItemIsEnabled || RedoLastCommitMenuItemIsEnabled;
            }
        }

        public bool UndoLastCommitMenuItemIsEnabled
        {
            get
            {
                return mSegmenter.DatasetLoaded && !mSegmenter.CommittedSegmentationEqualsUndoBuffer;
            }
        }

        public bool RedoLastCommitMenuItemIsEnabled
        {
            get
            {
                return mSegmenter.DatasetLoaded && !mSegmenter.CommittedSegmentationEqualsRedoBuffer && mSegmenter.CommittedSegmentationEqualsUndoBuffer;
            }
        }

        public bool AdjustSegmentationToolRadioButtonIsChecked
        {
            get
            {
                return mSegmenter.CurrentSegmenterToolMode == SegmenterToolMode.Adjust && mSegmenter.DatasetLoaded;
            }
            set
            {
                if ( value )
                {
                    if ( mSegmenter.CurrentSegmenterToolMode != SegmenterToolMode.Adjust )
                    {
                        mSegmenter.Internal.InitializeEdgeXYMap( mSegmenter.DatasetDescription.VolumeDescriptions );
                        mSegmenter.CommitSegmentation();
                    }

                    mSegmenter.CurrentSegmenterToolMode = SegmenterToolMode.Adjust;
                    Refresh();
                }
            }
        }

        public bool MergeSegmentationToolRadioButtonIsChecked
        {
            get
            {
                return mSegmenter.CurrentSegmenterToolMode == SegmenterToolMode.Merge && mSegmenter.DatasetLoaded;
            }
            set
            {

                if ( value )
                {
                    // Is this section of code necessary?
                    // Gives me an error on loading a dataset - Cmor.
                    //    if ( mSegmenter.CurrentSegmenterToolMode != SegmenterToolMode.Merge )
                    //{
                    //    mSegmenter.Internal.InitializeEdgeXYMap( mSegmenter.DatasetDescription.VolumeDescriptions );
                    //    mSegmenter.CommitSegmentation();
                    //}

                    mSegmenter.CurrentSegmenterToolMode = SegmenterToolMode.Merge;
                    Refresh();
                }
            }
        }

        public bool SplitSegmentationToolRadioButtonIsChecked
        {
            get
            {
                return mSegmenter.CurrentSegmenterToolMode == SegmenterToolMode.Split && mSegmenter.DatasetLoaded;
            }
            set
            {
                if ( value )
                {
                    if ( mSegmenter.CurrentSegmenterToolMode != SegmenterToolMode.Split )
                    {
                        mSegmenter.Internal.InitializeEdgeXYMap( mSegmenter.DatasetDescription.VolumeDescriptions );
                        mSegmenter.CommitSegmentation();
                    }

                    mSegmenter.CurrentSegmenterToolMode = SegmenterToolMode.Split;
                    Refresh();
                }
            }
        }

        public bool NotMergeModeAndDatasetLoaded
        {
            get
            {
                return mSegmenter.CurrentSegmenterToolMode != SegmenterToolMode.Merge;
            }
        }

        public bool MergeModeAndDatasetLoaded
        {
            get
            {
                return mSegmenter.CurrentSegmenterToolMode == SegmenterToolMode.Merge;
            }
        }

        public object CurrentSegmentationLabel
        {
            get
            {
                return mSegmenter.CurrentSegmentationLabel == null
                           ? (object)null
                           : new KeyValuePair<int, SegmentationLabelDescription>( mSegmenter.CurrentSegmentationLabel.Id, mSegmenter.CurrentSegmentationLabel );
            }
            set
            {
                if ( mSegmenter.CurrentSegmenterToolMode == SegmenterToolMode.Split )
                {
                    if ( mSegmenter.CurrentSegmentationLabel != null )
                    {
                        mSegmenter.Internal.UpdateCommittedSegmentationDoNotRemove( mSegmenter.CurrentSegmentationLabel.Id, mSegmenter.CurrentSegmentationLabel.Color );
                        mSegmenter.Internal.InitializeConstraintMap();
                        mSegmenter.Internal.InitializeSegmentation();
                        mSegmenter.Internal.InitializeEdgeXYMapForSplitting( mSegmenter.DatasetDescription.VolumeDescriptions, mSegmenter.SplitSegmentationLabel.Id );
                        mSegmenter.Internal.InitializeConstraintMapFromIdMapForSplitting( mSegmenter.SplitSegmentationLabel.Id );
                        mSegmenter.Internal.VisualUpdate();
                        mSegmenter.Internal.VisualUpdateColorMap();

                        mSegmenter.CommittedSegmentationEqualsUndoBuffer = false;
                        mSegmenter.CommittedSegmentationEqualsRedoBuffer = false;
                    }
                }

                mSegmenter.SelectSegmentationLabel( ( (KeyValuePair<int, SegmentationLabelDescription>)value ).Key );
                Refresh();
            }
        }

        public event EventHandler StateChanged;

        public SegmenterDataContext( Segmenter segmenter )
        {
            mSegmenter = segmenter;

            mSegmenter.PropertyChanged += PropertyChangedHandler;
            mSegmenter.Internal.PropertyChanged += PropertyChangedHandler;
        }

        public void Dispose()
        {
            if ( mSegmenter != null )
            {
                mSegmenter.Internal.PropertyChanged -= PropertyChangedHandler;
                mSegmenter.PropertyChanged -= PropertyChangedHandler;
                mSegmenter.Dispose();
                mSegmenter = null;
            }
        }

        public void Refresh()
        {
            OnPropertyChanged( "EditMenuIsEnabled" );
            OnPropertyChanged( "UndoLastCommitMenuItemIsEnabled" );
            OnPropertyChanged( "RedoLastCommitMenuItemIsEnabled" );
            OnPropertyChanged( "AdjustSegmentationToolRadioButtonIsChecked" );
            OnPropertyChanged( "MergeSegmentationToolRadioButtonIsChecked" );
            OnPropertyChanged( "SplitSegmentationToolRadioButtonIsChecked" );
            OnPropertyChanged( "MergeModeAndDatasetLoaded" );
            OnPropertyChanged( "NotMergeModeAndDatasetLoaded" );
            OnPropertyChanged( "CurrentSegmentationLabel" );
            OnPropertyChanged( "ToolbarString" );
        }

        protected void OnStateChanged( EventArgs e )
        {
            EventHandler handler = StateChanged;
            if ( handler != null ) handler( this, e );
        }

        private void PropertyChangedHandler( object sender, PropertyChangedEventArgs e )
        {
            OnStateChanged( e );
        }
    }
}
