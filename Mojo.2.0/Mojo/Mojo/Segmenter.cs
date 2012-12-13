using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows.Threading;
using Emgu.CV;
using Emgu.CV.Structure;
using Mojo.Interop;
using SlimDX;
using SlimDX.DXGI;
using SlimDX.Direct3D11;

namespace Mojo
{
    public enum ConstraintType
    {
        Foreground,
        Background
    }

    public enum InteractionMode
    {
        HighLatency,
        LowLatency
    }

    public enum DimensionMode
    {
        Two,
        Three
    }

    public enum InitializeCostMapMode
    {
        Idle,
        Initialize,
        Forward,
        Backward,
        Finalize
    }

    public enum SegmenterToolMode
    {
        Adjust,
        Merge,
        Split
    }

    public enum SaveSegmentationAsCommitPolicy
    {
        Commit,
        DoNotCommit
    }

    public class Segmenter : NotifyPropertyChanged, IDisposable
    {
        private const int NUM_BYTES_PER_COLOR_MAP_PIXEL = 4;
        private const int NUM_BYTES_PER_ID_MAP_PIXEL = 4;

        private readonly DispatcherTimer mAutoSaveTimer = new DispatcherTimer( DispatcherPriority.Input );
        private readonly Random mRandom = new Random();

        private string mAutoSavePath;
        private int mMousePreviousX;
        private int mMousePreviousY;

        public Interop.Segmenter Internal { get; private set; }

        public IEnumerator<KeyValuePair< int, SegmentationLabelDescription > > SegmentationLabelEnumerator { get; private set; }
        public IEnumerator<KeyValuePair< string, ShaderResourceView > > D3D11CudaTextureEnumerator { get; private set; }

        public InteractionMode InteractionMode { get; set; }
        public DimensionMode DimensionMode { get; set; }
        public InitializeCostMapMode InitializeCostMapMode { get; set; }

        public int CurrentSlice { get; set; }
        public int CurrentD3D11CudaTextureIndex { get; set; }
        public int CurrentBrushWidth { get; set; }

        private bool mDatasetLoaded;
        public bool DatasetLoaded
        {
            get
            {
                return mDatasetLoaded;
            }
            private set
            {
                mDatasetLoaded = value;
                OnPropertyChanged( "DatasetLoaded" );
            }
        }

        private DatasetDescription mDatasetDescription;
        public DatasetDescription DatasetDescription
        {
            get
            {
                return mDatasetDescription;
            }
            set
            {
                mDatasetDescription = value;
                OnPropertyChanged( "DatasetDescription" );
            }
        }

        private SegmenterToolMode mCurrentSegmenterToolMode;
        public SegmenterToolMode CurrentSegmenterToolMode
        {
            get
            {
                return mCurrentSegmenterToolMode;
            }
            set
            {
                mCurrentSegmenterToolMode = value;
                OnPropertyChanged( "CurrentSegmenterToolMode" );
            }
        }

        private SegmentationLabelDescription mCurrentSegmentationLabel;
        public SegmentationLabelDescription CurrentSegmentationLabel
        {
            get
            {
                return mCurrentSegmentationLabel;
            }
            set
            {
                mCurrentSegmentationLabel = value;
                OnPropertyChanged( "CurrentSegmentationLabel" );
            }
        }

        private SegmentationLabelDescription mMergeSourceSegmentationLabel;
        public SegmentationLabelDescription MergeSourceSegmentationLabel
        {
            get
            {
                return mMergeSourceSegmentationLabel;
            }
            set
            {
                mMergeSourceSegmentationLabel = value;
                OnPropertyChanged( "MergeSourceSegmentationLabel" );
            }
        }

        private SegmentationLabelDescription mMergeDestinationSegmentationLabel;
        public SegmentationLabelDescription MergeDestinationSegmentationLabel
        {
            get
            {
                return mMergeDestinationSegmentationLabel;
            }
            set
            {
                mMergeDestinationSegmentationLabel = value;
                OnPropertyChanged( "MergeDestinationSegmentationLabel" );
            }
        }

        private SegmentationLabelDescription mSplitSegmentationLabel;
        public SegmentationLabelDescription SplitSegmentationLabel
        {
            get
            {
                return mSplitSegmentationLabel;
            }
            set
            {
                mSplitSegmentationLabel = value;
                OnPropertyChanged( "SplitSegmentationLabel" );
            }
        }

        private bool mCommittedSegmentationEqualsUndoBuffer = true;
        public bool CommittedSegmentationEqualsUndoBuffer
        {
            get
            {
                return mCommittedSegmentationEqualsUndoBuffer;
            }
            set
            {
                mCommittedSegmentationEqualsUndoBuffer = value;
                OnPropertyChanged( "CommittedSegmentationEqualsUndoBuffer" );
            }
        }

        private bool mCommittedSegmentationEqualsRedoBuffer = true;
        public bool CommittedSegmentationEqualsRedoBuffer
        {
            get
            {
                return mCommittedSegmentationEqualsRedoBuffer;
            }
            set
            {
                mCommittedSegmentationEqualsRedoBuffer = value;
                OnPropertyChanged( "CommittedSegmentationEqualsRedoBuffer" );
            }
        }

        private bool mConstrainSegmentationMergeToCurrentSlice = true;
        public bool ConstrainSegmentationMergeToCurrentSlice
        {
            get
            {
                return mConstrainSegmentationMergeToCurrentSlice;
            }
            set
            {
                mConstrainSegmentationMergeToCurrentSlice = value;
                OnPropertyChanged( "ConstrainSegmentationMergeToCurrentSlice" );
            }
        }

        private bool mConstrainSegmentationMergeToConnectedComponent = true;
        public bool ConstrainSegmentationMergeToConnectedComponent
        {
            get
            {
                return mConstrainSegmentationMergeToConnectedComponent;
            }
            set
            {
                mConstrainSegmentationMergeToConnectedComponent = value;
                OnPropertyChanged( "ConstrainSegmentationMergeToConnectedComponent" );
            }
        }

        private bool mShowSegmentation = true;
        public bool ShowSegmentation
        {
            get
            {
                return mShowSegmentation;
            }
            set
            {
                mShowSegmentation = value;
                OnPropertyChanged( "ShowSegmentation" );
            }
        }


        public Segmenter( Interop.Segmenter segmenter )
        {
            Internal = segmenter;

            mAutoSaveTimer.Tick += AutoSave;
        }

        public void Dispose()
        {
            UnloadDataset();

            mAutoSaveTimer.Tick -= AutoSave;

            if ( Internal != null )
            {
                Internal.Dispose();
                Internal = null;
            }
        }

        public void LoadDataset( SegmenterImageStackLoadDescription segmenterImageStackLoadDescription )
        {
            var volumeDescriptions = SegmenterImageStackLoader.LoadDataset( segmenterImageStackLoadDescription );

            var segmentationLabelDescriptions = new ObservableDictionary<int, SegmentationLabelDescription>
                                            {
                                                { Constants.DEFAULT_SEGMENTATION_LABEL.Id, Constants.DEFAULT_SEGMENTATION_LABEL }
                                            };

            var datasetDescription = new DatasetDescription
            {
                SegmentationLabelDescriptions = segmentationLabelDescriptions,
                VolumeDescriptions = volumeDescriptions
            };

            LoadDataset( datasetDescription );
        }

        public void LoadDataset( DatasetDescription datasetDescription )
        {
            UnloadDataset();

            Internal.LoadVolume( datasetDescription.VolumeDescriptions );

            if ( Internal.IsVolumeLoaded() )
            {
                Internal.VisualUpdate();

                var segmentationLabelEnumerator = datasetDescription.SegmentationLabelDescriptions.GetEnumerator() as IEnumerator<KeyValuePair<int, SegmentationLabelDescription>>;
                segmentationLabelEnumerator.MoveNext();

                var d3d11CudaTextureEnumerator = Internal.GetD3D11CudaTextures().GetEnumerator() as IEnumerator<KeyValuePair<string, ShaderResourceView>>;
                d3d11CudaTextureEnumerator.MoveNext();

                DatasetDescription = datasetDescription;
                SegmentationLabelEnumerator = segmentationLabelEnumerator;
                D3D11CudaTextureEnumerator = d3d11CudaTextureEnumerator;
                CurrentSegmentationLabel = SegmentationLabelEnumerator.Current.Value;
                CurrentD3D11CudaTextureIndex = 0;
                CurrentSegmenterToolMode = SegmenterToolMode.Adjust;
                CommittedSegmentationEqualsUndoBuffer = true;
                CommittedSegmentationEqualsRedoBuffer = true;
                DatasetLoaded = true;

                D3D11CudaTextureEnumerator.MoveNext();
                CurrentD3D11CudaTextureIndex++;

                D3D11CudaTextureEnumerator.MoveNext();
                CurrentD3D11CudaTextureIndex++;
            }
        }

        public void UnloadDataset()
        {
            if ( DatasetLoaded )
            {
                Internal.UnloadVolume();

                DatasetDescription = new DatasetDescription();
                CurrentSegmenterToolMode = SegmenterToolMode.Adjust;
                DatasetLoaded = false;
            }
        }

        public void LoadSegmentation( SegmenterImageStackLoadDescription segmenterImageStackLoadDescription )
        {
            var volumeDescriptions = SegmenterImageStackLoader.LoadSegmentation( segmenterImageStackLoadDescription );

            Internal.LoadSegmentation( volumeDescriptions );
            Internal.VisualUpdate();
            Internal.VisualUpdateColorMap();

            volumeDescriptions.Get( "ColorMap" ).DataStream.Seek( 0, SeekOrigin.Begin );
            volumeDescriptions.Get( "IdMap" ).DataStream.Seek( 0, SeekOrigin.Begin );

            var uniqueIds = new Dictionary<int, Rgb>();

            while ( volumeDescriptions.Get( "ColorMap" ).DataStream.Position < volumeDescriptions.Get( "ColorMap" ).DataStream.Length &&
                    volumeDescriptions.Get( "IdMap" ).DataStream.Position < volumeDescriptions.Get( "IdMap" ).DataStream.Length )
            {
                var id = volumeDescriptions.Get( "IdMap" ).DataStream.Read<int>();
                var colorMapValue = volumeDescriptions.Get( "ColorMap" ).DataStream.Read<int>();

                var r = ( colorMapValue & ( 0x0000ff << 0 ) ) >> 0;
                var g = ( colorMapValue & ( 0x0000ff << 8 ) ) >> 8;
                var b = ( colorMapValue & ( 0x0000ff << 16 ) ) >> 16;

                if ( uniqueIds.ContainsKey( id ) )
                {
                    Release.Assert( uniqueIds[ id ].Equals( new Rgb( r, g, b ) ) );
                }

                uniqueIds[ id ] = new Rgb( r, g, b );
            }

            uniqueIds.Remove( Constants.NULL_SEGMENTATION_LABEL.Id );

            uniqueIds.ToList().ForEach( keyValuePair =>
                                        DatasetDescription.SegmentationLabelDescriptions.Add( keyValuePair.Key,
                                                                                          new SegmentationLabelDescription( keyValuePair.Key )
                                                                                          {
                                                                                              Name = "Autogenerated Segmentation Label (ID " + keyValuePair.Key + ") " + keyValuePair.Value,
                                                                                              Color =
                                                                                                  new Vector3( (float)keyValuePair.Value.Red,
                                                                                                               (float)keyValuePair.Value.Green,
                                                                                                               (float)keyValuePair.Value.Blue ),
                                                                                          } ) );

            CurrentSegmentationLabel = null;
            MergeSourceSegmentationLabel = null;
            MergeDestinationSegmentationLabel = null;
            SplitSegmentationLabel = null;
            CommittedSegmentationEqualsUndoBuffer = false;
            CommittedSegmentationEqualsRedoBuffer = false;
        }

        public void SaveSegmentationAs( SegmenterImageStackSaveDescription segmenterImageStackSaveDescription, SaveSegmentationAsCommitPolicy doNotCommit )
        {
            CommitSegmentation();

            var colorMapDataStream =
                new DataStream(
                    Internal.GetVolumeDescription().NumVoxelsX * Internal.GetVolumeDescription().NumVoxelsY * Internal.GetVolumeDescription().NumVoxelsZ * NUM_BYTES_PER_COLOR_MAP_PIXEL,
                    true,
                    true );
            var idMapDataStream =
                new DataStream(
                    Internal.GetVolumeDescription().NumVoxelsX * Internal.GetVolumeDescription().NumVoxelsY * Internal.GetVolumeDescription().NumVoxelsZ * NUM_BYTES_PER_COLOR_MAP_PIXEL,
                    true,
                    true );

            var volumeDescriptions = new ObservableDictionary<string, VolumeDescription>
                                     {
                                         {
                                             "ColorMap", new VolumeDescription
                                                         {
                                                             DxgiFormat = Format.R8G8B8A8_UNorm,
                                                             DataStream = colorMapDataStream,
                                                             Data = colorMapDataStream.DataPointer,
                                                             NumBytesPerVoxel = NUM_BYTES_PER_COLOR_MAP_PIXEL,
                                                             NumVoxelsX = Internal.GetVolumeDescription().NumVoxelsX,
                                                             NumVoxelsY = Internal.GetVolumeDescription().NumVoxelsY,
                                                             NumVoxelsZ = Internal.GetVolumeDescription().NumVoxelsZ,
                                                             IsSigned = false
                                                         }
                                             },
                                         {
                                             "IdMap", new VolumeDescription
                                                      {
                                                          DxgiFormat = Format.R32_UInt,
                                                          DataStream = idMapDataStream,
                                                          Data = idMapDataStream.DataPointer,
                                                          NumBytesPerVoxel = NUM_BYTES_PER_ID_MAP_PIXEL,
                                                          NumVoxelsX = Internal.GetVolumeDescription().NumVoxelsX,
                                                          NumVoxelsY = Internal.GetVolumeDescription().NumVoxelsY,
                                                          NumVoxelsZ = Internal.GetVolumeDescription().NumVoxelsZ,
                                                          IsSigned = false
                                                      }
                                             },
                                     };

            Internal.SaveSegmentationAs( volumeDescriptions );

            segmenterImageStackSaveDescription.VolumeDescriptions = volumeDescriptions;

            SegmenterImageStackLoader.SaveSegmentation( segmenterImageStackSaveDescription );
        }

        public void AddSegmentationLabel( string segmentationLabelName )
        {
            var trimmedSegmentationLabelName = segmentationLabelName.Trim();
            var tmpSegmentationLabelName = trimmedSegmentationLabelName;

            if ( DatasetDescription.SegmentationLabelDescriptions.Internal.Any( segmentationLabelDescription => segmentationLabelDescription.Value.Name.Equals( tmpSegmentationLabelName ) ) )
            {
                Console.WriteLine( "The name " + trimmedSegmentationLabelName + " is already being used" );
                return;
            }

            var segmentationLabelId = DatasetDescription.SegmentationLabelDescriptions.Internal.Keys.Max() + 1;
            var randomColor = GetRandomColor();

            if ( trimmedSegmentationLabelName.Length == 0 )
            {
                tmpSegmentationLabelName = "Autogenerated Segmentation Label (ID " + segmentationLabelId + ") " + randomColor;
            }

            if ( CurrentSegmenterToolMode == SegmenterToolMode.Split )
            {
                if ( CurrentSegmentationLabel != null )
                {
                    Internal.UpdateCommittedSegmentationDoNotRemove( CurrentSegmentationLabel.Id, CurrentSegmentationLabel.Color );
                    Internal.InitializeSegmentation();
                    Internal.InitializeConstraintMap();
                    Internal.InitializeEdgeXYMapForSplitting( DatasetDescription.VolumeDescriptions, SplitSegmentationLabel.Id );
                    Internal.InitializeConstraintMapFromIdMapForSplitting( SplitSegmentationLabel.Id );
                    Internal.VisualUpdate();
                    Internal.VisualUpdateColorMap();

                    CommittedSegmentationEqualsUndoBuffer = false;
                    CommittedSegmentationEqualsRedoBuffer = false;
                }
            }
            else
            {
                CommitSegmentation();
            }

            DatasetDescription.SegmentationLabelDescriptions.Add( segmentationLabelId,
                                           new SegmentationLabelDescription( segmentationLabelId )
                                           {
                                               Name = tmpSegmentationLabelName,
                                               Color = new Vector3( (float)randomColor.Red, (float)randomColor.Green, (float)randomColor.Blue ),
                                           } );

            SelectSegmentationLabel( segmentationLabelId );
        }

        public void RemoveSegmentationLabel( string segmentationLabelName )
        {
            if ( segmentationLabelName != Constants.DEFAULT_SEGMENTATION_LABEL.Name )
            {
                try
                {
                    var segmentationLabel = ( from segmentationLabelDescription in DatasetDescription.SegmentationLabelDescriptions.Internal.Values
                                          where segmentationLabelDescription.Name.Equals( segmentationLabelName )
                                          select segmentationLabelDescription ).First();

                    SelectSegmentationLabel( segmentationLabel.Id );
                    ClearSegmentationAndCostMap();

                    CurrentSegmentationLabel = null;
                    MergeSourceSegmentationLabel = null;
                    MergeDestinationSegmentationLabel = null;
                    SplitSegmentationLabel = null;
                    CommittedSegmentationEqualsUndoBuffer = true;
                    CommittedSegmentationEqualsRedoBuffer = true;

                    DatasetDescription.SegmentationLabelDescriptions.Internal.Remove( segmentationLabel.Id );
                }
                catch
                {
                    Console.WriteLine( "There is no segmentation label with the name " + segmentationLabelName );
                }
            }
        }

        public void Update()
        {
            if ( DatasetLoaded )
            {
                if ( InitializeCostMapMode != InitializeCostMapMode.Idle )
                {
                    switch ( InitializeCostMapMode )
                    {
                        case InitializeCostMapMode.Initialize:
                            Internal.InitializeCostMapFromPrimalMap();
                            InitializeCostMapMode = InitializeCostMapMode.Forward;
                            break;

                        case InitializeCostMapMode.Forward:
                            Internal.IncrementCostMapFromPrimalMapForward();
                            InitializeCostMapMode = InitializeCostMapMode.Backward;
                            break;

                        case InitializeCostMapMode.Backward:
                            Internal.IncrementCostMapFromPrimalMapBackward();
                            InitializeCostMapMode = InitializeCostMapMode.Finalize;
                            break;

                        case InitializeCostMapMode.Finalize:
                            Internal.FinalizeCostMapFromPrimalMap();
                            InitializeCostMapMode = InitializeCostMapMode.Idle;
                            break;
                    }
                }
                else if ( Internal.GetConvergenceGap() > Constants.CONVERGENCE_GAP_THRESHOLD &&
                            Internal.GetConvergenceGapDelta() > Constants.CONVERGENCE_DELTA_THRESHOLD )
                {
                    if ( DimensionMode == DimensionMode.Two )
                    {
                        if ( !Constants.ConstParameters.GetBool( "DIRECT_SCRIBBLE_PROPAGATION" ) )
                        {
                            Internal.Update2D( InteractionMode == InteractionMode.HighLatency
                                                                ? Constants.NUM_ITERATIONS_PER_VISUAL_UPDATE_HIGH_LATENCY_2D
                                                                : Constants.NUM_ITERATIONS_PER_VISUAL_UPDATE_LOW_LATENCY_2D,
                                                            CurrentSlice );
                        }
                    }
                    else
                    {
                        Internal.Update3D( InteractionMode == InteractionMode.HighLatency
                                                            ? Constants.NUM_ITERATIONS_PER_VISUAL_UPDATE_HIGH_LATENCY_3D
                                                            : Constants.NUM_ITERATIONS_PER_VISUAL_UPDATE_LOW_LATENCY_3D );
                    }

                    Internal.VisualUpdate();
                }
            }
        }

        public void InitializeSegmentation2D()
        {
            DimensionMode = Constants.ConstParameters.GetBool( "DIRECT_ANISOTROPIC_TV" ) ? DimensionMode.Three : DimensionMode.Two;
            Internal.SetConvergenceGap( Constants.ConstParameters.GetFloat( "MAX_CONVERGENCE_GAP" ) );
            Internal.SetConvergenceGapDelta( Constants.ConstParameters.GetFloat( "MAX_CONVERGENCE_GAP_DELTA" ) );
        }

        public void InitializeSegmentation3D()
        {
            Internal.InitializeConstraintMapFromPrimalMap();
            Internal.UpdateConstraintMapAndPrimalMapFromCostMap();
            Internal.VisualUpdate();

            DimensionMode = DimensionMode.Three;
            Internal.SetConvergenceGap( Constants.ConstParameters.GetFloat( "MAX_CONVERGENCE_GAP" ) );
            Internal.SetConvergenceGapDelta( Constants.ConstParameters.GetFloat( "MAX_CONVERGENCE_GAP_DELTA" ) );
        }

        public void InitializeCostMap()
        {
            InitializeCostMapMode = InitializeCostMapMode.Initialize;
        }

        public void IncrementMaxForegroundCostDelta()
        {
            Internal.SetMaxForegroundCostDelta( Internal.GetMaxForegroundCostDelta() + 5 );
            Console.WriteLine( "MaxForegroundCostDelta = {0}", Internal.GetMaxForegroundCostDelta() );
        }

        public void DecrementMaxForegroundCostDelta()
        {
            Internal.SetMaxForegroundCostDelta( Internal.GetMaxForegroundCostDelta() - 5 );
            Console.WriteLine( "MaxForegroundCostDelta = {0}", Internal.GetMaxForegroundCostDelta() );
        }

        public void CommitSegmentation()
        {
            if ( CurrentSegmentationLabel != null )
            {
                if ( CurrentSegmenterToolMode == SegmenterToolMode.Split )
                {
                    Internal.UpdateCommittedSegmentationDoNotRemove( CurrentSegmentationLabel.Id, CurrentSegmentationLabel.Color );
                }
                else
                {
                    Internal.UpdateCommittedSegmentation( CurrentSegmentationLabel.Id, CurrentSegmentationLabel.Color );
                }
            }

            Internal.InitializeConstraintMap();
            Internal.InitializeSegmentation();
            Internal.VisualUpdate();
            Internal.VisualUpdateColorMap();

            Internal.SetConvergenceGap( 0 );
            Internal.SetConvergenceGapDelta( 0 );

            DimensionMode = DimensionMode.Two;
            CurrentSegmentationLabel = null;
            MergeSourceSegmentationLabel = null;
            MergeDestinationSegmentationLabel = null;
            SplitSegmentationLabel = null;
            CommittedSegmentationEqualsUndoBuffer = false;
            CommittedSegmentationEqualsRedoBuffer = false;
        }

        public void CancelSegmentation()
        {
            Internal.InitializeCostMap();
            Internal.InitializeConstraintMap();
            Internal.InitializeSegmentation();
            Internal.VisualUpdate();

            RedoLastCommit();
        }

        public void UndoLastCommit()
        {
            Internal.InitializeCostMap();
            Internal.InitializeConstraintMap();
            Internal.InitializeSegmentation();
            Internal.RedoLastChangeToCommittedSegmentation();
            Internal.UndoLastChangeToCommittedSegmentation();
            Internal.VisualUpdate();
            Internal.VisualUpdateColorMap();

            DimensionMode = DimensionMode.Two;
            CurrentSegmentationLabel = null;
            MergeSourceSegmentationLabel = null;
            MergeDestinationSegmentationLabel = null;
            SplitSegmentationLabel = null;

            CommittedSegmentationEqualsUndoBuffer = true;
            CommittedSegmentationEqualsRedoBuffer = false;
        }

        public void RedoLastCommit()
        {
            Internal.InitializeCostMap();
            Internal.InitializeConstraintMap();
            Internal.InitializeSegmentation();
            Internal.RedoLastChangeToCommittedSegmentation();
            Internal.VisualUpdate();
            Internal.VisualUpdateColorMap();

            DimensionMode = DimensionMode.Two;
            CurrentSegmentationLabel = null;
            MergeSourceSegmentationLabel = null;
            MergeDestinationSegmentationLabel = null;
            SplitSegmentationLabel = null;

            CommittedSegmentationEqualsUndoBuffer = false;
            CommittedSegmentationEqualsRedoBuffer = true;
        }

        public void ClearSegmentationAndCostMap()
        {
            Internal.InitializeCostMap();
            Internal.InitializeConstraintMap();
            Internal.InitializeSegmentation();

            if ( CurrentSegmenterToolMode == SegmenterToolMode.Split )
            {
                Internal.InitializeEdgeXYMapForSplitting( DatasetDescription.VolumeDescriptions, SplitSegmentationLabel.Id );
                Internal.InitializeConstraintMapFromIdMapForSplitting( SplitSegmentationLabel.Id );
            }
            else
            {
                InitializeNewSegmentationLabel();
            }

            Internal.VisualUpdate();

            DimensionMode = DimensionMode.Two;
            Internal.SetConvergenceGap( 0 );
            Internal.SetConvergenceGapDelta( 0 );
        }

        public void ClearSegmentation()
        {
            Internal.InitializeConstraintMap();
            Internal.InitializeSegmentation();

            InitializeNewSegmentationLabel();

            Internal.VisualUpdate();

            DimensionMode = DimensionMode.Two;
            Internal.SetConvergenceGap( 0 );
            Internal.SetConvergenceGapDelta( 0 );
        }

        public void IncrementCurrentSlice()
        {
            if ( Internal.GetVolumeDescription().NumVoxelsZ > 1 )
            {
                CurrentSlice++;
                if ( CurrentSlice > Internal.GetVolumeDescription().NumVoxelsZ - 1 )
                {
                    CurrentSlice = Internal.GetVolumeDescription().NumVoxelsZ - 1;
                }
            }
        }

        public void DecrementCurrentSlice()
        {
            if ( Internal.GetVolumeDescription().NumVoxelsZ > 1 )
            {
                CurrentSlice--;
                if ( CurrentSlice < 0 )
                {
                    CurrentSlice = 0;
                }
            }
        }

        public void IncrementCurrentTexture()
        {
            CurrentD3D11CudaTextureIndex++;
            D3D11CudaTextureEnumerator.MoveNext();

            if ( CurrentD3D11CudaTextureIndex > Internal.GetD3D11CudaTextures().Internal.Count - 1 )
            {
                CurrentD3D11CudaTextureIndex = 0;
                D3D11CudaTextureEnumerator.Reset();
                D3D11CudaTextureEnumerator.MoveNext();
            }
        }

        public void DecrementCurrentTexture()
        {
            CurrentD3D11CudaTextureIndex--;

            if ( CurrentD3D11CudaTextureIndex < 0 )
            {
                CurrentD3D11CudaTextureIndex = Internal.GetD3D11CudaTextures().Internal.Count - 1;
            }

            D3D11CudaTextureEnumerator.Reset();
            D3D11CudaTextureEnumerator.MoveNext();

            for ( var i = 0; i < CurrentD3D11CudaTextureIndex; i++ )
            {
                D3D11CudaTextureEnumerator.MoveNext();
            }
        }

        public void ToggleShowSegmentation()
        {
            ShowSegmentation = !ShowSegmentation;
        }

        public void BeginScribble( int x, int y )
        {
            mMousePreviousX = x;
            mMousePreviousY = y;

            InteractionMode = InteractionMode.LowLatency;

            if ( !Constants.ConstParameters.GetBool( "DIRECT_ANISOTROPIC_TV" ) )
            {
                DimensionMode = DimensionMode.Two;
            }
        }

        public void EndScribble()
        {
            InteractionMode = InteractionMode.HighLatency;
        }

        public void SelectSegmentationLabelOrScribble( int x, int y, ConstraintType constraintType, int brushWidth )
        {
            mMousePreviousX = x;
            mMousePreviousY = y;

            var newSegmentationLabelId = Internal.GetSegmentationLabelId( new Vector3( x, y, CurrentSlice ) );

            if ( newSegmentationLabelId != Constants.NULL_SEGMENTATION_LABEL.Id )
            {
                SelectSegmentationLabel( newSegmentationLabelId );
            }
            else
            {
                Scribble( x, y, constraintType, brushWidth );
            }
        }

        public void Scribble( int x, int y, ConstraintType constraintType, int brushWidth )
        {
            if ( CurrentSegmentationLabel != null )
            {
                Internal.SetConvergenceGap( Constants.ConstParameters.GetFloat( "MAX_CONVERGENCE_GAP" ) );
                Internal.SetConvergenceGapDelta( Constants.ConstParameters.GetFloat( "MAX_CONVERGENCE_GAP_DELTA" ) );

                switch ( constraintType )
                {
                    case ConstraintType.Foreground:
                        Internal.AddForegroundHardConstraint(
                            new Vector3( x, y, CurrentSlice ),
                            new Vector3( mMousePreviousX, mMousePreviousY, CurrentSlice ),
                            brushWidth );
                        break;

                    case ConstraintType.Background:
                        Internal.AddBackgroundHardConstraint(
                            new Vector3( x, y, CurrentSlice ),
                            new Vector3( mMousePreviousX, mMousePreviousY, CurrentSlice ),
                            brushWidth );
                        break;

                    default:
                        Release.Assert( false );
                        break;
                }

                mMousePreviousX = x;
                mMousePreviousY = y;
            }
        }

        public void SelectSegmentationLabel( int id )
        {
            switch ( CurrentSegmenterToolMode )
            {
                case SegmenterToolMode.Adjust:

                    SaveOldSegmentationLabel();
                    CurrentSegmentationLabel = DatasetDescription.SegmentationLabelDescriptions.Get( id );
                    InitializeNewSegmentationLabel();
                    break;

                case SegmenterToolMode.Merge:
                    Release.Assert( false );
                    break;

                case SegmenterToolMode.Split:
                    CurrentSegmentationLabel = DatasetDescription.SegmentationLabelDescriptions.Get( id );
                    break;

                default:
                    Release.Assert( false );
                    break;
            }
        }

        public void SelectMergeSourceSegmentationLabel( int x, int y )
        {
            var segmentationLabelId = Internal.GetSegmentationLabelId( new Vector3( x, y, CurrentSlice ) );

            if ( segmentationLabelId != Constants.NULL_SEGMENTATION_LABEL.Id )
            {
                MergeSourceSegmentationLabel = DatasetDescription.SegmentationLabelDescriptions.Get( segmentationLabelId );
            }
        }

        public void SelectMergeDestinationSegmentationLabel( int x, int y )
        {
            var clickCoordinates = new Vector3( x, y, CurrentSlice );
            var segmentationLabelId = Internal.GetSegmentationLabelId( clickCoordinates );

            if ( segmentationLabelId != Constants.NULL_SEGMENTATION_LABEL.Id )
            {
                MergeDestinationSegmentationLabel = DatasetDescription.SegmentationLabelDescriptions.Get( segmentationLabelId );

                if ( MergeSourceSegmentationLabel != null && MergeDestinationSegmentationLabel != null )
                {
                    if ( ConstrainSegmentationMergeToCurrentSlice && ConstrainSegmentationMergeToConnectedComponent )
                    {
                        Internal.ReplaceSegmentationLabelInCommittedSegmentation2DConnectedComponentOnly( MergeDestinationSegmentationLabel.Id, MergeSourceSegmentationLabel.Id, MergeSourceSegmentationLabel.Color, CurrentSlice, clickCoordinates );
                    }
                    else
                        if ( ConstrainSegmentationMergeToCurrentSlice )
                        {
                            Internal.ReplaceSegmentationLabelInCommittedSegmentation2D( MergeDestinationSegmentationLabel.Id, MergeSourceSegmentationLabel.Id, MergeSourceSegmentationLabel.Color, CurrentSlice );
                        }
                        else
                            if ( ConstrainSegmentationMergeToConnectedComponent )
                            {
                                Internal.ReplaceSegmentationLabelInCommittedSegmentation3DConnectedComponentOnly( MergeDestinationSegmentationLabel.Id, MergeSourceSegmentationLabel.Id, MergeSourceSegmentationLabel.Color, clickCoordinates );
                            }
                            else
                            {
                                Internal.ReplaceSegmentationLabelInCommittedSegmentation3D( MergeDestinationSegmentationLabel.Id, MergeSourceSegmentationLabel.Id, MergeSourceSegmentationLabel.Color );
                            }

                    Internal.VisualUpdateColorMap();

                    CommittedSegmentationEqualsUndoBuffer = false;
                    CommittedSegmentationEqualsRedoBuffer = true;
                }
            }
        }

        public void SelectSplitSegmentationLabelOrScribble( int x, int y, ConstraintType constraintType, int brushWidth )
        {
            if ( SplitSegmentationLabel == null )
            {
                var segmentationLabelId = Internal.GetSegmentationLabelId( new Vector3( x, y, CurrentSlice ) );

                if ( segmentationLabelId != Constants.NULL_SEGMENTATION_LABEL.Id )
                {
                    SplitSegmentationLabel = DatasetDescription.SegmentationLabelDescriptions.Get( segmentationLabelId );

                    if ( SplitSegmentationLabel != null )
                    {
                        InitializeSplit();
                    }
                }
            }
            else
            {
                Scribble( x, y, constraintType, brushWidth );
            }
        }

        public void EnableAutoSave( int autoSaveSegmentationFrequencySeconds, string autoSaveSegmentationPath )
        {
            mAutoSavePath = autoSaveSegmentationPath;
            mAutoSaveTimer.Interval = TimeSpan.FromSeconds( autoSaveSegmentationFrequencySeconds );

            mAutoSaveTimer.Start();

            Console.WriteLine( "Auto-saving turned on. Auto-saving every {0} seconds.", autoSaveSegmentationFrequencySeconds );
        }

        public void DisableAutoSave()
        {
            mAutoSaveTimer.Stop();

            Console.WriteLine( "Auto-saving turned off." );
        }

        public void AutoSave( object sender, EventArgs eventArgs )
        {
            if ( DatasetLoaded )
            {
                var dateTimeString = String.Format( "{0:s}", DateTime.Now ).Replace( ':', '-' );

                Console.WriteLine( "Auto-saving segmentation: " + dateTimeString );

                var segmenterImageStackSaveDescription = new SegmenterImageStackSaveDescription
                                                         {
                                                             Directories = new ObservableDictionary<string, string>
                                                                               {
                                                                                   { "ColorMap", Directory.GetCurrentDirectory() + @"\" + mAutoSavePath + @"\" + dateTimeString + @"\Colors" },
                                                                                   { "IdMap", Directory.GetCurrentDirectory() + @"\" + mAutoSavePath + @"\" + dateTimeString + @"\Ids" }
                                                                               },
                                                         };

                SaveSegmentationAs( segmenterImageStackSaveDescription, SaveSegmentationAsCommitPolicy.DoNotCommit );
            }
        }

        void InitializeNewSegmentationLabel()
        {
            if ( CurrentSegmentationLabel != null )
            {
                Internal.InitializeConstraintMap();
                Internal.InitializeConstraintMapFromIdMap( CurrentSegmentationLabel.Id );

                for ( var i = 0; i < Constants.NUM_CONSTRAINT_MAP_DILATION_PASSES_INITIALIZE_NEW_PROCESS; i++ )
                {
                    Internal.VisualUpdate();
                    Internal.DilateConstraintMap();
                }

                Internal.InitializeSegmentationAndRemoveFromCommittedSegmentation( CurrentSegmentationLabel.Id );
                Internal.VisualUpdateColorMap();
                Internal.VisualUpdate();

                CommittedSegmentationEqualsUndoBuffer = false;
                CommittedSegmentationEqualsRedoBuffer = false;
            }
        }

        void SaveOldSegmentationLabel()
        {
            if ( CurrentSegmentationLabel != null )
            {
                if ( CurrentSegmenterToolMode == SegmenterToolMode.Split )
                {
                    Internal.UpdateCommittedSegmentationDoNotRemove( CurrentSegmentationLabel.Id, CurrentSegmentationLabel.Color );
                }
                else
                {
                    Internal.UpdateCommittedSegmentation( CurrentSegmentationLabel.Id, CurrentSegmentationLabel.Color );
                }
                Internal.VisualUpdateColorMap();

                CommittedSegmentationEqualsUndoBuffer = false;
                CommittedSegmentationEqualsRedoBuffer = false;
            }
        }

        void InitializeSplit()
        {
            if ( SplitSegmentationLabel != null )
            {
                Internal.InitializeConstraintMap();
                Internal.InitializeEdgeXYMapForSplitting( DatasetDescription.VolumeDescriptions, SplitSegmentationLabel.Id );
                Internal.InitializeConstraintMapFromIdMapForSplitting( SplitSegmentationLabel.Id );
                Internal.InitializeSegmentation();
                Internal.VisualUpdate();
                Internal.VisualUpdateColorMap();

                CommittedSegmentationEqualsUndoBuffer = false;
                CommittedSegmentationEqualsRedoBuffer = false;
            }
        }

        Rgb GetRandomColor()
        {
            var hsvImage = new Image<Hsv, byte>( 1, 1 );
            hsvImage[ 0, 0 ] = new Hsv( mRandom.Next( 0, 179 ), 255, 255 );

            var bmp = hsvImage.ToBitmap();

            var rgbImage = new Image<Rgb, byte>( bmp );
            return rgbImage[ 0, 0 ];
        }
    }
}
