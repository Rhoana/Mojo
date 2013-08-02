using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    public class SplitSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private readonly Engine mEngine;

        private bool mCurrentlyDrawing = false;
        private bool mCurrentlyMovingZ = false;

        public SplitSegmentationTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
            mEngine = engine;
        }

        public override void Select()
        {
            if ( mTileManager.SelectedSegmentId != 0 )
            {
                var p = new Vector2( 0.5f, 0.5f );
                mTileManager.PrepForSplit( p );
            }
        }

        public override void SelectSegment( uint segmentId )
        {
            if ( mTileManager.SelectedSegmentId != segmentId )
            {
                mTileManager.SelectedSegmentId = segmentId;
                var p = new Vector2( 0.5f, 0.5f );
                mTileManager.PrepForSplit( p );
            }
        }

        public override void MoveZ()
        {
            //
            // Z index has changed - prep for splitting
            //
            if ( mTileManager.SelectedSegmentId != 0 && !mCurrentlyMovingZ )
            {
                var p = new Vector2( 0.5f, 0.5f );
                mTileManager.PrepForSplit( p );
                if ( mTileManager.JoinSplits3D )
                {
                    mTileManager.PredictSplit( p );
                }
            }
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {

            var p = new Vector2( 0.5f, 0.5f );

            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.W:
                    mCurrentlyMovingZ = true;
                    break;
                case System.Windows.Input.Key.S:
                    mCurrentlyMovingZ = true;
                    break;
                case System.Windows.Input.Key.Q:
                    mTileManager.ToggleShowBoundaryLines();
                    break;
                case System.Windows.Input.Key.A:
                    mTileManager.ToggleShowSegmentation();
                    break;
                case System.Windows.Input.Key.E:
                    mTileManager.IncreaseSegmentationVisibility();
                    break;
                case System.Windows.Input.Key.D:
                    mTileManager.DecreaseSegmentationVisibility();
                    break;
                case System.Windows.Input.Key.Z:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.UndoChange();
                        mTileManager.PrepForSplit( p );
                    }
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.RedoChange();
                        mTileManager.PrepForSplit( p );
                    }
                    break;
                case System.Windows.Input.Key.N:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.SelectNewId();
                        mTileManager.PrepForSplit( p );
                    }
                    break;
                case System.Windows.Input.Key.Tab:
                    mTileManager.CommmitSplitChange();
                    keyEventArgs.Handled = true;
                    break;

                case System.Windows.Input.Key.Escape:
                    mTileManager.CancelSplitChange();
                    break;

                case System.Windows.Input.Key.OemComma:
                case System.Windows.Input.Key.OemMinus:
                case System.Windows.Input.Key.Subtract:
                    mTileManager.DecreaseBrushSize();
                    break;
                case System.Windows.Input.Key.OemPeriod:
                case System.Windows.Input.Key.OemPlus:
                case System.Windows.Input.Key.Add:
                    mTileManager.IncreaseBrushSize();
                    break;

            }

            base.OnKeyDown( keyEventArgs, width, height );

        }

        public override void OnKeyUp(System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height)
        {
            base.OnKeyUp(keyEventArgs, width, height);
            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.W:
                    mCurrentlyMovingZ = false;
                    MoveZ();
                    break;
                case System.Windows.Input.Key.S:
                    mCurrentlyMovingZ = false;
                    MoveZ();
                    break;
            }
        }

        public override void OnMouseDown( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseDown( mouseEventArgs, width, height );
            if ( mTileManager.CurrentSplitMode != SplitMode.JoinPoints )
            {
                if ( mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress )
                {

                    //Get the id of the segment being clicked

                    var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                    var clickedId = mTileManager.GetSegmentationLabelId( p );

                    if ( clickedId == mTileManager.SelectedSegmentId && mTileManager.CurrentSplitMode != SplitMode.JoinPoints )
                    {
                        mCurrentlyDrawing = true;

                        if ( mTileManager.CurrentSplitMode == SplitMode.DrawSplit )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.DrawSplit( p );
                                mEngine.QuickRender();
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.DrawErase( p );
                                mEngine.QuickRender();
                            }
                        }
                        else if ( mTileManager.CurrentSplitMode == SplitMode.DrawRegions )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.DrawRegionA( p );
                                mEngine.QuickRender();
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.DrawRegionB( p );
                                mEngine.QuickRender();
                            }
                        }
                    }
                }
            }
        }

        public override void OnMouseUp( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseUp( mouseEventArgs, width, height );

            if ( mCurrentlyDrawing )
            {
                mCurrentlyDrawing = false;

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                if ( mTileManager.CurrentSplitMode == SplitMode.DrawSplit )
                {
                    mTileManager.FindBoundaryWithinRegion2D( p );
                }
                if ( mTileManager.CurrentSplitMode == SplitMode.DrawRegions )
                {
                    mTileManager.FindBoundaryBetweenRegions2D( p );
                }
            }
            mCurrentlyDrawing = false;
        }

        public override void OnMouseClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress && !mCurrentlyDrawing )
            {
                //Get the id of the segment being clicked

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                var clickedId = mTileManager.GetSegmentationLabelId( p );

                if ( mouseEventArgs.Button == MouseButtons.Left )
                {
                    //Select this segment
                    if ( clickedId > 0 )
                    {
                        if ( clickedId != mTileManager.SelectedSegmentId )
                        {
                            //Select this segment
                            mTileManager.SelectedSegmentId = clickedId;
                            mEngine.UpdateExternalViewerLocation( mTileManager.GetPointDataSpace( p ) );
                            mTileManager.PrepForSplit( p );
                        }
                    }
                }
                else if ( mouseEventArgs.Button == MouseButtons.Right )
                {
                    if ( clickedId > 0 && clickedId == mTileManager.SelectedSegmentId && mTileManager.CurrentSplitMode == SplitMode.JoinPoints )
                    {
                        mTileManager.AddSplitSource( p );
                        mTileManager.DrawRegionB( p, 4 );
                        mTileManager.FindBoundaryJoinPoints2D( p );
                    }
                    else if ( clickedId > 0 && mTileManager.SelectedSegmentId > 0 && clickedId != mTileManager.SelectedSegmentId )
                    {
                        //
                        // Merge with the clicked segment in 2D ( try this out to see if it makes sense )
                        //
                        switch ( mTileManager.CurrentMergeMode )
                        {
                            case MergeMode.Fill2D:
                                mTileManager.ReplaceSegmentationLabelCurrentSlice( clickedId, p );
                                break;
                            case MergeMode.Fill3D:
                                mTileManager.ReplaceSegmentationLabelCurrentConnectedComponent( clickedId, p );
                                break;
                            default:
                                mTileManager.RemapSegmentLabel( clickedId );
                                break;
                        }
                        mTileManager.PrepForSplit( p );
                    }
                }
            }
        }

        public override void OnMouseMove( MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseMove( mouseEventArgs, width, height );
            if ( !mCurrentlyPanning && !mCurrentlyHandlingMouseOver && mTileManager.TiledDatasetLoaded && mTileManager.SegmentationLoaded && !mTileManager.SegmentationChangeInProgress )
            {
                mCurrentlyHandlingMouseOver = true;

                //Mouseover - update display to highlight segment or area under mouse
                //Get the id of the segment under the mouse

                var p = new Vector2( (float)mouseEventArgs.X / width, (float)mouseEventArgs.Y / height );

                mTileManager.MouseOver( p );

                if ( mTileManager.MouseOverSegmentId > 0 && mTileManager.MouseOverSegmentId == mTileManager.SelectedSegmentId )
                {
                    if ( mCurrentlyDrawing )
                    {
                        if ( mTileManager.CurrentSplitMode == SplitMode.DrawSplit )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.DrawSplit( p );
                                mEngine.QuickRender();
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.DrawErase( p );
                                mEngine.QuickRender();
                            }
                        }
                        else if ( mTileManager.CurrentSplitMode == SplitMode.DrawRegions )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.DrawRegionA( p );
                                mEngine.QuickRender();
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.DrawRegionB( p );
                                mEngine.QuickRender();
                            }
                        }
                    }
                }

                mCurrentlyHandlingMouseOver = false;
            }
        }

    }
}
