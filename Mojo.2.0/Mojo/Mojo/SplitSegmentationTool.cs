using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    public class SplitSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private readonly Engine mEngine;

        private bool mCurrentlyDrawing = false;

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
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
                mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, p );
            }
        }

        public override void SelectSegment( uint segmentId )
        {
            if ( mTileManager.SelectedSegmentId != segmentId )
            {
                mTileManager.SelectedSegmentId = segmentId;
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
                mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, p );
            }
        }

        public override void MoveZ()
        {
            //
            // Z index has changed - prep for splitting
            //
            if ( mTileManager.SelectedSegmentId != 0 )
            {
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
                mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, p );
                if ( mTileManager.JoinSplits3D )
                {
                    mTileManager.Internal.PredictSplit( mTileManager.SelectedSegmentId, p, mTileManager.BrushSize );
                }
            }
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {
            base.OnKeyDown( keyEventArgs, width, height );

            var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
            var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );

            switch ( keyEventArgs.Key )
            {
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
                        mTileManager.UndoChange();
                        mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, p );
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                        mTileManager.RedoChange();
                        mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, p );
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
        }

        public override void OnMouseDown( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseDown( mouseEventArgs, width, height );
            if ( mTileManager.CurrentSplitMode != SplitMode.JoinPoints )
            {
                if ( mTileManager.SegmentationLoaded )
                {

                    //Get the id of the segment being clicked

                    var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                    var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                    var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                    var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                    var offsetDataSpaceX = ( (float)mouseEventArgs.X / width ) * extentDataSpace.X;
                    var offsetDataSpaceY = ( (float)mouseEventArgs.Y / height ) * extentDataSpace.Y;

                    var x = topLeftDataSpaceX + offsetDataSpaceX;
                    var y = topLeftDataSpaceY + offsetDataSpaceY;
                    var z = centerDataSpace.Z;

                    var p = new Vector3( x, y, z );

                    var clickedId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                    if ( clickedId == mTileManager.SelectedSegmentId && mTileManager.CurrentSplitMode != SplitMode.JoinPoints )
                    {
                        mCurrentlyDrawing = true;

                        if ( mTileManager.CurrentSplitMode == SplitMode.DrawSplit )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.Internal.DrawSplit( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                                mEngine.QuickRender();
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.Internal.DrawErase( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                                mEngine.QuickRender();
                            }
                        }
                        else if ( mTileManager.CurrentSplitMode == SplitMode.DrawRegions )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.Internal.DrawRegionA( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                                mEngine.QuickRender();
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.Internal.DrawRegionB( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
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

                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                var offsetDataSpaceX = ( (float)mouseEventArgs.X / width ) * extentDataSpace.X;
                var offsetDataSpaceY = ( (float)mouseEventArgs.Y / height ) * extentDataSpace.Y;
                var x = topLeftDataSpaceX + offsetDataSpaceX;
                var y = topLeftDataSpaceY + offsetDataSpaceY;
                var z = centerDataSpace.Z;

                var p = new Vector3( x, y, z );

                if ( mTileManager.CurrentSplitMode == SplitMode.DrawSplit )
                {
                    mTileManager.Internal.FindBoundaryWithinRegion2D( mTileManager.SelectedSegmentId, p );
                }
                if ( mTileManager.CurrentSplitMode == SplitMode.DrawRegions )
                {
                    mTileManager.Internal.FindBoundaryBetweenRegions2D( mTileManager.SelectedSegmentId, p );
                }
            }
        }

        public override void OnMouseClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mTileManager.SegmentationLoaded && !mCurrentlyDrawing )
            {
                //Get the id of the segment being clicked

                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                var offsetDataSpaceX = ( (float) mouseEventArgs.X / width ) * extentDataSpace.X;
                var offsetDataSpaceY = ( (float) mouseEventArgs.Y / height ) * extentDataSpace.Y;

                var x = topLeftDataSpaceX + offsetDataSpaceX;
                var y = topLeftDataSpaceY + offsetDataSpaceY;
                var z = centerDataSpace.Z;

                var p = new Vector3( x, y, z );

                var clickedId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                if ( mouseEventArgs.Button == MouseButtons.Left )
                {
                    //Select this segment
                    if ( clickedId > 0 )
                    {
                        if ( clickedId != mTileManager.SelectedSegmentId )
                        {
                            //Select this segment
                            mTileManager.SelectedSegmentId = clickedId;
                            mTileManager.Internal.PrepForSplit( clickedId, p );
                        }
                    }
                }
                else if ( mouseEventArgs.Button == MouseButtons.Right )
                {
                    if ( clickedId > 0 && clickedId == mTileManager.SelectedSegmentId && mTileManager.CurrentSplitMode == SplitMode.JoinPoints )
                    {
                        mTileManager.Internal.AddSplitSource( mTileManager.TiledDatasetView, p );
                        mTileManager.Internal.DrawRegionB( mTileManager.TiledDatasetView, p, 4 );
                        mTileManager.Internal.FindBoundaryJoinPoints2D( clickedId, p );
                    }
                    else if ( clickedId > 0 && mTileManager.SelectedSegmentId > 0 && clickedId != mTileManager.SelectedSegmentId )
                    {
                        //
                        // Merge with the clicked segment in 2D ( try this out to see if it makes sense )
                        //
                        switch ( mTileManager.CurrentMergeMode )
                        {
                            case MergeMode.Fill2D:
                                mTileManager.Internal.ReplaceSegmentationLabelCurrentSlice( clickedId, mTileManager.SelectedSegmentId, mTileManager.TiledDatasetView, p );
                                break;
                            case MergeMode.Fill3D:
                                mTileManager.Internal.ReplaceSegmentationLabelCurrentConnectedComponent( clickedId, mTileManager.SelectedSegmentId, mTileManager.TiledDatasetView, p );
                                break;
                            default:
                                mTileManager.Internal.ReplaceSegmentationLabel( clickedId, mTileManager.SelectedSegmentId );
                                break;
                        }
                        mTileManager.ChangesMade = true;
                        mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, p );
                    }
                }
            }
        }

        public override void OnMouseMove( MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseMove( mouseEventArgs, width, height );
            if ( !mCurrentlyPanning && !mCurrentlyHandlingMouseOver && mTileManager.TiledDatasetLoaded && mTileManager.SegmentationLoaded )
            {
                mCurrentlyHandlingMouseOver = true;

                //Mouseover - update display to highlight segment or area under mouse
                //Get the id of the segment under the mouse

                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

                var topLeftDataSpaceX = centerDataSpace.X - ( extentDataSpace.X / 2f );
                var topLeftDataSpaceY = centerDataSpace.Y - ( extentDataSpace.Y / 2f );

                var offsetDataSpaceX = ( (float)mouseEventArgs.X / width ) * extentDataSpace.X;
                var offsetDataSpaceY = ( (float)mouseEventArgs.Y / height ) * extentDataSpace.Y;

                var x = topLeftDataSpaceX + offsetDataSpaceX;
                var y = topLeftDataSpaceY + offsetDataSpaceY;
                var z = centerDataSpace.Z;

                var p = new Vector3( x, y, z );

                mTileManager.MouseOverSegmentId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                if ( mTileManager.MouseOverSegmentId > 0 && mTileManager.MouseOverSegmentId == mTileManager.SelectedSegmentId )
                {
                    //Make a hover circle
                    mTileManager.MouseOverX = x;
                    mTileManager.MouseOverY = y;

                    if ( mCurrentlyDrawing )
                    {
                        if ( mTileManager.CurrentSplitMode == SplitMode.DrawSplit )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.Internal.DrawSplit( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                                mEngine.QuickRender();
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.Internal.DrawErase( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                                mEngine.QuickRender();
                            }
                        }
                        else if ( mTileManager.CurrentSplitMode == SplitMode.DrawRegions )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.Internal.DrawRegionA( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                                mEngine.QuickRender();
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.Internal.DrawRegionB( mTileManager.TiledDatasetView, p, mTileManager.BrushSize );
                                mEngine.QuickRender();
                            }
                        }
                    }
                }
                else
                {
                    mTileManager.MouseOverX = 0;
                    mTileManager.MouseOverY = 0;
                }

                mCurrentlyHandlingMouseOver = false;
            }
        }

    }
}
