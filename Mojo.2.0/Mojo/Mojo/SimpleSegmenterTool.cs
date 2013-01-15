using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    public class SimpleSegmenterTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private int newId = 0;
        private bool mCurrentlyDrawing = false;

        public SimpleSegmenterTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
        }

        public override void Select()
        {
            if ( mTileManager.SelectedSegmentId != 0 )
            {
                mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, (int)mTileManager.TiledDatasetView.CenterDataSpace.Z );
            }
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {
            base.OnKeyDown( keyEventArgs, width, height );

            switch ( keyEventArgs.Key )
            {
                case System.Windows.Input.Key.A:
                    mTileManager.ToggleShowSegmentation();
                    break;
                case System.Windows.Input.Key.E:
                    mTileManager.SegmentationVisibilityRatio = System.Math.Min( mTileManager.SegmentationVisibilityRatio + 0.1f, 1.0f );
                    break;
                case System.Windows.Input.Key.D:
                    mTileManager.SegmentationVisibilityRatio = System.Math.Max( mTileManager.SegmentationVisibilityRatio - 0.1f, 0f );
                    break;
                case System.Windows.Input.Key.Z:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                        mTileManager.Internal.UndoChange();
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                        mTileManager.Internal.RedoChange();
                    break;
                case System.Windows.Input.Key.Tab:
                    mTileManager.Internal.CompleteSplit( mTileManager.SelectedSegmentId );
                    break;

                case System.Windows.Input.Key.Escape:
                    mTileManager.Internal.ResetSplitState();
                    break;

                //Base class will move the view - make sure we prep for splitting again
                case System.Windows.Input.Key.W:
                    mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, (int) mTileManager.TiledDatasetView.CenterDataSpace.Z );
                    break;
                case System.Windows.Input.Key.S:
                    mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, (int) mTileManager.TiledDatasetView.CenterDataSpace.Z );
                    break;

                case System.Windows.Input.Key.OemComma:
                    if ( mTileManager.DrawSize > 4 )
                    {
                        mTileManager.DrawSize -= 2;
                    }
                    break;
                case System.Windows.Input.Key.OemPeriod:
                    if ( mTileManager.DrawSize < 16 )
                    {
                        mTileManager.DrawSize += 2;
                    }
                    break;

            }
        }

        public override void OnMouseDown( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mouseEventArgs.Button == System.Windows.Forms.MouseButtons.Right )
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
                    }
                }
            }
            else
            {
                base.OnMouseDown( mouseEventArgs, width, height );
            }
        }

        public override void OnMouseUp( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mCurrentlyDrawing )
            {
                mCurrentlyDrawing = false;

                if ( mTileManager.CurrentSplitMode == SplitMode.DrawSplit )
                {
                    mTileManager.Internal.FindBoundaryWithinRegion2D( mTileManager.SelectedSegmentId );
                }
                if ( mTileManager.CurrentSplitMode == SplitMode.DrawRegions )
                {
                    mTileManager.Internal.FindCutBetweenRegions2D( mTileManager.SelectedSegmentId );
                }
            }
            else
            {
                base.OnMouseUp( mouseEventArgs, width, height );
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
                        if ( newId == clickedId )
                        {
                            //Unselect this segment
                            //newId = 0;
                            //mTileManager.SelectedSegmentId = 0;

                            //Load 2D segment at full res for path finding
                            //mTileManager.Internal.ResetSplitState();
                        }
                        else
                        {
                            //Select this segment
                            newId = clickedId;
                            mTileManager.SelectedSegmentId = clickedId;

                            //Load 2D segment at full res for path finding
                            mTileManager.Internal.PrepForSplit( clickedId, (int) z );

                        }
                    }
                }
                else if ( mouseEventArgs.Button == MouseButtons.Right )
                {
                    if ( clickedId > 0 && clickedId == mTileManager.SelectedSegmentId && mTileManager.CurrentSplitMode == SplitMode.JoinPoints )
                    {
                        mTileManager.Internal.AddSplitSource( mTileManager.TiledDatasetView, p );
                        mTileManager.Internal.FindBoundaryJoinPoints2D( clickedId );
                    }
                    else if ( clickedId > 0 )
                    {
                        //
                        // Merge with the clicked segment in 2D ( try this out to see if it makes sense )
                        //
                        mTileManager.Internal.ReplaceSegmentationLabelCurrentSlice( clickedId, mTileManager.SelectedSegmentId, mTileManager.TiledDatasetView, p );
                        mTileManager.Internal.PrepForSplit( mTileManager.SelectedSegmentId, (int)z );
                    }
                }
            }
        }

        public override void OnMouseMove(MouseEventArgs mouseEventArgs, int width, int height)
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

                int segmentId = mTileManager.Internal.GetSegmentationLabelId( mTileManager.TiledDatasetView, p );

                if ( segmentId > 0 )
                {
                    mTileManager.MouseOverSegmentId = segmentId;
                }
                else
                {
                    mTileManager.MouseOverSegmentId = 0;
                }

                if ( segmentId > 0 && segmentId == mTileManager.SelectedSegmentId )
                {
                    //Make a hover circle
                    mTileManager.MouseOverX = x;
                    mTileManager.MouseOverY = y;

                    if ( mCurrentlyDrawing )
                    {
                        if ( mTileManager.CurrentSplitMode == SplitMode.DrawSplit )
                        {
                            mTileManager.Internal.DrawSplit( mTileManager.TiledDatasetView, p, mTileManager.DrawSize );
                        }
                        else if ( mTileManager.CurrentSplitMode == SplitMode.DrawRegions )
                        {
                            if ( mouseEventArgs.Button == MouseButtons.Left )
                            {
                                mTileManager.Internal.DrawRegionA( mTileManager.TiledDatasetView, p, mTileManager.DrawSize );
                            }
                            else if ( mouseEventArgs.Button == MouseButtons.Right )
                            {
                                mTileManager.Internal.DrawRegionB( mTileManager.TiledDatasetView, p, mTileManager.DrawSize );
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
