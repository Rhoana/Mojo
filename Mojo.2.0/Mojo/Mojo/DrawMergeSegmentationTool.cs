using System.Windows.Forms;
using SlimDX;

namespace Mojo
{
    class DrawMergeSegmentationTool : ToolBase
    {
        private readonly TileManager mTileManager;
        private readonly Engine mEngine;

        private bool mCurrentlyDrawing = false;
        private bool mCurrentlyMovingZ = false;

        public DrawMergeSegmentationTool( TileManager tileManager, Engine engine )
            : base( tileManager, engine )
        {
            mTileManager = tileManager;
            mEngine = engine;
        }

        public override void Select()
        {
            var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
            var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
            mTileManager.Internal.PrepForDrawMerge( p );
            mTileManager.Internal.ResetDrawMergeState( p );
        }

        public override void SelectSegment( uint segmentId )
        {
            if ( mTileManager.SelectedSegmentId != segmentId )
            {
                mTileManager.SelectedSegmentId = segmentId;
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
                mTileManager.Internal.PrepForDrawMerge( p );
            }
        }

        public override void MoveZ()
        {
            if ( mTileManager.SelectedSegmentId != 0 && !mCurrentlyMovingZ )
            {
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );
                mTileManager.Internal.PrepForDrawMerge( p );
            }
        }

        public override void OnKeyDown( System.Windows.Input.KeyEventArgs keyEventArgs, int width, int height )
        {

            var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
            var p = new Vector3( centerDataSpace.X, centerDataSpace.Y, centerDataSpace.Z );

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
                        mTileManager.Internal.PrepForDrawMerge( p );
                    }
                    break;
                case System.Windows.Input.Key.Y:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.RedoChange();
                        mTileManager.Internal.PrepForDrawMerge( p );
                    }
                    break;
                case System.Windows.Input.Key.N:
                    if ( keyEventArgs.KeyboardDevice.Modifiers == System.Windows.Input.ModifierKeys.Control )
                    {
                        mTileManager.SelectNewId();
                        mTileManager.Internal.PrepForDrawMerge( p );
                    }
                    break;
                case System.Windows.Input.Key.Left:
                case System.Windows.Input.Key.Right:
                case System.Windows.Input.Key.Up:
                case System.Windows.Input.Key.Down:
                case System.Windows.Input.Key.X:
                case System.Windows.Input.Key.C:
                    mTileManager.Internal.PrepForDrawMerge( p );
                    break;

                case System.Windows.Input.Key.OemComma:
                case System.Windows.Input.Key.OemMinus:
                case System.Windows.Input.Key.Subtract:
                    mTileManager.DecreaseMergeBrushSize();
                    break;
                case System.Windows.Input.Key.OemPeriod:
                case System.Windows.Input.Key.OemPlus:
                case System.Windows.Input.Key.Add:
                    mTileManager.IncreaseMergeBrushSize();
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

            if ( mTileManager.SegmentationLoaded )
            {
                //
                // Draw here
                //

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

                if ( mouseEventArgs.Button == MouseButtons.Left || mouseEventArgs.Button == MouseButtons.Right )
                {
                    mTileManager.Internal.PrepForDrawMerge( p );
                    mCurrentlyDrawing = true;
                    mTileManager.Internal.DrawRegionA( mTileManager.TiledDatasetView, p, mTileManager.MergeBrushSize );
                    mEngine.QuickRender();
                }
            }
        }

        public override void OnMouseUp( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseUp( mouseEventArgs, width, height );

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

            if ( mouseEventArgs.Button == MouseButtons.Middle )
            {
                mTileManager.Internal.PrepForDrawMerge( p );
            }
            else if ( mCurrentlyDrawing && ( mouseEventArgs.Button == MouseButtons.Left || mouseEventArgs.Button == MouseButtons.Right ) )
            {
                switch ( mTileManager.CurrentMergeMode )
                {
                    case MergeMode.Fill2D:
                        mTileManager.SelectedSegmentId = mTileManager.CommitDrawMergeCurrentSlice();
                        break;
                    case MergeMode.Fill3D:
                        mTileManager.SelectedSegmentId = mTileManager.CommitDrawMergeCurrentConnectedComponent();
                        break;
                    default:
                        mTileManager.SelectedSegmentId = mTileManager.CommitDrawMerge();
                        break;
                }
                mTileManager.ChangesMade = true;
            }
            mCurrentlyDrawing = false;
        }

        public override void OnMouseMove( MouseEventArgs mouseEventArgs, int width, int height )
        {
            base.OnMouseMove( mouseEventArgs, width, height );
            if ( !mCurrentlyPanning && !mCurrentlyHandlingMouseOver && mTileManager.TiledDatasetLoaded && mTileManager.SegmentationLoaded )
            {
                mCurrentlyHandlingMouseOver = true;

                //
                // Mouseover - update display to highlight segment or area under mouse
                // Get the id of the segment under the mouse
                //

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

                //
                // Make a hover circle
                //
                mTileManager.MouseOverX = x;
                mTileManager.MouseOverY = y;

                if ( mCurrentlyDrawing )
                {
                    mTileManager.Internal.DrawRegionA( mTileManager.TiledDatasetView, p, mTileManager.MergeBrushSize );
                    mEngine.QuickRender();
                }

                mCurrentlyHandlingMouseOver = false;

            }
        }

        public override void OnMouseWheel( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            //
            // Only scroll if the mouse within the extent
            //
            if ( mouseEventArgs.X < 0 || mouseEventArgs.X > width ||
                 mouseEventArgs.Y < 0 || mouseEventArgs.Y > height )
            {
                return;
            }

            base.OnMouseWheel( mouseEventArgs, width, height );

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
            mTileManager.Internal.PrepForDrawMerge( p );
        }


    }
}
