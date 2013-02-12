using System;
using System.Windows.Input;
using SlimDX;

namespace Mojo
{
    public class ToolBase : ITool
    {
        private readonly TileManager mTileManager;
        private readonly Engine mEngine;

        protected bool mCurrentlyPanning = false;
        protected bool mCurrentlyHandlingMouseOver = false;
        private int mPreviousMousePositionX;
        private int mPreviousMousePositionY;

        public ToolBase( TileManager tileImageManager, Engine engine )
        {
            mTileManager = tileImageManager;
            mEngine = engine;
        }

        public virtual void Select()
        {
        }

        public virtual void SelectSegment( uint segmentId )
        {
            
        }

        public virtual void OnKeyDown( KeyEventArgs keyEventArgs, int width, int height )
        {
            var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
            var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;
            var dataSpaceUnitWidthNumPixels = width / extentDataSpace.X;

            switch ( keyEventArgs.Key )
            {
                case Key.D1:
                    mEngine.CurrentToolMode = ToolMode.AdjustSegmentation;
                    break;
                case Key.D2:
                    mEngine.CurrentToolMode = ToolMode.MergeSegmentation;
                    break;
                case Key.D3:
                    mEngine.CurrentToolMode = ToolMode.SplitSegmentation;
                    break;

                case Key.Left:
                    centerDataSpace.X += Constants.ARROW_KEY_STEP / dataSpaceUnitWidthNumPixels;
                    mTileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;
                    keyEventArgs.Handled = true;
                    break;
                case Key.Right:
                    centerDataSpace.X -= Constants.ARROW_KEY_STEP / dataSpaceUnitWidthNumPixels;
                    mTileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;
                    keyEventArgs.Handled = true;
                    break;
                case Key.Up:
                    centerDataSpace.Y += Constants.ARROW_KEY_STEP / dataSpaceUnitWidthNumPixels;
                    mTileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;
                    keyEventArgs.Handled = true;
                    break;
                case Key.Down:
                    centerDataSpace.Y -= Constants.ARROW_KEY_STEP / dataSpaceUnitWidthNumPixels;
                    mTileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;
                    keyEventArgs.Handled = true;
                    break;

                case Key.W:
                    mEngine.NextImage();
                    break;
                case Key.S:
                    mEngine.PreviousImage();
                    break;

                case Key.X:
                    mEngine.ZoomOut();
                    break;
                case Key.C:
                    mEngine.ZoomIn();
                    break;

            }
        }

        public virtual void OnMouseDown( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mouseEventArgs.Button == System.Windows.Forms.MouseButtons.Middle )
            {
                mCurrentlyPanning = true;
                mPreviousMousePositionX = mouseEventArgs.X;
                mPreviousMousePositionY = mouseEventArgs.Y;
            }
        }

        public virtual void OnMouseUp( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
            switch ( mouseEventArgs.Button )
            {
                case System.Windows.Forms.MouseButtons.Middle:
                    mCurrentlyPanning = false;
                    break;
                case System.Windows.Forms.MouseButtons.XButton1:
                    mEngine.PreviousImage();
                    break;
                case System.Windows.Forms.MouseButtons.XButton2:
                    mEngine.NextImage();
                    break;
            }
        }

        public virtual void OnMouseClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
        }

        public virtual void OnMouseDoubleClick( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mTileManager.SegmentationLoaded && mouseEventArgs.Button == System.Windows.Forms.MouseButtons.Left )
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

                mEngine.CenterAndZoomToSegment2D( clickedId );
            }
        }

        public virtual void OnMouseMove( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            if ( mCurrentlyPanning )
            {
                var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
                var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;
                var dataSpaceUnitWidthNumPixels = width / extentDataSpace.X;

                centerDataSpace.X -= ( ( mouseEventArgs.X - mPreviousMousePositionX ) / dataSpaceUnitWidthNumPixels );
                centerDataSpace.Y -= ( ( mouseEventArgs.Y - mPreviousMousePositionY ) / dataSpaceUnitWidthNumPixels );

                mTileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;

                mPreviousMousePositionX = mouseEventArgs.X;
                mPreviousMousePositionY = mouseEventArgs.Y;
            }
        }

        public virtual void OnMouseWheel( System.Windows.Forms.MouseEventArgs mouseEventArgs, int width, int height )
        {
            var centerDataSpace = mTileManager.TiledDatasetView.CenterDataSpace;
            var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;
            var tiledVolumeDescription = mTileManager.TiledDatasetDescription.TiledVolumeDescriptions.Get( "SourceMap" );

            var dataSpaceUnitWidthNumPixels = width / extentDataSpace.X;

            var changeBy = (float) Math.Pow( Constants.MAGNIFICATION_STEP, Math.Abs( (double) mouseEventArgs.Delta ) / Constants.NUM_DETENTS_PER_WHEEL_MOVE );

            //
            // Prepare for mouse location calculations - these coordinates are relative the the center of the view window
            //
            var relativeMouseLocationX = mouseEventArgs.X - ( width / 2f );
            var relativeMouseLocationY = mouseEventArgs.Y - ( height / 2f );

            if ( mouseEventArgs.Delta > 0 && extentDataSpace.X > 1e-3 && extentDataSpace.Y > 1e-3 )
            {
                //
                // Decrease the view extent
                //
                extentDataSpace.X /= changeBy;
                extentDataSpace.Y /= changeBy;

                //
                // Change the center location so that the point under the mouse remains stationary
                //
                centerDataSpace.X += ( relativeMouseLocationX * changeBy - relativeMouseLocationX ) / dataSpaceUnitWidthNumPixels / changeBy;
                centerDataSpace.Y += ( relativeMouseLocationY * changeBy - relativeMouseLocationY ) / dataSpaceUnitWidthNumPixels / changeBy;
            }
            else if ( mouseEventArgs.Delta < 0 && extentDataSpace.X < tiledVolumeDescription.NumTilesX * 10 && extentDataSpace.Y < tiledVolumeDescription.NumTilesY * 10 )
            {
                //
                // Increase the view extent
                //
                extentDataSpace.X *= changeBy;
                extentDataSpace.Y *= changeBy;

                //
                // Change the center location so that the point under the mouse remains stationary
                //
                centerDataSpace.X -= ( relativeMouseLocationX * changeBy - relativeMouseLocationX ) / dataSpaceUnitWidthNumPixels;
                centerDataSpace.Y -= ( relativeMouseLocationY * changeBy - relativeMouseLocationY ) / dataSpaceUnitWidthNumPixels;
            }

            if ( centerDataSpace.X < 0 )
            {
                centerDataSpace.X = 0;

            }
            if ( centerDataSpace.X > tiledVolumeDescription.NumTilesX )
            {
                centerDataSpace.X = tiledVolumeDescription.NumTilesX;
            }

            if ( centerDataSpace.Y < 0 )
            {
                centerDataSpace.Y = 0;
            }
            if ( centerDataSpace.Y > tiledVolumeDescription.NumTilesY )
            {
                centerDataSpace.Y = tiledVolumeDescription.NumTilesY;
            }

            mTileManager.TiledDatasetView.CenterDataSpace = centerDataSpace;
            mTileManager.TiledDatasetView.ExtentDataSpace = extentDataSpace;

            mEngine.QuickRender();
        }

        public virtual void SetSize( int oldWidth, int oldHeight, int newWidth, int newHeight )
        {
            var extentDataSpace = mTileManager.TiledDatasetView.ExtentDataSpace;

            extentDataSpace.X *= (float)newWidth / (float)oldWidth;
            extentDataSpace.Y = extentDataSpace.X * (float)newHeight / (float)newWidth;

            mTileManager.TiledDatasetView.ExtentDataSpace = extentDataSpace;
            mTileManager.TiledDatasetView.WidthNumPixels = newWidth;
        }
    }
}
