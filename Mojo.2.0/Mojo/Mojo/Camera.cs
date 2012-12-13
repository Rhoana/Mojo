using SlimDX;

namespace Mojo
{
    public class Camera
    {
        private Vector3 mPosition;
        private Vector3 mUp;
        private Vector3 mTarget;
        private Matrix mProjectionMatrix;

        public Camera( Vector3 position, Vector3 target, Vector3 upHint, Matrix projectionMatrix )
        {
            mProjectionMatrix = projectionMatrix;
            SetLookAtVectors( position, target, upHint );
        }

        public void GetLookAtVectors( out Vector3 position, out Vector3 target, out Vector3 up )
        {
            position = mPosition;
            target = mTarget;
            up = mUp;
        }

        public void SetLookAtVectors( Vector3 position, Vector3 target, Vector3 upHint )
        {
            mPosition = position;
            mTarget   = target;

            // compute the stored up vector to be orthogonal to the other vectors
            mUp = Vector3.Cross( Vector3.Cross( mTarget - mPosition, upHint ), mTarget - mPosition );
            mUp.Normalize();
        }

        public Matrix GetProjectionMatrix()
        {
            return mProjectionMatrix;
        }

        public void SetProjectionMatrix( Matrix projectionMatrix )
        {
            mProjectionMatrix = projectionMatrix;
        }

        public Matrix GetLookAtMatrix()
        {
            return Matrix.LookAtLH( mPosition, mTarget, mUp );
        }
    }
}
