using SlimDX;

namespace Mojo
{
    static class MathUtil
    {
        public static bool ApproxEqual( float a, float b, float e )
        {
            return ( System.Math.Abs( a - b ) <= e );
        }

        public static Vector3 TransformAndHomogeneousDivide( Vector3 vector3, Matrix matrix )
        {
            var vector4 = Vector3.Transform( vector3, matrix );
            return new Vector3( vector4.X / vector4.W, vector4.Y / vector4.W, vector4.Z / vector4.W );
        }

        public static Vector3 ConvertToFloatColor( Vector3 color )
        {
            return color / 255f;
        }
    }
}
