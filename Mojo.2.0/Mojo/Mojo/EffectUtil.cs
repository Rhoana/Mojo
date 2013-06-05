using System;
using System.IO;
using SlimDX.D3DCompiler;
using SlimDX.Direct3D11;

namespace Mojo
{
    class EffectUtil
    {
        public static Effect CompileEffect( Device device, string effectFile )
        {
            Effect effect = null;

            var compilationSucceeded = false;
            while ( !compilationSucceeded )
            {
                try
                {
                    if ( !File.Exists( effectFile ) )
                    {
                        throw new Exception( "Effect file " + effectFile + " not found." );
                    }

                    using ( var shaderBytecode = ShaderBytecode.CompileFromFile( effectFile,
                                                                                 "fx_5_0",
                                                                                 ShaderFlags.None,
                                                                                 EffectFlags.None ) )
                    {
                        effect = new Effect( device, shaderBytecode );
                        compilationSucceeded = true;
                    }
                }
                catch ( Exception e )
                {
                    compilationSucceeded = false;
                    throw new Exception( "Could not compile effect file " + effectFile + ":\n" + e.Message );
                }
            }

            return effect;
        }
    }
}
