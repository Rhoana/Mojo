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
                    Release.Assert( File.Exists( effectFile ) );

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
                    Release.Assert( false, e.Message );
                }
            }

            return effect;
        }
    }
}
