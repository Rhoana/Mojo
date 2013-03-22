//
// globals
//
Texture2D<float4> gTexture2D;
Texture3D<float4> gTexture3D;
float4x4          gTransform;
float4            gColor;

sampler gTextureSampler = 
sampler_state
{
    Filter = MIN_LINEAR_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
    AddressW = Clamp;    
};


//
// position only
//
struct VS_POSITION_IN
{
    float4 position : POSITION;
};

struct PS_POSITION_IN
{
    float4 position : SV_POSITION;
};

PS_POSITION_IN VS_POSITION( VS_POSITION_IN input )
{
    PS_POSITION_IN output = (PS_POSITION_IN)0;
    
    output.position = mul( input.position, gTransform );
    
    return output;
}

PS_POSITION_IN VS_POSITION_DEPTH_BIAS( VS_POSITION_IN input )
{
    PS_POSITION_IN output = (PS_POSITION_IN)0;
    
    output.position = mul( input.position, gTransform );
    output.position.z -= 0.0005f;

    return output;
}

float4 PS_POSITION( PS_POSITION_IN input ) : SV_Target
{
    return gColor;
}


//
// position and texcoord
//
struct VS_POSITION_TEXCOORD_IN
{
    float4 position : POSITION;
    float4 texCoord : TEXCOORD0;
};

struct PS_POSITION_TEXCOORD_IN
{
    float4 position : SV_POSITION;
    float4 texCoord : TEXCOORD0;
};

PS_POSITION_TEXCOORD_IN VS_POSITION_TEXCOORD( VS_POSITION_TEXCOORD_IN input )
{
    PS_POSITION_TEXCOORD_IN output = (PS_POSITION_TEXCOORD_IN)0;
    
    output.position = mul( input.position, gTransform );
    output.texCoord = input.texCoord;

    return output;
}

float4 PS_POSITION_TEXCOORD_3D( PS_POSITION_TEXCOORD_IN input ) : SV_Target
{
    return float4( gTexture3D.Sample( gTextureSampler, input.texCoord.xyz ).xyz, 1.0f );
}

float4 PS_POSITION_TEXCOORD_3D_GREY_SCALE( PS_POSITION_TEXCOORD_IN input ) : SV_Target
{
    return float4( gTexture3D.Sample( gTextureSampler, input.texCoord.xyz ).x, gTexture3D.Sample( gTextureSampler, input.texCoord.xyz ).x, gTexture3D.Sample( gTextureSampler, input.texCoord.xyz ).x, 1.0f );
}

//
// state
//
BlendState gBlendStateColorWriteMaskEnable
{
    RenderTargetWriteMask[ 0 ] = 15; // D3D11_COLOR_WRITE_ENABLE_ALL
};

RasterizerState gRasterizerStateSolid
{
    FillMode = Solid;
    CullMode = None;
};

RasterizerState gRasterizerStateWireframe
{
    FillMode = Wireframe;
    CullMode = None;
};

DepthStencilState gDepthStencilState
{
    DepthEnable = true;
    DepthWriteMask = All;
    StencilEnable = false;
};


//
// techniques
//
technique11 RenderWireframe
{
    pass RenderWireframe
    {
        SetBlendState( gBlendStateColorWriteMaskEnable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
        SetDepthStencilState( gDepthStencilState, 0x00000000 ); 
        SetRasterizerState( gRasterizerStateWireframe );
        SetGeometryShader( 0 );
        SetVertexShader( CompileShader( vs_5_0, VS_POSITION_DEPTH_BIAS() ) );
        SetPixelShader( CompileShader( ps_5_0, PS_POSITION() ) );
    }
}

technique11 RenderSolid
{
    pass RenderSolid
    {
        SetBlendState( gBlendStateColorWriteMaskEnable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
        SetDepthStencilState( gDepthStencilState, 0x00000000 ); 
        SetRasterizerState( gRasterizerStateSolid );
        SetGeometryShader( 0 );
        SetVertexShader( CompileShader( vs_5_0, VS_POSITION() ) );
        SetPixelShader( CompileShader( ps_5_0, PS_POSITION() ) );
    }
}

technique11 RenderTexture3D
{
    pass RenderTexture3D
    {
        SetBlendState( gBlendStateColorWriteMaskEnable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
        SetDepthStencilState( gDepthStencilState, 0x00000000 ); 
        SetRasterizerState( gRasterizerStateSolid );
        SetGeometryShader( 0 );
        SetVertexShader( CompileShader( vs_5_0, VS_POSITION_TEXCOORD() ) );
        SetPixelShader( CompileShader( ps_5_0, PS_POSITION_TEXCOORD_3D() ) );
    }
}

technique11 RenderGreyScaleTexture3D
{
    pass RenderGreyScaleTexture3D
    {
        SetBlendState( gBlendStateColorWriteMaskEnable, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xffffffff );
        SetDepthStencilState( gDepthStencilState, 0x00000000 ); 
        SetRasterizerState( gRasterizerStateSolid );
        SetGeometryShader( 0 );
        SetVertexShader( CompileShader( vs_5_0, VS_POSITION_TEXCOORD() ) );
        SetPixelShader( CompileShader( ps_5_0, PS_POSITION_TEXCOORD_3D_GREY_SCALE() ) );
    }
}
