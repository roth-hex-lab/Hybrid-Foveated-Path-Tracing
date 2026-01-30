Shader "Custom/DisplacementReprojection" {
    Properties
    {
        [Toggle(REPROJECTION_ON)] _EnableReprojection ("Enable Reprojection", Float) = 0
        [Toggle(VR_MODE)] _VRMode ("VR Mode", Float) = 0
        
        // Left eye (or mono) textures
        _MainTex ("Texture (Left Eye)", 2D) = "white" {}
        _DepthMap ("Depth Map (Left Eye - World Units)", 2D) = "white" {}
        
        // Right eye textures
        _MainTexRight ("Texture (Right Eye)", 2D) = "white" {}
        _DepthMapRight ("Depth Map (Right Eye - World Units)", 2D) = "white" {}
        
        _Color ("Tint Color", Color) = (1,1,1,1)
        _DepthOffset ("Additional Offset For Depth", Float) = 0
        _DepthScale ("Scale Adjustment for Depth", Float) = 0
        _EdgeSoftness ("Fade Distance on sides", Range(0, 1)) = 0.05
    }
    
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        Blend One OneMinusSrcAlpha
        ZWrite Off
        Cull Off
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma shader_feature_local REPROJECTION_ON
            #pragma shader_feature_local VR_MODE
            #pragma multi_compile_instancing
            
            #include "UnityCG.cginc"
            
            struct appdata {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            
            struct v2f {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };
            
            sampler2D _MainTex;
            sampler2D _DepthMap;
            sampler2D _MainTexRight;
            sampler2D _DepthMapRight;
            float4 _Color;
            float _EdgeSoftness;

            #if REPROJECTION_ON
                // Arrays for per-eye matrices in VR
                float4x4 _Prev_ViewProjectionInverse;
                float4x4 _Prev_ViewProjectionInverseRight;
                float4 _PastCamPos;
                float4 _PastCamPosRight;
                float _DepthOffset;
                float _DepthScale;
            #endif

            v2f vert (appdata v) {
                v2f o;
                
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                #if REPROJECTION_ON
                    #if VR_MODE
                        bool isRightEye = unity_StereoEyeIndex == 1;
                        
                        float linear_depth;
                        float4x4 viewProjInv;
                        float4 camPos;
                        
                        if (isRightEye) {
                            linear_depth = tex2Dlod(_DepthMapRight, float4(v.uv, 0, 0)).r;
                            viewProjInv = _Prev_ViewProjectionInverseRight;
                            camPos = _PastCamPosRight;
                        } else {
                            linear_depth = tex2Dlod(_DepthMap, float4(v.uv, 0, 0)).r;
                            viewProjInv = _Prev_ViewProjectionInverse;
                            camPos = _PastCamPos;
                        }
                        
                        linear_depth = (linear_depth * _DepthScale) + _DepthOffset;
                        
                        float2 ndc = v.uv * 2.0 - 1.0;
                        float4 clip_space_pos = float4(ndc.x, ndc.y, 1.0, 1.0);
                        
                        float4 world_space_far_point = mul(viewProjInv, clip_space_pos);
                        world_space_far_point /= world_space_far_point.w;
                        
                        float3 ray_direction = normalize(world_space_far_point.xyz - camPos.xyz);
                        float3 true_world_pos = camPos.xyz + ray_direction * linear_depth;

                        o.vertex = mul(UNITY_MATRIX_VP, float4(true_world_pos, 1.0));
                    #else
                        // Non-VR reprojection
                        float linear_depth = tex2Dlod(_DepthMap, float4(v.uv, 0, 0)).r;
                        linear_depth = (linear_depth * _DepthScale) + _DepthOffset;
                        
                        float2 ndc = v.uv * 2.0 - 1.0;
                        float4 clip_space_pos = float4(ndc.x, ndc.y, 1.0, 1.0);
                        
                        float4 world_space_far_point = mul(_Prev_ViewProjectionInverse, clip_space_pos);
                        world_space_far_point /= world_space_far_point.w;
                        
                        float3 ray_direction = normalize(world_space_far_point.xyz - _PastCamPos.xyz);
                        float3 true_world_pos = _PastCamPos.xyz + ray_direction * linear_depth;

                        o.vertex = mul(UNITY_MATRIX_VP, float4(true_world_pos, 1.0));
                    #endif
                #else
                    o.vertex = UnityObjectToClipPos(v.vertex);
                #endif
                
                o.uv = v.uv;
                return o;
            }
            
            fixed4 frag (v2f i) : SV_Target {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                
                fixed4 color;
                
                #if VR_MODE
                    // Sample appropriate texture based on eye
                    bool isRightEye = unity_StereoEyeIndex == 1;
                    if (isRightEye) {
                        color = tex2D(_MainTexRight, i.uv) * _Color;
                    } else {
                        color = tex2D(_MainTex, i.uv) * _Color;
                    }
                #else
                    color = tex2D(_MainTex, i.uv) * _Color;
                #endif

                float2 edgeDistance = min(i.uv, 1.0 - i.uv);
                
                float2 fadeFactors = smoothstep(0.0, _EdgeSoftness, edgeDistance);
                float fadeMask = fadeFactors.x * fadeFactors.y;
                
                color.a *= fadeMask;
                color.rgb *= color.a;

                return color;
            }
            ENDCG
        }
    }
}