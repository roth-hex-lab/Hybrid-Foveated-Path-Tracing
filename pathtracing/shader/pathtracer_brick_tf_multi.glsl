#version 450 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) writeonly uniform image2D color_out;
layout (binding = 1, rgba32f) writeonly uniform image2D motion_out;
layout (binding = 2, rgba32f) writeonly uniform image2D albedo_out; 
layout (binding = 3, rgba32f) writeonly uniform image2D depth_out; 

// ---------------------------------------------------
// settings

#define USE_DDA
#define USE_TRANSFERFUNC

// Include
#include "common.glsl"

// ---------------------------------------------------
// uniforms

uniform int useed;
uniform ivec2 resolution;
uniform uint trace_samples;

uniform mat4 current_view_projection_matrix;
uniform mat4 prev_view_projection_matrix;

uniform int generate_motionvec;
uniform int generate_albedo;
uniform int generate_depth;

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, resolution))) return;

    const vec3 pos = cam_pos;
    bool first_sample = true;

    vec4 L_accum = vec4(0.0);
    vec4 MOTION  = vec4(0.0);
    vec4 ALBEDO  = vec4(0.0);
    vec4 DEPTH   = vec4(0.0);

    vec3 dir;
    
    uint pixelIndex = uint(pixel.y) * uint(resolution.x) + uint(pixel.x);
    uint baseSeed = tea(useed ^ pixelIndex, 0u, 8);   
    uint seed = baseSeed; 

    for (uint i = 0; i < trace_samples; ++i) {
        seed = tea(baseSeed, i, 8);
        dir = view_dir(pixel, resolution, rng2(seed));
        vec4 L_pass;

        bool should_mv = bool(generate_motionvec) && first_sample;
        bool should_d = bool(generate_depth) && first_sample;

        trace_path_multi(pos, dir, seed, L_pass, 
                   should_mv, current_view_projection_matrix, prev_view_projection_matrix, resolution, MOTION,
                   should_d, DEPTH);
    
        first_sample = false;

        L_accum += sanitize(L_pass);
        if (should_mv) {
            imageStore(motion_out, pixel, sanitize(MOTION));
        }
        if (should_d) {
            imageStore(depth_out, pixel, sanitize(DEPTH));
        }
    }
    
    if (bool(generate_albedo)) {
        ALBEDO = calculate_integrated_albedo(pos, dir, seed);
        imageStore(albedo_out, pixel, sanitize(ALBEDO));
    }

    // write result
    vec4 final_color = vec4(L_accum / float(trace_samples));
    imageStore(color_out, pixel, final_color);
}
