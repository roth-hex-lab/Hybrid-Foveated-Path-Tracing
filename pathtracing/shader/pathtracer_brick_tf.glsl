#version 450 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 0, rgba32f) uniform image2D color;
layout (binding = 1, rg32f) uniform image2D motion_vectors;

// ---------------------------------------------------
// settings

#define USE_DDA
#define USE_TRANSFERFUNC

// Include
#include "common.glsl"

// ---------------------------------------------------
// uniforms

uniform int current_sample;
uniform int seed;
uniform ivec2 resolution;

uniform mat4 current_view_projection_matrix;
uniform mat4 prev_view_projection_matrix;

// ---------------------------------------------------
// main

void main() {
	const ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, resolution))) return;

    // setup random seed and camera ray
    uint seed = tea(seed * (pixel.y * resolution.x + pixel.x), current_sample, 32);
    const vec3 pos = cam_pos;
    const vec3 dir = view_dir(pixel, resolution, rng2(seed));
    bool first_sample = (current_sample == 1);
    bool calculate_motion = false; // TODO: Configurable

    // trace ray
    vec4 L;
    vec2 mv = vec2(0.0);
    mv = trace_path(pos, dir, seed, L, current_view_projection_matrix, prev_view_projection_matrix, resolution, first_sample, calculate_motion);
    //L = direct_volume_rendering(pos, dir, seed);

    // write result
    imageStore(color, pixel, mix(imageLoad(color, pixel), sanitize(L), 1.f / current_sample));
    if (first_sample && calculate_motion) {
        imageStore(motion_vectors, pixel, vec4(mv, 0.0, 0.0));
    }
}
