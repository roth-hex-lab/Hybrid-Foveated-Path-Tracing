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

// Foveation uniforms
uniform int   u_use_foveation;
uniform float u_r_max_source;   // Maximum radius of the screen in UV space
uniform float u_r_fovea_target; // Radius of the magnified fovea in UV space
uniform float u_fovea_mult;
uniform vec2 u_fovea_center;

vec3 foveated_view_dir(const ivec2 xy, const ivec2 wh, const vec2 pixel_sample) {
    // 1. Calculate the initial pixel coordinate in the shader's native UV space.
    vec2 uv_pixel = (vec2(xy) + pixel_sample - vec2(wh) * 0.5) / float(wh.y);

    // Calculate vector relative to the arbitrary foveation center
    vec2 rel_vec = uv_pixel - u_fovea_center;
    
    // Radius in the target (rendered) image space
    float r_target = length(rel_vec); 
    
    // Resulting relative vector in source (linear) space
    vec2 rel_source = rel_vec;

    if (u_use_foveation == 1) {
        float scale_at_fovea_edge = 1.0 / u_fovea_mult;
        float scale_at_screen_edge = 1.0;
        float final_scale;

        if (r_target <= u_r_fovea_target) {
            // --- Region 1: Foveal (Magnified) ---
            final_scale = scale_at_fovea_edge;
        } else {
            // --- Region 2: Peripheral (Interpolated) ---
            
            // Direction from fovea to current pixel
            vec2 dir = (r_target > 1e-8) ? (rel_vec / r_target) : vec2(1.0, 0.0);

            // Calculate distance to screen edge from the fovea center in this direction.
            // Screen box is defined by +/- screen_dims
            vec2 screen_dims = vec2(float(resolution.x) / float(resolution.y), 1.0) * 0.5;
            
            // Ray-Box Intersection:
            // We check the distance to the vertical walls (x) and horizontal walls (y).
            // (sign(dir) * dims) gives the coordinate of the wall we are pointing at.
            vec2 dist_to_wall = (sign(dir) * screen_dims - u_fovea_center) / (dir + vec2(1e-9));
            
            // The edge is the closer of the two walls
            float r_edge_in_direction = min(dist_to_wall.x, dist_to_wall.y);

            // Interpolation factor 't'
            float t = (r_target - u_r_fovea_target) / max(1e-6, r_edge_in_direction - u_r_fovea_target);
            t = clamp(t, 0.0, 1.0);

            // Linearly interpolate the SCALE FACTOR
            final_scale = mix(scale_at_fovea_edge, scale_at_screen_edge, t);
        }
        
        // Apply scale to the relative vector
        rel_source = rel_vec * final_scale;
    }

    // The final UV source is the Fovea Center + Scaled Relative Vector
    vec2 uv_source = u_fovea_center + rel_source;

    // Generate the final view direction
    const float z = -0.5 / tan(0.5 * M_PI * cam_fov / 180.0);
    return normalize(cam_transform * normalize(vec3(uv_source.x, uv_source.y, z)));
}


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
        dir = foveated_view_dir(pixel, resolution, rng2(seed));
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
