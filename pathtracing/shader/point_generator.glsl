#version 450 core

layout (local_size_x = 256) in;

// ---------------------------------------------------
// settings

#define USE_DDA
#define USE_TRANSFERFUNC


// --- Data Structures ---

struct PointCloudElement {
    vec4 world_pos;
    vec4 view_dir;
    uint accumulated_color_r;
    uint accumulated_color_g;
    uint accumulated_color_b;
    uint sample_count;
};

layout(std430, binding = 0) buffer PointBuffer {
    PointCloudElement points[];
};

layout(std430, binding = 1) buffer AtomicCounter {
    uint point_counter;
};

// Include
#include "common.glsl"

// ---------------------------------------------------
// uniforms

uniform int current_sample;
uniform int seed;
uniform ivec2 resolution;

uniform int u_mode; // 0 for Generation, 1 for Convergence
uniform int u_target_point_count;
uniform int u_seed_offset;


// --- Helper Functions ---

// A simplified trace function that finds the *first* real collision
// and returns its world-space position.
vec3 find_first_surface(vec3 pos, vec3 dir, inout uint seed) {
    float t;
    vec3 ignored_throughput = vec3(1);
    vec3 ignored_Le = vec3(0);

    // Call existing sample_volume function, but we only care about the first hit.
    if (sample_volume(pos, dir, t, ignored_throughput, ignored_Le, seed)) {
        return pos + t * dir;
    }
    return vec3(0.0); // Return zero if no surface was hit
}

// A function to calculate the lighting at a specific point in space.
// This is a one-bounce global illumination calculation.
vec3 calculate_lighting_at_point(vec3 pos, inout uint seed) {
    vec3 ipos = vec3(vol_density_inv_transform * vec4(pos, 1.0));

    float density = lookup_density_trilinear(ipos);
    vec4 rgba = tf_lookup(density * vol_inv_majorant);
    vec3 material_albedo = rgba.rgb;

    vec3 w_i;
    vec4 Le_pdf = sample_environment(rng2(seed), w_i);
    if (Le_pdf.w > 0) {
        float Tr = transmittance(pos, w_i, seed);
        vec3 incoming_light = Tr * Le_pdf.rgb / Le_pdf.w;

        return incoming_light * material_albedo;
    }
    return vec3(0.0);
}

vec3 calculate_color_full_gi(vec3 pos, vec3 view_dir, inout uint seed) {
    vec3 ipos = vec3(vol_density_inv_transform * vec4(pos, 1.0));
    float cone_angle_rad = 1.5; // 90deg ish
    vec3 random_dir = sample_cone(view_dir, cone_angle_rad, rng2(seed));
    
    vec4 L_out = trace_path_pointcloud(pos, random_dir, seed);

    return L_out.rgb;
}

// ---------------------------------------------------
// main

void main() {
    uint thread_id = gl_GlobalInvocationID.x;
    uint seed = tea(thread_id, u_seed_offset, 32);

    if (u_mode == 0) {
        // --- Phase 1: Point Generation ---

        // 1. Generate a unique, quasi-random ray for this thread
        // We will sample a point on a sphere surrounding the volume's AABB
        vec3 center = (vol_bb_min + vol_bb_max) * 0.5;
        float radius = length(vol_bb_max - center) * 1.5; // Start outside the volume
        
        vec3 ray_origin = center + sample_phase_isotropic(rng2(seed)) * radius;
        vec3 ray_dir = normalize(center + (rng3(seed) - 0.5) * length(vol_bb_max - vol_bb_min) - ray_origin);

        // 2. Trace to find the first surface
        vec3 surface_pos = find_first_surface(ray_origin, ray_dir, seed);

        // 3. If we found a surface, claim a spot in the buffer
        if (surface_pos != vec3(0.0)) {
            uint index = atomicAdd(point_counter, 1);
            if (index < u_target_point_count) {
                points[index].world_pos = vec4(surface_pos, 1.0); // Mark as valid
                points[index].view_dir = vec4(ray_dir, 0.0);
            }
        }
    }
    else if (u_mode == 1) {
        // --- Phase 2: Color Convergence ---
        uint point_index = thread_id;
        if (point_index >= u_target_point_count) return;

        // 1. Read the position of the point this thread is responsible for
        vec3 pos = points[point_index].world_pos.xyz;
        if (points[point_index].world_pos.w == 0.f) return; // Skip invalid points
                
        vec3 view_dir = points[point_index].view_dir.xyz;

        // 2. Calculate one sample of lighting for this point
        vec3 light_sample = calculate_color_full_gi(pos, view_dir, seed);

        // Firefly Clamping
        const float max_sample_brightness = 100.0; // Tweak
        light_sample = clamp(light_sample, vec3(0.0), vec3(max_sample_brightness));

        // We scale up to preserve precision after the decimal point.
        uvec3 light_as_uint = uvec3(light_sample * 1024.0);

        atomicAdd(points[point_index].accumulated_color_r, light_as_uint.r);
        atomicAdd(points[point_index].accumulated_color_g, light_as_uint.g);
        atomicAdd(points[point_index].accumulated_color_b, light_as_uint.b);
        atomicAdd(points[point_index].sample_count, 1);
    }
}