#version 130
in vec2 tc;
out vec4 out_col;

uniform sampler2D foveated_texture;
uniform vec2 resolution;
uniform float r_fovea_target; // Radius in foveated (rendered) UV space
uniform float fovea_mult;
uniform vec2 u_fovea_center;  // New uniform

void main() {
    // 1. Build the centered UV for the current fragment (Linear/Physical Space)
    vec2 aspect = vec2(resolution.x / resolution.y, 1.0);
    vec2 uv_fragment = tc * aspect - aspect * 0.5; 
    
    // 2. Calculate vector relative to the fovea
    vec2 rel_vec = uv_fragment - u_fovea_center;
    float r_src = length(rel_vec); // Radius in Linear Source Space

    float r_tgt; // Radius in Texture Space (what we need to find)

    if (fovea_mult <= 1.000001) {
        r_tgt = r_src;
    }
    // Check linear radius against fovea radius in linear space
    // Note: r_fovea_source = r_fovea_target / fovea_mult
    else if (r_src <= (r_fovea_target / fovea_mult)) {
        r_tgt = r_src * fovea_mult;
    } else {
        // Direction unit vector relative to fovea
        vec2 d = (r_src > 1e-8) ? rel_vec / r_src : vec2(1.0, 0.0);

        // 3. Compute edge radius from Fovea Center in direction d
        vec2 screen_dims = aspect * 0.5;
        
        // Ray-Box Intersection logic (same as tracer)
        vec2 dist_to_wall = (sign(d) * screen_dims - u_fovea_center) / (d + vec2(1e-9));
        float Re = min(dist_to_wall.x, dist_to_wall.y); 

        // 4. Solve Quadratic for r_tgt (R)
        // The relationship is defined by interpolation of the *scale factor*.
        // Detailed derivation:
        // r_src = R * scale(R)
        // scale(R) = A + B * (R - Rf) / (Re - Rf)
        // ... leads to: B*R^2 + (A*(Re-Rf) - B*Rf)*R - r_src*(Re - Rf) = 0
        
        float A = 1.0 / fovea_mult;
        float B = 1.0 - A;
        float Rf = r_fovea_target;

        float a_coeff = B;
        float b_coeff = A * (Re - Rf) - B * Rf;
        float c_coeff = -r_src * (Re - Rf);

        float disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
        disc = max(disc, 0.0);
        
        float R = (-b_coeff + sqrt(disc)) / (2.0 * a_coeff);
        
        // Clamp for numerical safety
        r_tgt = clamp(R, Rf, Re);
    }

    // 5. Map the target radius back to a UV coordinate
    vec2 uv_sample_relative = rel_vec;
    if (r_src > 1e-8) {
        uv_sample_relative = rel_vec * (r_tgt / r_src);
    }
    
    // Add fovea center back to get absolute UV
    vec2 uv_sample_abs = u_fovea_center + uv_sample_relative;

    // Convert back to 0..1 texture coords
    vec2 final_tc = uv_sample_abs / aspect + 0.5;

    out_col = texture(foveated_texture, final_tc);
}