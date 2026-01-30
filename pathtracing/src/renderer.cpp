#include "renderer.h"
#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>

using namespace cppgl;

static int RESIZE_DEBOUNCE_MS = 200;

// -----------------------------------------------------------
// helper funcs

void blit(const Texture2D& tex) {
    static Shader blit_shader = Shader("blit", "shader/quad.vs", "shader/blit.fs");
    blit_shader->bind();
    blit_shader->uniform("tex", tex, 0);
    Quad::draw();
    blit_shader->unbind();
}

void tonemap(const Texture2D& tex, float exposure, float gamma) {
    static Shader tonemap_shader = Shader("tonemap", "shader/quad.vs", "shader/tonemap.fs");
    tonemap_shader->bind();
    tonemap_shader->uniform("tex", tex, 0);
    tonemap_shader->uniform("exposure", exposure);
    tonemap_shader->uniform("gamma", gamma);
    Quad::draw();
    tonemap_shader->unbind();
}

void unfoveate(const Texture2D& tex, glm::ivec2 res , float target_radius, float mult, glm::vec2 fovea_center) {
    static Shader unfoveate_shader = Shader("unfoveate", "shader/quad.vs", "shader/unfoveate.fs");
    unfoveate_shader->bind();
    unfoveate_shader->uniform("foveated_texture", tex, 0);
    unfoveate_shader->uniform("resolution", glm::vec2(res.x, res.y));
    unfoveate_shader->uniform("r_fovea_target", target_radius);
    unfoveate_shader->uniform("fovea_mult", mult);
    unfoveate_shader->uniform("u_fovea_center", fovea_center);
    Quad::draw();
    unfoveate_shader->unbind();
}

// This is a direct C++ port of the GLSL hable function using GLM.
glm::vec3 hable(glm::vec3 rgb) {
    const float A = 0.15f;
    const float B = 0.50f;
    const float C = 0.10f;
    const float D = 0.20f;
    const float E = 0.02f;
    const float F = 0.30f;
    // GLM's component-wise multiplication works just like GLSL's.
    return ((rgb * (A * rgb + C * B) + D * E) / (rgb * (A * rgb + B) + D * F)) - E / F;
}

// This is a direct C++ port of the GLSL hable_tonemap function.
glm::vec3 hable_tonemap_cpp(glm::vec3 rgb, float exposure) {
    const float W = 11.2f;
    return hable(exposure * rgb) / hable(glm::vec3(W));
}

// -----------------------------------------------------------
// OpenGL renderer

void RendererOpenGL::init_framebuffers() {
    // Cleanup old FBOs if they exist (e.g., on context recreation)
    cleanup_framebuffers();

    // Create and configure the FBO for post-processing HDR->HDR effects (like unfoveate)
    glGenFramebuffers(1, &postprocess_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, postprocess_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postprocess_hdr_A->id, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Error: Postprocess FBO is not complete!" << std::endl;

    glGenFramebuffers(1, &depth_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postprocess_depth_B->id, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Error: Depth FBO is not complete!" << std::endl;
    
    // Create and configure the FBO for the final HDR->LDR tonemapping pass
    glGenFramebuffers(1, &final_ldr_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, final_ldr_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, final_ldr_texture->id, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Error: Final LDR FBO is not complete!" << std::endl;


    // Unbind to return to the default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RendererOpenGL::init(bool interactive) {
    // load default volume
    if (!volume)
        volume = std::make_shared<voldata::Volume>();

    // load default environment map
    if (!environment) {
        glm::vec3 color(1.f);
        environment = std::make_shared<Environment>(Texture2D("background", 1, 1, GL_RGB32F, GL_RGB, GL_FLOAT, &color.x));
    }
    // compile shaders
    if (!trace_shader)
        trace_shader = Shader("trace", "shader/pathtracer_brick.glsl");
    if (!trace_shader_tf)
        trace_shader_tf = Shader("trace_tf", "shader/pathtracer_brick_tf.glsl");
    if (!trace_shader_tf_multi)
        trace_shader_tf_multi = Shader("trace_tf_multi", "shader/pathtracer_brick_tf_multi_foveated.glsl");

    // point cloud generation shader
    pointcloud_shader = Shader("pointcloud", "shader/point_generator.glsl");


    // setup color texture
    const glm::ivec2 res = Context::resolution();
    if (!color_texture) {
        color_texture = Texture2D("color", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    }

    if (!motion_vector_texture) {
        motion_vector_texture = Texture2D("motion_vectors", res.x, res.y, GL_RGBA32F, GL_RG, GL_FLOAT);
    }
    if (!albedo_texture) {
        albedo_texture = Texture2D("albedo", res.x, res.y, GL_RGBA32F, GL_RGB, GL_FLOAT);
    }
    if (!depth_texture) {
        depth_texture = Texture2D("depth", res.x, res.y, GL_RGBA32F, GL_RED, GL_FLOAT);
    }
    if (!postprocess_depth_B) {
        postprocess_depth_B = Texture2D("post_depth_B", res.x, res.y, GL_RGBA32F, GL_RED, GL_FLOAT);
    }
    if (!denoised_output) {
        denoised_output = Texture2D("denoised", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    }
    if (!postprocess_hdr_A) {
        postprocess_hdr_A = Texture2D("post_hdr_A", res.x, res.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    }
    if (!final_ldr_texture) {
        final_ldr_texture = Texture2D("final_ldr", res.x, res.y, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
    }

    recreate_denoiser();

    Camera cam = current_camera();
    cam->update(); // Ensure matrices are valid before we access them
    prev_view_projection_matrix = cam->proj * cam->view;

    init_framebuffers(); 
}

void RendererOpenGL::cleanup_framebuffers() {
    if (postprocess_fbo) {
        glDeleteFramebuffers(1, &postprocess_fbo);
        postprocess_fbo = 0;
    }
    if (depth_fbo) {
        glDeleteFramebuffers(1, &depth_fbo);
        depth_fbo = 0;
    }
    if (final_ldr_fbo) {
        glDeleteFramebuffers(1, &final_ldr_fbo);
        final_ldr_fbo = 0;
    }
}

RendererOpenGL::~RendererOpenGL() {
    cleanup_framebuffers();
}


void RendererOpenGL::recreate_denoiser() {
    const glm::ivec2 res = Context::resolution();
    if (!optixDenoiser.init(res.x, res.y, denoise_temporal_enabled, generate_albedo)) {
        std::cerr << "Failed to initialize OptiX denoiser" << std::endl;
    }
    reset();
}


void RendererOpenGL::resize(uint32_t w, uint32_t h) {
    if (color_texture) color_texture->resize(w, h);
    if (denoised_output) denoised_output->resize(w, h);
    if (postprocess_hdr_A) postprocess_hdr_A->resize(w, h);
    if (final_ldr_texture) final_ldr_texture->resize(w, h);
    if (motion_vector_texture) motion_vector_texture->resize(w, h);
    if (albedo_texture) albedo_texture->resize(w, h);
    if (depth_texture) depth_texture->resize(w, h);
    if (postprocess_depth_B) postprocess_depth_B->resize(w, h);

    // Only reinitialize optix buffers after some timout so window resize is smooth
    pendingWidth = w;
    pendingHeight = h;
    resizePending = true;
    lastResizeTime = std::chrono::steady_clock::now();
}

void RendererOpenGL::commit() {
    density_grids.clear();
    emission_grids.clear();
    majorant_emission = 0.f;
    std::cout << "Preparing brick grids for OpenGL..." << std::endl;
    for (const auto& frame : volume->grids) {
        voldata::Volume::GridPtr density_grid = frame.at("density");
        auto brick_grid = voldata::Volume::to_brick_grid(density_grid);
        density_grids.push_back(brick_grid_to_textures(brick_grid));
        voldata::Volume::GridPtr emission_grid;
        for (const auto& name : { "flame", "flames", "temperature" }) {
            if (frame.find(name) != frame.end()) {
                emission_grid = frame.at(name);
                break;
            }
        }
        if (emission_grid) {
            emission_grids.push_back(brick_grid_to_textures(voldata::Volume::to_brick_grid(emission_grid)));
            majorant_emission = std::max(majorant_emission, emission_grid->minorant_majorant().second);
        }
    }
}

void RendererOpenGL::trace() {
    // select shader
    Shader& shader = transferfunc ? trace_shader_tf : trace_shader;

    // bind
    shader->bind();
    color_texture->bind_image(0, GL_READ_WRITE, GL_RGBA32F);
    motion_vector_texture->bind_image(1, GL_WRITE_ONLY, GL_RG32F);

    // Additional shit
    shader->uniform("cutoff", hounsfieldCutoff);

    // uniforms
    uint32_t tex_unit = 0;
    shader->uniform("bounces", bounces);
    shader->uniform("seed", seed);
    shader->uniform("show_environment", show_environment ? 1 : 0);
    shader->uniform("optimization", 0);

    // camera
    shader->uniform("cam_pos", current_camera()->pos);
    shader->uniform("cam_fov", current_camera()->fov_degree);
    shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));

    shader->uniform("current_view_projection_matrix", current_camera()->proj * current_camera()->view);
    shader->uniform("prev_view_projection_matrix", prev_view_projection_matrix);    

    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->minorant_majorant();
    shader->uniform("vol_bb_min", bb_min + vol_clip_min * (bb_max - bb_min));
    shader->uniform("vol_bb_max", bb_min + vol_clip_max * (bb_max - bb_min));
    shader->uniform("vol_minorant", min * density_scale);
    shader->uniform("vol_majorant", maj * density_scale);
    shader->uniform("vol_inv_majorant", 1.f / (maj * density_scale));
    shader->uniform("vol_albedo", albedo);
    shader->uniform("vol_phase_g", phase);
    shader->uniform("vol_density_scale", density_scale);
    shader->uniform("vol_emission_scale", emission_scale);
    shader->uniform("vol_emission_norm", majorant_emission > 0.f ? 1.f / majorant_emission : 1.f);
    // density brick grid data
    const BrickGridGL density = density_grids[volume->grid_frame_counter];
    shader->uniform("vol_density_transform", volume->transform * density.transform);
    shader->uniform("vol_density_inv_transform", glm::inverse(volume->transform * density.transform));
    shader->uniform("vol_density_indirection", density.indirection, tex_unit++);
    shader->uniform("vol_density_range", density.range, tex_unit++);
    shader->uniform("vol_density_atlas", density.atlas, tex_unit++);
    // emission brick grid data
    if (volume->grid_frame_counter < emission_grids.size()) {
        const BrickGridGL emission = emission_grids[volume->grid_frame_counter];
        shader->uniform("vol_emission_transform", volume->transform * emission.transform);
        shader->uniform("vol_emission_inv_transform", glm::inverse(volume->transform * emission.transform));
        shader->uniform("vol_emission_indirection", emission.indirection, tex_unit++);
        shader->uniform("vol_emission_range", emission.range, tex_unit++);
        shader->uniform("vol_emission_atlas", emission.atlas, tex_unit++);
    }
    // transfer function
    if (transferfunc) transferfunc->set_uniforms(shader, 4);
    // environment
    shader->uniform("env_transform", environment->transform);
    shader->uniform("env_inv_transform", glm::inverse(environment->transform));
    shader->uniform("env_strength", environment->strength);
    shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    shader->uniform("env_envmap", environment->envmap, tex_unit++);
    shader->uniform("env_impmap", environment->impmap, tex_unit++);

    // trace
    const glm::ivec2 resolution = Context::resolution();
    shader->uniform("current_sample", ++sample);
    shader->uniform("resolution", resolution);
    shader->dispatch_compute(resolution.x, resolution.y);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // unbind
    color_texture->unbind_image(0);
    motion_vector_texture->unbind_image(1);
    shader->unbind();
}

void RendererOpenGL::traceMulti(uint32_t traceSamples) {
    // select shader
    Shader& shader = trace_shader_tf_multi;
    const glm::ivec2 resolution = Context::resolution();

    // bind
    shader->bind();
    color_texture->bind_image(0, GL_WRITE_ONLY, GL_RGBA32F);

    shader->uniform("generate_motionvec", generate_motionvec ? 1 : 0);
    shader->uniform("generate_albedo", generate_albedo ? 1 : 0);
    shader->uniform("generate_depth", generate_depth ? 1 : 0);
    if (generate_motionvec) {
        motion_vector_texture->bind_image(1, GL_WRITE_ONLY, GL_RGBA32F);
    }
    if (generate_albedo) {
        albedo_texture->bind_image(2, GL_WRITE_ONLY, GL_RGBA32F);
    }
    if (generate_depth) {
        depth_texture->bind_image(3, GL_WRITE_ONLY, GL_RGBA32F);
    }

    Camera cam = current_camera();
    float r_fovea_target = compute_foveation(cam->fov_degree, cam->aspect_ratio(), fovea_deg, fovea_mult, fovea_center);

    // Additional shit
    shader->uniform("cutoff", hounsfieldCutoff);

    // uniforms
    uint32_t tex_unit = 0;
    shader->uniform("bounces", bounces);
    shader->uniform("seed", seed);
    shader->uniform("show_environment", show_environment ? 1 : 0);
    shader->uniform("optimization", 0);

    // camera
    shader->uniform("cam_pos", current_camera()->pos);
    shader->uniform("cam_fov", current_camera()->fov_degree);
    shader->uniform("cam_transform", glm::inverse(glm::mat3(current_camera()->view)));

    shader->uniform("current_view_projection_matrix", current_camera()->proj * current_camera()->view);
    shader->uniform("prev_view_projection_matrix", prev_view_projection_matrix);

    // foveation uniforms
    shader->uniform("u_use_foveation", use_foveation ? 1 : 0);
    shader->uniform("u_r_fovea_target", r_fovea_target);
    shader->uniform("u_fovea_mult", fovea_mult);
    shader->uniform("u_fovea_center", fovea_center);

    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->minorant_majorant();
    shader->uniform("vol_bb_min", bb_min + vol_clip_min * (bb_max - bb_min));
    shader->uniform("vol_bb_max", bb_min + vol_clip_max * (bb_max - bb_min));
    shader->uniform("vol_minorant", min * density_scale);
    shader->uniform("vol_majorant", maj * density_scale);
    shader->uniform("vol_inv_majorant", 1.f / (maj * density_scale));
    shader->uniform("vol_albedo", albedo);
    shader->uniform("vol_phase_g", phase);
    shader->uniform("vol_density_scale", density_scale);
    shader->uniform("vol_emission_scale", emission_scale);
    shader->uniform("vol_emission_norm", majorant_emission > 0.f ? 1.f / majorant_emission : 1.f);
    // density brick grid data
    const BrickGridGL density = density_grids[volume->grid_frame_counter];
    shader->uniform("vol_density_transform", volume->transform * density.transform);
    shader->uniform("vol_density_inv_transform", glm::inverse(volume->transform * density.transform));
    shader->uniform("vol_density_indirection", density.indirection, tex_unit++);
    shader->uniform("vol_density_range", density.range, tex_unit++);
    shader->uniform("vol_density_atlas", density.atlas, tex_unit++);
    // emission brick grid data
    if (volume->grid_frame_counter < emission_grids.size()) {
        const BrickGridGL emission = emission_grids[volume->grid_frame_counter];
        shader->uniform("vol_emission_transform", volume->transform * emission.transform);
        shader->uniform("vol_emission_inv_transform", glm::inverse(volume->transform * emission.transform));
        shader->uniform("vol_emission_indirection", emission.indirection, tex_unit++);
        shader->uniform("vol_emission_range", emission.range, tex_unit++);
        shader->uniform("vol_emission_atlas", emission.atlas, tex_unit++);
    }
    // transfer function
    if (transferfunc) transferfunc->set_uniforms(shader, 4);
    // environment
    shader->uniform("env_transform", environment->transform);
    shader->uniform("env_inv_transform", glm::inverse(environment->transform));
    shader->uniform("env_strength", environment->strength);
    shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    shader->uniform("env_envmap", environment->envmap, tex_unit++);
    shader->uniform("env_impmap", environment->impmap, tex_unit++);

    // trace
    shader->uniform("resolution", resolution);
    shader->uniform("trace_samples", traceSamples);

    shader->dispatch_compute(resolution.x, resolution.y);

    sample = traceSamples;

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // unbind
    color_texture->unbind_image(0);
    if (generate_motionvec) {
        motion_vector_texture->unbind_image(1);
    }
    if (generate_albedo) {
        albedo_texture->unbind_image(2);
    }
    if (generate_depth) {
        depth_texture->unbind_image(3);
    }

    shader->unbind();
    this->prev_view_projection_matrix = current_camera()->proj * current_camera()->view;
}

const cppgl::Texture2D& RendererOpenGL::get_hdr_input() {
    return (denoising_enabled && show_denoised) ? denoised_output : color_texture;
}

// Helper: Denoises the raw render if the sample count is high enough.
void RendererOpenGL::denoise_if_ready() {
    show_denoised = false;
    if (!denoising_enabled || sample < sppx) return;
    
    Texture2D mv_tex, albedo_tex;
    if (generate_motionvec) mv_tex = motion_vector_texture;
    if (generate_albedo) albedo_tex = albedo_texture;
    show_denoised = optixDenoiser.denoise(color_texture, denoised_output, denoise_temporal_enabled, mv_tex, albedo_tex);
}


void RendererOpenGL::process_frame(bool do_unfoveate) {
    if (resizePending) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastResizeTime).count() > RESIZE_DEBOUNCE_MS) {
            if (denoising_enabled) recreate_denoiser();
            resizePending = false;
        }
    }
    
    denoise_if_ready();
    const Texture2D& hdr_input = get_hdr_input();

    const Texture2D* final_hdr_image = &hdr_input;
    if (do_unfoveate) {
        glBindFramebuffer(GL_FRAMEBUFFER, postprocess_fbo);
        const glm::ivec2 res = Context::resolution();
        glViewport(0, 0, res.x, res.y);
        glClear(GL_COLOR_BUFFER_BIT);
        
        float f_target = compute_foveation(current_camera()->fov_degree, current_camera()->aspect_ratio(), fovea_deg, fovea_mult, fovea_center);
        unfoveate(hdr_input, res, f_target, fovea_mult, fovea_center);
        
        final_hdr_image = &postprocess_hdr_A;


        // Also unfoveate depth here if we have any
        if (depth_texture) {
            glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo);
            glViewport(0, 0, res.x, res.y);
            glClear(GL_COLOR_BUFFER_BIT);

            unfoveate(depth_texture, res, f_target, fovea_mult, fovea_center);
            glCopyImageSubData(
                postprocess_depth_B->id, GL_TEXTURE_2D, 0, 0, 0, 0,
                depth_texture->id,       GL_TEXTURE_2D, 0, 0, 0, 0,
                res.x, res.y, 1
            );
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, final_ldr_fbo);
    const glm::ivec2 res = Context::resolution();
    glViewport(0, 0, res.x, res.y);
    glClear(GL_COLOR_BUFFER_BIT);

    if (tonemapping) {
        tonemap(*final_hdr_image, tonemap_exposure, tonemap_gamma);
    } else {
        blit(*final_hdr_image);
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void RendererOpenGL::draw() {
    process_frame();

    const glm::ivec2 res = Context::resolution();
    glViewport(0, 0, res.x, res.y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    blit(final_ldr_texture);
}


void RendererOpenGL::save_frame(const std::string& filename, bool with_alpha, bool blocking) {
    const glm::ivec2 size = Context::resolution();
    std::vector<uint8_t> pixels(size.x * size.y * 4);
    glGetTextureImage(final_ldr_texture->id, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.size(), pixels.data());
    const int channels_to_write = with_alpha ? 4 : 3;
    image_store_ldr(filename, pixels.data(), size.x, size.y, channels_to_write, true, !blocking);
}

void RendererOpenGL::save_depth(const std::string& filename, bool blocking) {
    const glm::ivec2 size = Context::resolution();
    std::vector<float> pixels(size.x * size.y);
    glGetTextureImage(depth_texture->id, 0, GL_RED, GL_FLOAT, pixels.size(), pixels.data());
    std::vector<uint8_t> ldr_pixels(size.x * size.y);
    for (size_t i = 0; i < pixels.size(); ++i) {
        ldr_pixels[i] = static_cast<uint8_t>(std::clamp(pixels[i] * 255.0f, 0.0f, 255.0f));
    }
    const fs::path outfile = fs::path(filename).replace_extension(".png");
    image_store_ldr(outfile, ldr_pixels.data(), size.x, size.y, 1, true, !blocking);
}

void RendererOpenGL::reset() {
    sample = 0;
    show_denoised = false;
    optixDenoiser.firstFrame = true;

    auto cam = current_camera();
    cam->update(); 
    prev_view_projection_matrix = cam->proj * cam->view; // Set prev to current
}


BrickGridGL RendererOpenGL::brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& bricks) {
    // create indirection texture

    Texture3D indirection = Texture3D("brick indirection",
            bricks->indirection.stride.x,
            bricks->indirection.stride.y,
            bricks->indirection.stride.z,
            GL_RGB10_A2UI,
            GL_RGBA_INTEGER,
            GL_UNSIGNED_INT_10_10_10_2,
            bricks->indirection.data.data());
    indirection->bind(0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    indirection->unbind();
    // create range texture
    Texture3D range = Texture3D("brick range",
            bricks->range.stride.x,
            bricks->range.stride.y,
            bricks->range.stride.z,
            GL_RG16F,
            GL_RG,
            GL_HALF_FLOAT,
            bricks->range.data.data());
    range->bind(0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // create min/max mipmaps
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, bricks->range_mipmaps.size());
    for (uint32_t i = 0; i < bricks->range_mipmaps.size(); ++i) {
        glTexImage3D(GL_TEXTURE_3D,
                i + 1,
                GL_RG16F,
                bricks->range_mipmaps[i].stride.x,
                bricks->range_mipmaps[i].stride.y,
                bricks->range_mipmaps[i].stride.z,
                0,
                GL_RG,
                GL_HALF_FLOAT,
                bricks->range_mipmaps[i].data.data());
    }
    range->unbind();
    // create atlas texture
    Texture3D atlas = Texture3D("brick atlas",
            bricks->atlas.stride.x,
            bricks->atlas.stride.y,
            bricks->atlas.stride.z,
            GL_COMPRESSED_RED,
            GL_RED,
            GL_UNSIGNED_BYTE,
            bricks->atlas.data.data());    
    atlas->bind(0);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    atlas->unbind();
    // return BrickGridGL
    return BrickGridGL{ indirection, range, atlas, bricks->transform };
}

void RendererOpenGL::scale_and_move_to_unit_cube() {
    // compute max AABB over whole volume (animation)
    glm::vec3 bb_min = glm::vec3(FLT_MAX), bb_max = glm::vec3(FLT_MIN);
    for (const auto frame : volume->grids) {
        const auto grid = frame.at("density");
        bb_min = glm::min(bb_min, glm::vec3(grid->transform * glm::vec4(0, 0, 0, 1)));
        bb_max = glm::max(bb_max, glm::vec3(grid->transform * glm::vec4(glm::vec3(grid->index_extent()), 1)));
    }
    // scale to unit cube and move to origin
    const glm::vec3 extent = bb_max - bb_min;
    const float size = fmaxf(extent.x, fmaxf(extent.y, extent.z));
    if (size != 1.f) {
        std::cout << "Scaling volume to unit cube by " << size << std::endl;
        volume->transform = glm::translate(glm::scale(glm::mat4(1), glm::vec3(1.f / size)), -bb_min - 0.5f * extent);
        density_scale *= size;
    }
}

// Helper to get a random float
inline float randf(float min, float max) {
    static std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(generator);
}

void RendererOpenGL::to_pointcloud_gpu(int targetNumPoints, int convergenceSamples, std::string path) {
    //std::cout << "Creating Pointcloud with " << targetNumPoints << " number of points." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    Shader& shader = pointcloud_shader;

    // Setup
    SSBO point_buffer("PointBuffer", targetNumPoints * sizeof(PointCloudElement));
    point_buffer->clear();
    point_buffer->bind_base(0);

    SSBO atomic_counter("AtomicCounter", sizeof(GLuint));
    atomic_counter->bind_base(1);
    atomic_counter->clear();
    pointcloud_shader->bind();


    // Additional shit
    shader->uniform("cutoff", hounsfieldCutoff);

    // uniforms
    uint32_t tex_unit = 0;
    shader->uniform("bounces", bounces);
    shader->uniform("seed", seed);
    shader->uniform("show_environment", show_environment ? 1 : 0);

    // volume
    const auto [bb_min, bb_max] = volume->AABB();
    const auto [min, maj] = volume->minorant_majorant();
    shader->uniform("vol_bb_min", bb_min + vol_clip_min * (bb_max - bb_min));
    shader->uniform("vol_bb_max", bb_min + vol_clip_max * (bb_max - bb_min));
    shader->uniform("vol_minorant", min * density_scale);
    shader->uniform("vol_majorant", maj * density_scale);
    shader->uniform("vol_inv_majorant", 1.f / (maj * density_scale));
    shader->uniform("vol_albedo", albedo);
    shader->uniform("vol_phase_g", phase);
    shader->uniform("vol_density_scale", density_scale);
    shader->uniform("vol_emission_scale", emission_scale);
    shader->uniform("vol_emission_norm", majorant_emission > 0.f ? 1.f / majorant_emission : 1.f);
    // density brick grid data
    const BrickGridGL density = density_grids[volume->grid_frame_counter];
    shader->uniform("vol_density_transform", volume->transform * density.transform);
    shader->uniform("vol_density_inv_transform", glm::inverse(volume->transform * density.transform));
    shader->uniform("vol_density_indirection", density.indirection, tex_unit++);
    shader->uniform("vol_density_range", density.range, tex_unit++);
    shader->uniform("vol_density_atlas", density.atlas, tex_unit++);
    // emission brick grid data
    if (volume->grid_frame_counter < emission_grids.size()) {
        const BrickGridGL emission = emission_grids[volume->grid_frame_counter];
        shader->uniform("vol_emission_transform", volume->transform * emission.transform);
        shader->uniform("vol_emission_inv_transform", glm::inverse(volume->transform * emission.transform));
        shader->uniform("vol_emission_indirection", emission.indirection, tex_unit++);
        shader->uniform("vol_emission_range", emission.range, tex_unit++);
        shader->uniform("vol_emission_atlas", emission.atlas, tex_unit++);
    }
    // transfer function
    if (transferfunc) transferfunc->set_uniforms(shader, 4);
    // environment
    shader->uniform("env_transform", environment->transform);
    shader->uniform("env_inv_transform", glm::inverse(environment->transform));
    shader->uniform("env_strength", environment->strength);
    shader->uniform("env_imp_inv_dim", glm::vec2(1.f / environment->dimension()));
    shader->uniform("env_imp_base_mip", int(floor(log2(environment->dimension()))));
    shader->uniform("env_envmap", environment->envmap, tex_unit++);
    shader->uniform("env_impmap", environment->impmap, tex_unit++);

    pointcloud_shader->uniform("u_mode", 0); // Set shader to Generation Mode
    pointcloud_shader->uniform("u_target_point_count", targetNumPoints);

    GLuint current_point_count = 0;
    const int threads_per_launch = 1024 * 512;
    while (current_point_count < targetNumPoints) {
        // Each launch gets a new random seed offset to ensure different rays
        pointcloud_shader->uniform("u_seed_offset", rand());
        pointcloud_shader->dispatch_compute(threads_per_launch, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

        // Check our progress
        glGetNamedBufferSubData(atomic_counter->id, 0, sizeof(GLuint), &current_point_count);
        //std::cout << "\rFound " << current_point_count << " / " << targetNumPoints << " points..." << std::endl;
    }

    auto endFind = std::chrono::high_resolution_clock::now();

    pointcloud_shader->uniform("u_mode", 1); // Set shader to Convergence Mode

    for (int i = 0; i < convergenceSamples; ++i) {
        pointcloud_shader->uniform("u_seed_offset", rand());
        // Dispatch one thread for each point
        pointcloud_shader->dispatch_compute((targetNumPoints), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    
    std::vector<PointCloudElement> final_points(targetNumPoints);
    glGetNamedBufferSubData(point_buffer->id, 0, targetNumPoints * sizeof(PointCloudElement), final_points.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedFind = endFind - start;
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Creating pointcloud with " << targetNumPoints << " points took " << elapsedFind.count() << " and with coloring " << convergenceSamples << " samples took: " << elapsed.count() << " ms\n";
    

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << std::endl;
        return;
    }

    int valid_points_written = 0;
    for (int i = 0; i < targetNumPoints; ++i) {
        const auto& point = final_points[i];
        if (point.world_pos.w == 0.f) continue; // Skip points that were never written

        float samples = static_cast<float>(point.sample_count);
        if (samples == 0.f) continue;

        float r = static_cast<float>(point.accumulated_color_r) / 1024.0f;
        float g = static_cast<float>(point.accumulated_color_g) / 1024.0f;
        float b = static_cast<float>(point.accumulated_color_b) / 1024.0f;
        glm::vec3 final_color = glm::vec3(r, g, b) / samples;
        
        glm::vec3 tonemapped_color = hable_tonemap_cpp(final_color, tonemap_exposure);
        final_color = glm::pow(tonemapped_color, glm::vec3(1.0f / tonemap_gamma));
        final_color = glm::clamp(final_color, 0.0f, 1.0f);

        file << valid_points_written++ << " "
             << point.world_pos.x << " " << point.world_pos.y << " " << point.world_pos.z << " "
             << static_cast<int>(final_color.r * 255) << " "
             << static_cast<int>(final_color.g * 255) << " "
             << static_cast<int>(final_color.b * 255) << " 0 0 0\n";
    }

    file.close();
    std::cout << "Successfully saved " << valid_points_written << " points to " << path << std::endl;

    point_buffer->unbind_base(0);
    atomic_counter->unbind_base(1);
    pointcloud_shader->unbind();
}

// Precompute some stuff that is constant so we dont have to do it over and over in the shader
// We adjust the radius close to corners to avoid information loss
static inline float deg2rad_f(float d) { 
    return d * 3.14159265358979323846f / 180.0f; 
}

float compute_foveation(float camFovDeg, float aspect, float fovea_deg, float fovea_mult, glm::vec2 fovea_center_uv)
{
    // The shader normalizes by screen height, so Y goes from -0.5 to 0.5. X from −aspect/2 to aspect/2
    float half_h = 0.5f;
    float half_w = aspect * 0.5f;

    float dist_l = (half_w + fovea_center_uv.x);
    float dist_r = (half_w - fovea_center_uv.x);
    float dist_b = (half_h + fovea_center_uv.y);
    float dist_t = (half_h - fovea_center_uv.y);
    float dist_closest_edge = std::min({dist_l, dist_r, dist_b, dist_t});

    const float tan_half_fov_y = tanf(0.5f * deg2rad_f(camFovDeg));
    float r_fovea_source = (tanf(deg2rad_f(fovea_deg * 0.5f)) / tan_half_fov_y) * half_h;

    float ideal_target_radius = r_fovea_source * fovea_mult;
    // Leave a little space at the adges ( * 0.8) to avoid loss. Possibly this should be configurable?
    float fovea_target = std::max(std::min(ideal_target_radius, dist_closest_edge * 0.8f), .0f);

    return fovea_target;
}

///
/// Save functions for debugging, not normally used
///

void RendererOpenGL::save_motion_vectors_to_file(const std::string& filepath) {
    if (!motion_vector_texture) {
        std::cerr << "Motion vector texture is not initialized." << std::endl;
        return;
    }

    const int w = motion_vector_texture->w;
    const int h = motion_vector_texture->h;
    std::vector<float> mv_data(w * h * 2); // 2 floats per pixel (RG)
    glGetTextureImage(motion_vector_texture->id, 0, GL_RG, GL_FLOAT, mv_data.size() * sizeof(float), mv_data.data());

    std::vector<unsigned char> image_data(w * h * 3); // 3 components for RGB PNG
    
    // Adjust this to make sense dependent on motion
    const float motion_scale = 0.05f;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int mv_idx = (y * w + x) * 2;
            int img_idx = (y * w + x) * 3;

            float dx = mv_data[mv_idx];
            float dy = mv_data[mv_idx + 1];

            float r_float = (dx * motion_scale) + 0.5f;
            float g_float = (dy * motion_scale) + 0.5f; 

            image_data[img_idx + 0] = static_cast<unsigned char>(glm::clamp(r_float, 0.0f, 1.0f) * 255.0f);
            image_data[img_idx + 1] = static_cast<unsigned char>(glm::clamp(g_float, 0.0f, 1.0f) * 255.0f);
            image_data[img_idx + 2] = 0; // Blue channel unused
        }
    }

    auto converted_texture = Texture2D("motion_vectors_converted", w, h, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, image_data.data());
    converted_texture->save_ldr(filepath, true, false);
}

void RendererOpenGL::save_albedo_to_file(const std::string& filepath) {
    if (!albedo_texture) {
        std::cerr << "Albedo texture is not initialized." << std::endl;
        return;
    }
    float exposure = 1.0;

    const int w = albedo_texture->w;
    const int h = albedo_texture->h;
    std::vector<float> float_pixels(w * h * 4); // 4 floats for RGBA
    glGetTextureImage(albedo_texture->id, 0, GL_RGBA, GL_FLOAT, float_pixels.size() * sizeof(float), float_pixels.data());

    std::vector<uint8_t> ldr_pixels(w * h * 4);

    for (int i = 0; i < w * h; ++i) {
        float r = float_pixels[i * 4 + 0];
        float g = float_pixels[i * 4 + 1];
        float b = float_pixels[i * 4 + 2];
        float a = 1; // Ignore alpha

        r = glm::clamp(r * exposure, 0.0f, 1.0f);
        g = glm::clamp(g * exposure, 0.0f, 1.0f);
        b = glm::clamp(b * exposure, 0.0f, 1.0f);
        a = glm::clamp(a * exposure, 0.0f, 1.0f);

        ldr_pixels[i * 4 + 0] = static_cast<uint8_t>(r * 255.0f);
        ldr_pixels[i * 4 + 1] = static_cast<uint8_t>(g * 255.0f);
        ldr_pixels[i * 4 + 2] = static_cast<uint8_t>(b * 255.0f);
        ldr_pixels[i * 4 + 3] = static_cast<uint8_t>(a * 255.0f);
    }

    // 4. Save the LDR data
    image_store_ldr(filepath, ldr_pixels.data(), w, h, 4, true, false); // Assuming cppgl helper
    std::cout << "Saved visualized albedo to " << filepath << std::endl;
}

void RendererOpenGL::save_depth_to_file(const std::string& filepath) {
    if (!depth_texture) {
        std::cerr << "Depth texture is not initialized." << std::endl;
        return;
    }
    
    const int w = depth_texture->w;
    const int h = depth_texture->h;

    std::vector<float> float_pixels(w * h);
    glGetTextureImage(depth_texture->id, 0, GL_RED, GL_FLOAT, float_pixels.size() * sizeof(float), float_pixels.data());

    std::vector<uint8_t> ldr_pixels(w * h); // Grayscale
    for (int i = 0; i < w * h; ++i) {
        float val = float_pixels[i];
        // Invert the depth map (common practice: near=white, far=black)
        if (val > 0.0f) 
            ldr_pixels[i] = static_cast<uint8_t>((1.0f - val) * 255.0f);
        else 
            ldr_pixels[i] = 0;

    }

    // 4. Save the LDR data
    image_store_ldr(filepath, ldr_pixels.data(), w, h, 1, true, false); // 1 channel for grayscale
    std::cout << "Saved visualized depth to " << filepath << std::endl;
}
