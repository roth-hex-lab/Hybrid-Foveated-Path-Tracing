#pragma once

#include <cppgl.h>
#include <voldata.h>
#include "environment.h"
#include "transferfunc.h"
#include "optix-denoiser.h"
#include <chrono> // For debounce timing

struct BrickGridGL {
    cppgl::Texture3D indirection;
    cppgl::Texture3D range;
    cppgl::Texture3D atlas;
    glm::mat4 transform;
};

struct PointCloudElement {
    alignas(16) glm::vec4 world_pos;
    alignas(16) glm::vec4 view_dir;
    uint32_t accumulated_color_r;
    uint32_t accumulated_color_g;
    uint32_t accumulated_color_b;
    uint32_t sample_count;
};


float compute_foveation(float camFovDeg, float aspect, float fovea_deg, float fovea_mult, glm::vec2 fovea_center);

struct RendererOpenGL {
    ~RendererOpenGL();

    // Renderer interface
    void init(bool interactive = true);
    void resize(uint32_t w, uint32_t h);
    void commit();
    void reset();

    void trace();
    void traceMulti(uint32_t sppx);

    void process_frame(bool do_unfoveate = false);
    void draw();
    void recreate_denoiser();
    void save_frame(const std::string& filename, bool with_alpha = true, bool blocking = true);
    void save_depth(const std::string& filename, bool blocking = true);

    // helper to convert brick grid to OpenGL 3D textures
    BrickGridGL brick_grid_to_textures(const std::shared_ptr<voldata::BrickGrid>& grid);
    // scale and move volume to fit into [-0.5, 0.5] unit cube
    void scale_and_move_to_unit_cube();

    // Creates a colmap compatible pointcloud from object
    void to_pointcloud_gpu(int targetNumPoints, int convergenceSamples, std::string path = "points3D.txt");

    // Save debug textures
    void save_motion_vectors_to_file(const std::string& filepath);
    void save_albedo_to_file(const std::string& filepath);
    void save_depth_to_file(const std::string& filepath);

    // Denoiser
    NVOptixDenoiser optixDenoiser;
    bool show_denoised = false;
    
    // General settings
    int sample = 0;
    int sppx = 1024;
    int seed = 42;
    int bounces = 100;
    float tonemap_exposure = 5.f;
    float tonemap_gamma = 2.2f;
    bool tonemapping = true;
    bool show_environment = true;

    bool denoising_enabled = false;
    bool denoise_temporal_enabled = false;
    bool generate_motionvec = false;
    bool generate_albedo = false;
    bool generate_depth = false;

    std::string loaded_lut;


    // Foveation settings
    bool use_foveation = false;
    float fovea_deg = 5.0f;
    float fovea_mult = 2.0f;
    glm::vec2 fovea_center = glm::vec2(0, 0);

    // Volume settings
    glm::vec3 albedo = glm::vec3(0.9);  // volume albedo
    float phase = 0.f;                  // volume phase (henyey-greenstein g parameter)
    float density_scale = 1.f;          // volume density scaling factor
    float emission_scale = 100.f;       // volume emission scaling factor
    float hounsfieldCutoff = 0.200f;

    // Members for debouncing reinitialization during window resize.
    uint32_t pendingWidth = 0;
    uint32_t pendingHeight = 0;
    bool resizePending = false;
    std::chrono::steady_clock::time_point lastResizeTime;

    // OpenGL data
    cppgl::Shader trace_shader, trace_shader_tf, trace_shader_tf_multi, tonemap_shader, pointcloud_shader;
    cppgl::Texture2D color_texture;
    cppgl::Texture2D depth_texture;
    cppgl::Texture2D postprocess_hdr_A;
    cppgl::Texture2D postprocess_depth_B; 
    cppgl::Texture2D final_ldr_texture;

    cppgl::Texture2D denoised_output;
    cppgl::Texture2D motion_vector_texture;
    cppgl::Texture2D albedo_texture;

    glm::mat4 prev_view_projection_matrix;
    std::vector<BrickGridGL> density_grids;
    std::vector<BrickGridGL> emission_grids;
    float majorant_emission = 0.f;


    // Volume data
    std::shared_ptr<voldata::Volume> volume;

    // Volume clip planes
    glm::vec3 vol_clip_min = glm::vec3(0.f);
    glm::vec3 vol_clip_max = glm::vec3(1.f);

    // Scene data
    std::shared_ptr<Environment> environment;
    std::shared_ptr<TransferFunction> transferfunc;

    GLuint postprocess_fbo = 0;
    GLuint final_ldr_fbo = 0;
    GLuint depth_fbo = 0;

private:
    void denoise_if_ready();
    const cppgl::Texture2D& get_hdr_input();
    void init_framebuffers(); 
    void cleanup_framebuffers(); 
};
