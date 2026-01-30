#pragma once
#include <cuda_runtime.h>
#include <optix_types.h>
#include <optix_denoiser_tiling.h>
#include "cppgl.h"

class NVOptixDenoiser {
private:
    bool initialized = false;
    CUcontext cudaContext = nullptr;
    OptixDeviceContext context = nullptr;
    OptixDenoiser denoiser = nullptr;
    CUdeviceptr denoiserState = 0;
    size_t denoiserStateSizeInBytes = 0;
    CUdeviceptr scratchBuffer = 0;
    size_t scratchSizeInBytes = 0;
    
    // CUDA buffers
    CUdeviceptr inputBuffer = 0;
    CUdeviceptr outputBuffer = 0;
    size_t bufferSizeInBytes = 0;

    CUdeviceptr albedoBuffer = 0;
    size_t albedoBufferSizeInBytes = 0;

    CUdeviceptr previousOutputBuffer = 0;
    CUdeviceptr guideBuffer = 0;       // for guide layer
    CUdeviceptr hdrIntensityBuffer = 0;  // for HDR intensity computation

    CUdeviceptr previousOutputInternalGuideLayer;
    CUdeviceptr outputInternalGuideLayer;
    size_t internalGuideLayerSizeInBytes;
    size_t internalGuideLayerPixelSizeInBytes;
    bool useFirstGuideBuffer;

public:
    NVOptixDenoiser() = default;
    ~NVOptixDenoiser();

    NVOptixDenoiser(const NVOptixDenoiser&) = delete;
    NVOptixDenoiser& operator=(const NVOptixDenoiser&) = delete;
    NVOptixDenoiser(NVOptixDenoiser&&) = delete;
    NVOptixDenoiser& operator=(NVOptixDenoiser&&) = delete;

    bool init(int width, int height, bool use_temporal_aov, bool use_albedo);
    bool denoise(const cppgl::NamedHandle<cppgl::Texture2DImpl>& inputTexture, 
                cppgl::NamedHandle<cppgl::Texture2DImpl>& outputTexture,
                bool enableTemporal,
                const cppgl::NamedHandle<cppgl::Texture2DImpl>& motionVectorTexture,
                const cppgl::NamedHandle<cppgl::Texture2DImpl>& albedoTexture);
    void cleanup();
    bool firstFrame = true;
    bool temporal_mode_active = false;
    bool albedo_enabled = false;

private:
    static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*);
};