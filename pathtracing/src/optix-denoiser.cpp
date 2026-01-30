#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include "optix-denoiser.h"

NVOptixDenoiser::~NVOptixDenoiser() {
    // Meh let the os do cleanup, prevent destruction race conditions
}

void NVOptixDenoiser::cleanup() {  
    if (!initialized) return;

    CUcontext currentContext;
    cuCtxGetCurrent(&currentContext);
    
    cudaDeviceSynchronize();

    if (denoiser) {
        optixDenoiserDestroy(denoiser);
        denoiser = nullptr;
    }

    if (inputBuffer) cuMemFree(inputBuffer);
    if (outputBuffer) cuMemFree(outputBuffer);
    if (scratchBuffer) cuMemFree(scratchBuffer);
    if (denoiserState) cuMemFree(denoiserState);
    if (guideBuffer) cuMemFree(guideBuffer);
    if (hdrIntensityBuffer) cuMemFree(hdrIntensityBuffer);
    if (albedoBuffer) cuMemFree(albedoBuffer);
    if (previousOutputBuffer) cuMemFree(previousOutputBuffer);
    if (previousOutputInternalGuideLayer) cuMemFree(previousOutputInternalGuideLayer);
    if (outputInternalGuideLayer) cuMemFree(outputInternalGuideLayer);
    
    inputBuffer = 0;
    outputBuffer = 0;
    scratchBuffer = 0;
    denoiserState = 0;
    guideBuffer = 0;
    hdrIntensityBuffer = 0;
    albedoBuffer = 0;
    previousOutputBuffer = 0;
    previousOutputInternalGuideLayer = 0;
    outputInternalGuideLayer = 0;

    if (context) {
        optixDeviceContextDestroy(context);
        context = nullptr;
    }
    
    if (cudaContext) {
        cuCtxDestroy(cudaContext);
        cudaContext = nullptr;
    }
    
    initialized = false;
}

bool NVOptixDenoiser::init(int width, int height, bool use_temporal_aov, bool use_albedo) {
    if (initialized) {
        cleanup();
    }
    
    temporal_mode_active = use_temporal_aov;
    albedo_enabled = use_albedo;

    CUresult cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(cuResult, &errorString);
        std::cerr << "Failed to initialize CUDA: " << errorString << std::endl;
        return false;
    }
    
    CUdevice device;
    cuDeviceGet(&device, 0);
    cuResult = cuCtxCreate(&cudaContext, 0, device);
    if (cuResult != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(cuResult, &errorString);
        std::cerr << "Failed to create CUDA context: " << errorString << std::endl;
        return false;
    }
    
    OptixResult result = optixInit();
    if (result != OPTIX_SUCCESS) {
        std::cerr << "Failed to initialize OptiX" << std::endl;
        return false;
    }

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    options.logCallbackLevel = 3;
    result = optixDeviceContextCreate(cudaContext, &options, &context);
    if (result != OPTIX_SUCCESS) {
        std::cerr << "Failed to create OptiX context" << std::endl;
        cleanup();
        return false;
    }

    OptixDenoiserOptions denoiserOptions = {};
    denoiserOptions.guideAlbedo = albedo_enabled ? 1 : 0;
    denoiserOptions.guideNormal = 0;
    denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

    OptixDenoiserModelKind modelKind = temporal_mode_active ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV : OPTIX_DENOISER_MODEL_KIND_AOV;

    result = optixDenoiserCreate(context, modelKind, &denoiserOptions, &denoiser);
    if (result != OPTIX_SUCCESS) {
        std::cerr << "Failed to create denoiser" << std::endl;
        cleanup();
        return false;
    }

    OptixDenoiserSizes denoiserSizes;
    result = optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiserSizes);
    if (result != OPTIX_SUCCESS) {
        std::cerr << "Failed to compute denoiser resources" << std::endl;
        cleanup();
        return false;
    }

    denoiserStateSizeInBytes = denoiserSizes.stateSizeInBytes;
    scratchSizeInBytes = denoiserSizes.withoutOverlapScratchSizeInBytes;
    
    if (temporal_mode_active) {
        internalGuideLayerPixelSizeInBytes = denoiserSizes.internalGuideLayerPixelSizeInBytes;
        internalGuideLayerSizeInBytes = width * height * internalGuideLayerPixelSizeInBytes;
    }

    cuResult = cuMemAlloc(&denoiserState, denoiserStateSizeInBytes);
    if (cuResult != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(cuResult, &errorString);
        std::cerr << "Failed to allocate denoiser state: " << errorString << std::endl;
        cleanup();
        return false;
    }

    cuResult = cuMemAlloc(&scratchBuffer, scratchSizeInBytes);
    if (cuResult != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(cuResult, &errorString);
        std::cerr << "Failed to allocate scratch buffer: " << errorString << std::endl;
        cleanup();
        return false;
    }

    result = optixDenoiserSetup(denoiser, nullptr, width, height, denoiserState, denoiserStateSizeInBytes, scratchBuffer, scratchSizeInBytes);

    if (result != OPTIX_SUCCESS) {
        std::cerr << "Failed to setup denoiser state" << std::endl;
        cleanup();
        return false;
    }

    bufferSizeInBytes = width * height * 4 * sizeof(float);
    cuResult = cuMemAlloc(&inputBuffer, bufferSizeInBytes);
    if (cuResult != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(cuResult, &errorString);
        std::cerr << "Failed to allocate input buffer: " << errorString << std::endl;
        cleanup();
        return false;
    }

    cuResult = cuMemAlloc(&outputBuffer, bufferSizeInBytes);
    if (cuResult != CUDA_SUCCESS) {
        const char* errorString;
        cuGetErrorString(cuResult, &errorString);
        std::cerr << "Failed to allocate output buffer: " << errorString << std::endl;
        cleanup();
        return false;
    }
    
    if (temporal_mode_active) {
        cuResult = cuMemAlloc(&previousOutputBuffer, bufferSizeInBytes);
        if (cuResult != CUDA_SUCCESS) {
            std::cerr << "Failed to allocate previous output buffer" << std::endl;
            cleanup();
            return false;
        }

        cuResult = cuMemAlloc(&guideBuffer, width * height * 2 * sizeof(float));
        if (cuResult != CUDA_SUCCESS) {
            std::cerr << "Failed to allocate guide buffer" << std::endl;
            cleanup();
            return false;
        }
        
        cuResult = cuMemAlloc(&previousOutputInternalGuideLayer, internalGuideLayerSizeInBytes);
        if (cuResult != CUDA_SUCCESS) {
            std::cerr << "Failed to allocate previous internal guide layer" << std::endl;
            cleanup();
            return false;
        }

        cuResult = cuMemAlloc(&outputInternalGuideLayer, internalGuideLayerSizeInBytes);
        if (cuResult != CUDA_SUCCESS) {
            std::cerr << "Failed to allocate output internal guide layer" << std::endl;
            cleanup();
            return false;
        }
        cuMemsetD8(previousOutputInternalGuideLayer, 0, internalGuideLayerSizeInBytes);
        cuMemsetD8(outputInternalGuideLayer, 0, internalGuideLayerSizeInBytes);
    }
    
    if (albedo_enabled) {
        albedoBufferSizeInBytes = width * height * 3 * sizeof(float);
        cuResult = cuMemAlloc(&albedoBuffer, albedoBufferSizeInBytes);
        if (cuResult != CUDA_SUCCESS) {
            std::cerr << "Failed to allocate albedo buffer" << std::endl;
            cleanup();
            return false;
        }
    }

    cuResult = cuMemAlloc(&hdrIntensityBuffer, sizeof(float) * 3);
    if (cuResult != CUDA_SUCCESS) {
        std::cerr << "Failed to allocate HDR intensity buffer" << std::endl;
        cleanup();
        return false;
    }
    
    initialized = true;
    return true;
}

bool NVOptixDenoiser::denoise(const cppgl::NamedHandle<cppgl::Texture2DImpl>& inputTexture,
                              cppgl::NamedHandle<cppgl::Texture2DImpl>& outputTexture,
                              bool enableTemporal,
                              const cppgl::NamedHandle<cppgl::Texture2DImpl>& motionVectorTexture,
                              const cppgl::NamedHandle<cppgl::Texture2DImpl>& albedoTexture) {
    if (!initialized) return false;

    std::vector<float> hostData(inputTexture->w * inputTexture->h * 4);
    glGetTextureImage(inputTexture->id, 0, GL_RGBA, GL_FLOAT, hostData.size() * sizeof(float), hostData.data());
    cuMemcpyHtoD(inputBuffer, hostData.data(), bufferSizeInBytes);

    if (temporal_mode_active && motionVectorTexture) {
        size_t mv_buffer_size = motionVectorTexture->w * motionVectorTexture->h * 2 * sizeof(float);
        std::vector<float> mv_host_data(motionVectorTexture->w * motionVectorTexture->h * 2);
        glGetTextureImage(motionVectorTexture->id, 0, GL_RG, GL_FLOAT, mv_buffer_size, mv_host_data.data());
        cuMemcpyHtoD(guideBuffer, mv_host_data.data(), mv_buffer_size);
    } else if (temporal_mode_active) {
        cuMemsetD8(guideBuffer, 0, inputTexture->w * inputTexture->h * 2 * sizeof(float));
    }

    if (albedo_enabled && albedoTexture) {
        std::vector<float> albedo_host_data(albedoTexture->w * albedoTexture->h * 3);
        glGetTextureImage(albedoTexture->id, 0, GL_RGB, GL_FLOAT, albedoBufferSizeInBytes, albedo_host_data.data());
        cuMemcpyHtoD(albedoBuffer, albedo_host_data.data(), albedoBufferSizeInBytes);
    }

    if (firstFrame && temporal_mode_active) {
        cuMemcpyDtoD(previousOutputBuffer, inputBuffer, bufferSizeInBytes);
        cuMemsetD8(previousOutputInternalGuideLayer, 0, internalGuideLayerSizeInBytes);
        cuMemsetD8(outputInternalGuideLayer, 0, internalGuideLayerSizeInBytes);
    }

    OptixDenoiserLayer layer = {};
    layer.input.data = inputBuffer;
    layer.input.width = inputTexture->w;
    layer.input.height = inputTexture->h;
    layer.input.rowStrideInBytes = inputTexture->w * 4 * sizeof(float);
    layer.input.pixelStrideInBytes = 4 * sizeof(float);
    layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    layer.output.data = outputBuffer;
    layer.output.width = inputTexture->w;
    layer.output.height = inputTexture->h;
    layer.output.rowStrideInBytes = inputTexture->w * 4 * sizeof(float);
    layer.output.pixelStrideInBytes = 4 * sizeof(float);
    layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    if (temporal_mode_active) {
        layer.previousOutput.data = previousOutputBuffer;
        layer.previousOutput.width = inputTexture->w;
        layer.previousOutput.height = inputTexture->h;
        layer.previousOutput.rowStrideInBytes = inputTexture->w * 4 * sizeof(float);
        layer.previousOutput.pixelStrideInBytes = 4 * sizeof(float);
        layer.previousOutput.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    }

    OptixDenoiserGuideLayer guideLayer = {};

    if (albedo_enabled) {
        guideLayer.albedo.data = albedoBuffer;
        guideLayer.albedo.width = albedoTexture->w;
        guideLayer.albedo.height = albedoTexture->h;
        guideLayer.albedo.rowStrideInBytes = albedoTexture->w * 3 * sizeof(float);
        guideLayer.albedo.pixelStrideInBytes = 3 * sizeof(float);
        guideLayer.albedo.format = OPTIX_PIXEL_FORMAT_FLOAT3;
    }
    
    if (temporal_mode_active) {
        guideLayer.flow.data = guideBuffer;
        guideLayer.flow.width = motionVectorTexture ? motionVectorTexture->w : inputTexture->w;
        guideLayer.flow.height = motionVectorTexture ? motionVectorTexture->h : inputTexture->h;
        guideLayer.flow.rowStrideInBytes = guideLayer.flow.width * 2 * sizeof(float);
        guideLayer.flow.pixelStrideInBytes = 2 * sizeof(float);
        guideLayer.flow.format = OPTIX_PIXEL_FORMAT_FLOAT2;

        guideLayer.previousOutputInternalGuideLayer.data = previousOutputInternalGuideLayer;
        guideLayer.previousOutputInternalGuideLayer.width = inputTexture->w;
        guideLayer.previousOutputInternalGuideLayer.height = inputTexture->h;
        guideLayer.previousOutputInternalGuideLayer.rowStrideInBytes = inputTexture->w * internalGuideLayerPixelSizeInBytes;
        guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes = internalGuideLayerPixelSizeInBytes;
        guideLayer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

        guideLayer.outputInternalGuideLayer.data = outputInternalGuideLayer;
        guideLayer.outputInternalGuideLayer.width = inputTexture->w;
        guideLayer.outputInternalGuideLayer.height = inputTexture->h;
        guideLayer.outputInternalGuideLayer.rowStrideInBytes = inputTexture->w * internalGuideLayerPixelSizeInBytes;
        guideLayer.outputInternalGuideLayer.pixelStrideInBytes = internalGuideLayerPixelSizeInBytes;
        guideLayer.outputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
    }

    OptixResult intensityResult = optixDenoiserComputeIntensity(denoiser, nullptr, &layer.input, hdrIntensityBuffer, scratchBuffer, scratchSizeInBytes);
    if (intensityResult != OPTIX_SUCCESS) {
        std::cerr << "Failed to compute HDR intensity" << std::endl;
        hdrIntensityBuffer = 0;
    }

    OptixDenoiserParams params = {};
    params.blendFactor = 0.02f;
    params.hdrIntensity = hdrIntensityBuffer;
    if(temporal_mode_active)
        params.temporalModeUsePreviousLayers = !firstFrame ? 1 : 0;

    OptixResult result = optixDenoiserInvoke(denoiser, nullptr, &params, denoiserState, denoiserStateSizeInBytes, &guideLayer, &layer, 1, 0, 0, scratchBuffer, scratchSizeInBytes);
    
    if (result != OPTIX_SUCCESS) {
        std::cerr << "Failed to denoise image" << std::endl;
        return false;
    }

    std::vector<float> outputData(inputTexture->w * inputTexture->h * 4);
    cuMemcpyDtoH(outputData.data(), outputBuffer, bufferSizeInBytes);
    glTextureSubImage2D(outputTexture->id, 0, 0, 0, outputTexture->w, outputTexture->h, GL_RGBA, GL_FLOAT, outputData.data());

    if(temporal_mode_active){
        cuMemcpyDtoD(previousOutputBuffer, outputBuffer, bufferSizeInBytes);
        std::swap(previousOutputInternalGuideLayer, outputInternalGuideLayer);
    }

    firstFrame = false;

    return true;
}

void NVOptixDenoiser::optixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[OptiX " << level << "][" << tag << "]: " << message << std::endl;
}