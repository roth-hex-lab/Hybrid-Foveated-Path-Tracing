using System;
using System.Diagnostics;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public static class BurstDepthProcessor
{
    public static void Cleanup(Span<float> depthData, int texSize, float minValidDepth = 0.01f, int maxFillDistance = 12)
    {
        var sourceDataNat = new NativeArray<float>(depthData.ToArray(), Allocator.TempJob);
        var processedDataNat = new NativeArray<float>(sourceDataNat.Length, Allocator.TempJob);
        
        try
        {
            /*
            var fillJob = new FillInvalidValuesJob
            {
                DepthData = processedDataNat,
                TexSize = texSize,
                MinValidDepth = minValidDepth
            };
            */
            
            var fillJob = new DistanceBasedFillJob
            {
                DepthData = processedDataNat,
                TexSize = texSize,
                MinValidDepth = minValidDepth,
                MaxFillDistance = maxFillDistance
            };
            
            var medianJob = new MedianFilterJob
            {
                OriginalData = sourceDataNat,
                OutputData = processedDataNat,
                TexSize = texSize,
                MinValidDepth = minValidDepth
            };
            JobHandle jobA = medianJob.Schedule(texSize, 32);

            var combinedHandle = fillJob.Schedule(jobA);
            
            combinedHandle.Complete();
        }
        finally
        {
            processedDataNat.AsSpan().CopyTo(depthData);

            sourceDataNat.Dispose();
            processedDataNat.Dispose();
        }
    }

    ///  Median Filter
    /// Also sometimes bugs out, need to investigate
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.Standard, FloatMode = FloatMode.Fast)]
    private struct MedianFilterJob : IJobParallelFor
    {
        [ReadOnly]
        [NativeDisableParallelForRestriction]
        public NativeArray<float> OriginalData;


        [WriteOnly]
        [NativeDisableParallelForRestriction]
        public NativeArray<float> OutputData;
        
        public int TexSize;
        public float MinValidDepth;

        private const float FilterThreshold = 0.007f; // James Bond Vibes
        private const int Radius = 1; // Going up is a little buggy?
        private const int KernelSize = (2 * Radius + 1) * (2 * Radius + 1);

        public void Execute(int y)
        {
            if (y < Radius || y >= TexSize - Radius)            
            {
                for (int x = 0; x < TexSize; x++)
                {
                    OutputData[y * TexSize + x] = OriginalData[y * TexSize + x];
                }
                return;
            }

            var neighbors = new NativeArray<float>(KernelSize, Allocator.Temp);
            
            for (int x = 1; x < TexSize - 1; x++)
            {
                int centerIndex = y * TexSize + x;
                float centerDepth = OriginalData[centerIndex];

                if (centerDepth < MinValidDepth)
                {
                    OutputData[centerIndex] = centerDepth;
                    continue;
                }
                
                bool needsFiltering = false;
                if (math.abs(centerDepth - OriginalData[centerIndex - 1]) > centerDepth * FilterThreshold) needsFiltering = true;
                else if (math.abs(centerDepth - OriginalData[centerIndex + 1]) > centerDepth * FilterThreshold) needsFiltering = true;
                else if (math.abs(centerDepth - OriginalData[centerIndex - TexSize]) > centerDepth * FilterThreshold) needsFiltering = true;
                else if (math.abs(centerDepth - OriginalData[centerIndex + TexSize]) > centerDepth * FilterThreshold) needsFiltering = true;

                if (!needsFiltering)
                {
                    OutputData[centerIndex] = centerDepth;
                    continue;
                }
                
                int validCount = 0;
                for (int dy = -Radius; dy <= Radius; dy++)
                {
                    for (int dx = -Radius; dx <= Radius; dx++)
                    {
                        float neighborDepth = OriginalData[(y + dy) * TexSize + (x + dx)];
                        if (neighborDepth >= MinValidDepth)
                        {
                            neighbors[validCount++] = neighborDepth;
                        }
                    }
                }

                if (validCount > KernelSize / 2)
                {
                    for (int i = 1; i < validCount; i++)
                    {
                        float key = neighbors[i];
                        int j = i - 1;
                        while (j >= 0 && neighbors[j] > key)
                        {
                            neighbors[j + 1] = neighbors[j];
                            j--;
                        }
                        neighbors[j + 1] = key;
                    }
                    OutputData[centerIndex] = neighbors[validCount / 2];
                }
                else
                {
                    OutputData[centerIndex] = centerDepth;
                }
            }
            neighbors.Dispose();
        }
    }

    // Fast flood filling
    // This is not working great tho as we fill from the corners, so with big depth differences there are visible discontinuities
    // It's really fast tho
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.Standard, FloatMode = FloatMode.Fast)]
    private struct FillInvalidValuesJob : IJob
    {
        public NativeArray<float> DepthData;
        public int TexSize;
        public float MinValidDepth;

        public void Execute()
        {
            for (int y = 0; y < TexSize; y++)
            {
                for (int x = 0; x < TexSize; x++)
                {
                    int i = y * TexSize + x;
                    if (DepthData[i] < MinValidDepth)
                    {
                        float neighborVal = 0f;
                        if (y > 0) neighborVal = DepthData[i - TexSize];
                        if (x > 0) neighborVal = math.max(neighborVal, DepthData[i - 1]);
                        
                        if (neighborVal >= MinValidDepth) DepthData[i] = neighborVal;
                    }
                }
            }

            for (int y = TexSize - 1; y >= 0; y--)
            {
                for (int x = TexSize - 1; x >= 0; x--)
                {
                    int i = y * TexSize + x;
                    if (DepthData[i] < MinValidDepth)
                    {
                        float neighborVal = 0f;
                        if (y < TexSize - 1) neighborVal = DepthData[i + TexSize];
                        if (x < TexSize - 1) neighborVal = math.max(neighborVal, DepthData[i + 1]);
                        
                        if (neighborVal >= MinValidDepth) DepthData[i] = neighborVal;
                    }
                }
            }
        }
    }
    
    // Fills invalid depth values by finding the closest valid pixel and using its depth.
    // Not super fast, but suitable as a proof of concept. Probably there is a better way to do this...
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.Standard, FloatMode = FloatMode.Fast)]
    private struct DistanceBasedFillJob : IJob
    {
        public NativeArray<float> DepthData;
        public int TexSize;
        public float MinValidDepth;
        public int MaxFillDistance;

        public void Execute()
        {
            var tempData = new NativeArray<float>(DepthData.Length, Allocator.Temp);
            DepthData.CopyTo(tempData);

            for (int iteration = 0; iteration < MaxFillDistance; iteration++)
            {
                bool anyChanges = false;

                for (int y = 0; y < TexSize; y++)
                {
                    for (int x = 0; x < TexSize; x++)
                    {
                        int i = y * TexSize + x;
                        if (tempData[i] >= MinValidDepth)
                            continue;

                        int validNeighbors = 0;
                        float sumDepth = 0f;

                        if (y > 0)
                        {
                            float upDepth = tempData[i - TexSize];
                            if (upDepth >= MinValidDepth)
                            {
                                sumDepth += upDepth;
                                validNeighbors++;
                            }
                        }

                        if (y < TexSize - 1)
                        {
                            float downDepth = tempData[i + TexSize];
                            if (downDepth >= MinValidDepth)
                            {
                                sumDepth += downDepth;
                                validNeighbors++;
                            }
                        }

                        if (x > 0)
                        {
                            float leftDepth = tempData[i - 1];
                            if (leftDepth >= MinValidDepth)
                            {
                                sumDepth += leftDepth;
                                validNeighbors++;
                            }
                        }

                        if (x < TexSize - 1)
                        {
                            float rightDepth = tempData[i + 1];
                            if (rightDepth >= MinValidDepth)
                            {
                                sumDepth += rightDepth;
                                validNeighbors++;
                            }
                        }

                        if (validNeighbors > 0)
                        {
                            DepthData[i] = sumDepth / validNeighbors;
                            anyChanges = true;
                        }
                    }
                }

                if (anyChanges)
                {
                    DepthData.CopyTo(tempData);
                }
                else
                {
                    break;
                }
            }

            tempData.Dispose();
        }
    }
}