using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using GaussianSplatting.Runtime;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class TestRunner : MonoBehaviour
{
    public GaussianSplatRenderer gs;
    public Camera cam;
    
    public int numTestView = 32;
    public int resX = 2000;
    public int resY = 2000;
    
    public List<NamedGSAsset> gs_assets = new();

    public string outPath = "eval";
    
    
    // State
    private static uint timings = 7;
    private FrameTiming[] frameTimings = new FrameTiming[timings];
    private int currentAsset = 0;
    
    // Start is called before the first frame update
    void Start()
    {
        if (gs == null) gs = GetComponent<GaussianSplatRenderer>();
        if (cam == null) cam = Camera.main;

        Screen.SetResolution(resX, resY, false);
    }

    // Update is called once per frame
    void Update()
    {
        if (Time.frameCount % (numTestView * (timings + 7)) == 0)
        {
            if (currentAsset < gs_assets.Count)
            {
                var asset = gs_assets[currentAsset];
                StartCoroutine(DoAsset(asset));
                currentAsset += 1;
            }
            else
            {
                Debug.Log("====== Done ======");
#if UNITY_EDITOR
                EditorApplication.ExitPlaymode();
#else
                Application.Quit();
#endif
            }
        }
    }

    IEnumerator DoAsset(NamedGSAsset asset)
    {
        Debug.Log($"Taking images for {asset.name} at f {Time.frameCount}");
        
        gs.m_Asset = asset.asset;
        var imageTimings = new List<ImageTiming>();
        Directory.CreateDirectory($"{outPath}/{asset.name}/renders/");

        for (int curImg = 0; curImg < numTestView; curImg++)
        {
            gs.ActivateCamera(curImg);
            // Render image
            yield return null;
            ScreenCapture.CaptureScreenshot($"{outPath}/{asset.name}/renders/shot_{curImg+1}.png"); // Start index at 1

            yield return null;
            var sw = Stopwatch.StartNew();
            for (var i2 = 0; i2 < timings; i2++)
            {
                cam.Render();
            }
            sw.Stop();
            var span = (float)((double)sw.ElapsedTicks / TimeSpan.TicksPerMillisecond / timings);
            yield return null;
            
            
            // Get timings
            yield return null;
            for (var i = 0; i <= timings; i++)
            {
                FrameTimingManager.CaptureFrameTimings();
                yield return null;
            }
            yield return null;
            FrameTimingManager.GetLatestTimings(timings, frameTimings);
            
            var imgTiming = new ImageTiming
            {
                id = curImg,
                camRenderTime = span,
                frameTimings = frameTimings.Select(ft => new SerializableFrameTimings()
                {
                    cpuFrameTime = ft.cpuFrameTime,
                    cpuMainThreadFrameTime = ft.cpuMainThreadFrameTime,
                    cpuRenderThreadFrameTime = ft.cpuRenderThreadFrameTime,
                    cpuMainThreadPresentWaitTime = ft.cpuMainThreadPresentWaitTime,
                    gpuFrameTime = ft.gpuFrameTime,
                }).ToArray(),
            };
            imageTimings.Add(imgTiming);
        }
        
        var timing = new AssetTiming
        {
            name = asset.name,
            testViews = numTestView,
            samplesPerView = timings,
            imageTimings = imageTimings
        };
        
        var json = JsonUtility.ToJson(timing);
        File.WriteAllText($"{outPath}/{asset.name}/timings.json", json);
    }
}

[System.Serializable]
public struct AssetTiming
{
    public string name;
    public int testViews;
    public uint samplesPerView;
    public List<ImageTiming> imageTimings;
}

[System.Serializable]
public struct ImageTiming
{
    public int id;
    public SerializableFrameTimings[] frameTimings;
    public float camRenderTime;
}

[System.Serializable]
public struct SerializableFrameTimings
{
    public double cpuFrameTime;
    public double cpuMainThreadFrameTime;
    public double cpuRenderThreadFrameTime;
    public double cpuMainThreadPresentWaitTime;
    public double gpuFrameTime;
}

[System.Serializable]
public struct NamedGSAsset
{
    public string name;
    public GaussianSplatAsset asset;
}