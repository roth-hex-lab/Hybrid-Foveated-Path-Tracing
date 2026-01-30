using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using GaussianSplatting.Runtime;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class TestRunnerOurs : MonoBehaviour
{
    public GaussianSplatRenderer gs;
    public GaussianSplatRenderer gs_CamControl;
    public Camera cam;
    public Transform camCutout;
    public PositionFoveatedTex fovTex;
    
    
    public int numTestView = 32;
    public int resX = 2000;
    public int resY = 2000;
    
    public List<NamedControlGSAsset> gs_assets = new();

    public string outPath = "eval";
    
    
    // State
    private static uint timings = 7;
    private FrameTiming[] frameTimings = new FrameTiming[timings];
    private int currentAsset = 0;
    private Material targetMat;
    
    private static readonly int PrevViewProjectionInverseID = Shader.PropertyToID("_Prev_ViewProjectionInverse");
    private static readonly int PastCamPosID = Shader.PropertyToID("_PastCamPos");
    private static readonly int DepthMap = Shader.PropertyToID("_DepthMap");

    
    // Start is called before the first frame update
    void Start()
    {
        if (gs == null) gs = GetComponent<GaussianSplatRenderer>();
        if (cam == null) cam = Camera.main;

        targetMat = fovTex.GetComponent<Renderer>().sharedMaterial;
        
        Screen.SetResolution(resX, resY, false);
    }

    // Update is called once per frame
    void Update()
    {
        if (Time.frameCount % (numTestView * (timings + 10)) == 0)
        {
            if (currentAsset < gs_assets.Count)
            {
                var asset = gs_assets[currentAsset];
                StartCoroutine(DoAsset(asset));
                currentAsset += 1;
            }
            else
            {
                //Debug.Log("====== Done ======");
#if UNITY_EDITOR
                //EditorApplication.ExitPlaymode();
#else
                Application.Quit();
#endif
            }
        }
    }

    IEnumerator DoAsset(NamedControlGSAsset asset)
    {
        Debug.Log($"Taking images for {asset.name} at f {Time.frameCount}. Looking for images in {(string.IsNullOrEmpty(asset.foveatedFolder) ? asset.name : asset.foveatedFolder)}");

        var foveatedViews = Resources.LoadAll(string.IsNullOrEmpty(asset.foveatedFolder) ? asset.name : asset.foveatedFolder, typeof(Texture2D)).Cast<Texture2D>().ToArray();
        Debug.Log($"Loaded {foveatedViews.Length} images");

        var camTransform = cam.transform;
        gs.m_Asset = asset.asset;
        gs_CamControl.m_Asset = asset.controlAsset;
        var imageTimings = new List<ImageTiming>();
        Directory.CreateDirectory($"{outPath}/{asset.name}/renders/");

        for (int curImg = 0; curImg < numTestView; curImg++)
        {
            gs_CamControl.ActivateCamera(curImg);
            
            var currentFovTex = foveatedViews[curImg * 2];
            var currentDepthTex = foveatedViews[curImg * 2 + 1];
            
            Debug.Log($"Current tex name {currentFovTex.name}");
            
            // Position texture
            fovTex.UpdatePosition();
            
            // Position cutout
            camCutout.SetPositionAndRotation(camTransform.position, Quaternion.LookRotation(camTransform.forward, camTransform.up));
            camCutout.Translate(new Vector3(0, 0, -0.04f), Space.Self);
            
            
            // Set depth
            /*
            Span<byte> bytes = currentDepthTex.GetRawTextureData();
            Span<float> floats = MemoryMarshal.Cast<byte, float>(bytes);
                
            BurstDepthProcessor.Cleanup(floats, 512);
                
            var depth = new Texture2D(512, 512, TextureFormat.RFloat, false);
            depth.LoadRawTextureData(message.depth);
            depth.Apply();
            */
            targetMat.SetTexture(DepthMap, currentDepthTex);
            
            // Set Image
            targetMat.mainTexture = currentFovTex;
            
            // Set Transform
            float aspectRatio = 1;
            Matrix4x4 pastProjectionMatrix = Matrix4x4.Perspective(20, aspectRatio, cam.nearClipPlane, cam.farClipPlane);
            Matrix4x4 pastVpInverseMatrix = (pastProjectionMatrix * cam.worldToCameraMatrix).inverse;
            
            targetMat.SetMatrix(PrevViewProjectionInverseID, pastVpInverseMatrix);
            targetMat.SetVector(PastCamPosID, camTransform.position); 
            
            
            // Render image
            yield return null;
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
            
            //Debug.Log("Waiting");
            //yield return new WaitForSeconds(1);
            
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
public struct NamedControlGSAsset
{
    public string name;
    public string foveatedFolder;
    public GaussianSplatAsset asset;
    public GaussianSplatAsset controlAsset;
}