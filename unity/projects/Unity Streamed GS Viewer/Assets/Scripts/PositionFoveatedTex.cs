using System.Collections;
using System.Collections.Generic;
using GaussianSplatting.Runtime;
using UnityEngine;

/**
 * Has some testing and older positioning logic. 
 * In the shader based reprojection scheme this is only used for enabling and disabling reprojection in the shader
 */
public class PositionFoveatedTex : MonoBehaviour
{
    public Camera cam;
    public GaussianSplatRenderer gsRender;
    public float targetFoV = 20;
    public float depth = 0.1f;
    public bool calcDepthFromGS = true;
    public bool alwaysUpdatePosition = false;
    public bool rewindPosition = true;
    
    private Material targetMat;
    private static readonly string ReprojectionKeyword = "REPROJECTION_ON";

    void Awake()
    {
        targetMat = GetComponent<Renderer>().material;
    }
    
    void OnEnable()
    {
        if (targetMat == null) return;

        // Disable for editor preview to not look annoying
        if (Application.isPlaying)
        {
            targetMat.EnableKeyword(ReprojectionKeyword);
        }
        else 
        {
            targetMat.DisableKeyword(ReprojectionKeyword);
        }
    }

    void OnDisable()
    {
        if (targetMat != null)
        {
            targetMat.DisableKeyword(ReprojectionKeyword);
        }
    }
    
    void Update()
    {
        if (alwaysUpdatePosition) UpdatePosition();
        if (calcDepthFromGS) depth = Vector3.Distance(cam.transform.position, gsRender.transform.position);
    }

    public void UpdatePosition(CamPose pose = default)
    {
        // Probably theres a more elegant way to do this...
        var originalFov = cam.fieldOfView;
        cam.fieldOfView = targetFoV;

        cam.transform.GetPositionAndRotation(out var originalPos, out var originalRot);

        if (rewindPosition)
        {
            cam.transform.SetPositionAndRotation(pose.position, Quaternion.LookRotation(pose.dir, pose.up));
        }
        
        var center = cam.ViewportToWorldPoint(new Vector3(0.5f, 0.5f, depth));
        var left = cam.ViewportToWorldPoint(new Vector3(0.5f, 0, depth));
        var size = 2 * Vector3.Distance(center, left);
        
        cam.fieldOfView = originalFov;
        cam.transform.SetPositionAndRotation(originalPos, originalRot);


        transform.position = center;
        transform.localScale = new Vector3(size, size, size);
        transform.rotation = Quaternion.LookRotation(cam.transform.forward, cam.transform.up);
    }
}
