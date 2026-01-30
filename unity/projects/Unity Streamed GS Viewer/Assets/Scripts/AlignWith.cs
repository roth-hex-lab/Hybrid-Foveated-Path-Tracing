using System.Collections;
using System.Collections.Generic;
using MessagePack.Unity;
using UnityEngine;
using UnityEngine.PlayerLoop;

public class AlignWithImage : MonoBehaviour
{
    public PathtracingConnector cn;
    public Vector3 offset;

    void OnEnable()
    {
        cn.OnImageReceived += UpdatePose;
    }

    void OnDisable()
    {
        cn.OnImageReceived -= UpdatePose;
    }
    
    void UpdatePose(ExtendedCamPose extCamPose)
    {
        var camPose = extCamPose.camPose;
        transform.SetPositionAndRotation(camPose.position, Quaternion.LookRotation(camPose.dir, camPose.up));
        transform.Translate(offset, Space.Self);
    }
}
