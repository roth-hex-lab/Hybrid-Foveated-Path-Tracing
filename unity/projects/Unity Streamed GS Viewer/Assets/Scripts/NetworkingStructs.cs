using MessagePack;
using UnityEngine;
// Sending

[MessagePackObject]
public struct RenderRequest
{
    [Key("type")] public string type;
    [Key("cam")] public CamPose cam;
    [Key("time")] public float time;
}

[MessagePackObject]
public struct CamPose
{
    [Key("pos")] public Vector3 position;
    [Key("dir")] public Vector3 dir;
    [Key("up")] public Vector3 up;
}


// Receiving

[MessagePackObject]
public struct Envelope<T>
{
    [Key("type")] public string type;
    [Key("payload")] public T payload;
}

[MessagePackObject]
public struct PongData
{
    [Key("time")] public float time;
}

[MessagePackObject]
public struct QualityAck
{
    [Key("samples")] public int? samples;
    [Key("bounces")] public int? bounces;
}

[MessagePackObject]
public struct TfAck
{
    [Key("status")] public bool status;
    [Key("message")] public string message;
}

[MessagePackObject]
public struct FrameData
{
    [Key("time")] public float time;
    [Key("l_rgba")] public byte[] left;
    [Key("r_rgba")] public byte[] right;
    [Key("l_depth")] public byte[] l_depth;
    [Key("r_depth")] public byte[] r_depth;
}
