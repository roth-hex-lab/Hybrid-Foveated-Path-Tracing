using System.Collections.Generic;
using UnityEngine;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using GaussianSplatting.Runtime;
using MessagePack;
using Debug = UnityEngine.Debug;


public class PathtracingConnector : MonoBehaviour
{
    public string serverIP = "localhost";
    public int serverPort = 9999;
    public int texSize = 512;
    [Range(0.005f, 1)]
    public float maxSendRate = 0.05f;
    
    public Camera cam;
    public GaussianSplatRenderer targetGS;
    public GameObject quad;

    public int targetSppx = 8;
    public int targetBounces = 4;
    private int lastTargetSppx;
    private int lastTargetBounces;

    public string transferFunction;
    private string lastTransferFunction;
    
    // React to new image arrive from this supplied camera position
    public Action<ExtendedCamPose> OnImageReceived;
    
    private TcpClient client;
    private NetworkStream stream;
    private bool isConnected = false;
    private bool shouldStop = false;
    private CancellationTokenSource cancellationTokenSource;
    private Task receiveTask;
    private ConcurrentQueue<System.Action> mainThreadActions = new ConcurrentQueue<System.Action>();
    
    private Material targetMat;
    private static readonly int PrevViewProjectionInverseID = Shader.PropertyToID("_Prev_ViewProjectionInverse");
    private static readonly int PastCamPosID = Shader.PropertyToID("_PastCamPos");
    private static readonly int DepthMap = Shader.PropertyToID("_DepthMap");
    private static readonly int MainTextureRight = Shader.PropertyToID("_MainTexRight");
    private static readonly int DepthMapRight = Shader.PropertyToID("_DepthMapRight");
    private Transform targetGStf;
    private Transform camTf;
    private PositionFoveatedTex positioner;

    private FixedSizeCache<float, ExtendedCamPose> poseCache;
    
    // For async sending
    private readonly SemaphoreSlim sendSemaphore = new SemaphoreSlim(1, 1);
    private ExtendedCamPose currentPose = new ExtendedCamPose();
    private float currentTime = 0;
    private float lastSendTime = 0;
    
    // Statistics
    public float statsRate = 0.5f;
    private int framesSent = 0;
    private int framesReceived = 0;
    private float lastFrameTime = 0;
    private float lastStatsTime = 0;
    
    void Start()
    {
        //Application.targetFrameRate = 90;
        lastTargetSppx = targetSppx;
        lastTargetBounces = targetBounces;
        
        GetRessources();
        ConnectToServer();
    }

    public void GetRessources()
    {
        if(cam == null) cam = Camera.main;
        
        targetMat = quad.GetComponent<Renderer>().sharedMaterial;
        positioner = quad.GetComponent<PositionFoveatedTex>();
        camTf = cam.transform;
        targetGStf = targetGS.transform;
        
        // Look back cache for original cam position of foveated img. If were 500 poses behind were probably done anyway
        poseCache = new FixedSizeCache<float, ExtendedCamPose>(500); 
    }
    
    void Update()
    {
        UpdatePoseData(camTf, targetGStf);

        // This is all a little hacky but works to demonstrate configuring the quality from the client
        if (targetSppx != lastTargetSppx || targetBounces != lastTargetBounces)
        {
            SendQualityAdjustmentAsync(targetSppx != lastTargetSppx, targetBounces != lastTargetBounces);
            lastTargetSppx = targetSppx;
            lastTargetBounces = targetBounces;
        }

        // Also hacky, sends tf as you type, but will be ignored as it won't match a path. Still,
        // probably best to just copy paths.
        if (!string.IsNullOrWhiteSpace(transferFunction) && transferFunction != lastTransferFunction)
        {
            SendTransferFunctionAsync();
            lastTransferFunction =  transferFunction;
        }

        if (Time.time - lastSendTime > maxSendRate)
        {
            SendPoseAsync();
            //SendPingPongAsync();
            lastSendTime = Time.time;
        }
        
        // Show statistics
        if (Time.time - lastStatsTime > statsRate)
        {
            Debug.Log($"Sent: {framesSent / statsRate}/s, Received: {framesReceived / statsRate}/s, RTT: {lastFrameTime})");
            framesSent = 0;
            framesReceived = 0;
            lastStatsTime = Time.time;
        }
    }

    private void LateUpdate()
    {
        // Process all queued actions on main thread at late update to increase chances data is ready
        while (mainThreadActions.TryDequeue(out Action action))
        {
            action?.Invoke();
        }
    }

    void UpdatePoseData(Transform camT, Transform gsT)
    {
        var pos = gsT.InverseTransformPoint(camT.position);
        var dir = gsT.InverseTransformDirection(camT.forward);
        var up = gsT.InverseTransformDirection(camT.up);
        var newPose = new CamPose
        {
            position = pos,
            dir = dir,
            up = up,
        };
        var extended = new ExtendedCamPose
        {
            camPose = newPose,
            worldToCamMat = cam.worldToCameraMatrix
        };
        currentPose = extended;
        currentTime = Time.time;
    }

    void ConnectToServer()
    {
        try
        {
            shouldStop = false;
            cancellationTokenSource = new CancellationTokenSource();
            
            client = new TcpClient(serverIP, serverPort);
            client.NoDelay = true; // Disable Nagle
            stream = client.GetStream();
            isConnected = true;
            
            // Start receiving thread
            receiveTask = Task.Run(() => ReceiveData(cancellationTokenSource.Token));
        }
        catch (Exception e)
        {
            Debug.LogError($"Connection failed: {e.Message}");
        }
    }

    void SendPoseAsync()
    {
        if (!isConnected || shouldStop) return;
        Task.Run(async () => await SendInternal(() =>
        {
            // Adjust pose for renderer
            var extCamPose = currentPose;
            var camPose = extCamPose.camPose;
            poseCache[currentTime] = extCamPose;
            camPose.up = new Vector3(-camPose.up.x, -camPose.up.y, -camPose.up.z);
            
            var req = new RenderRequest()
            {
                type = "frame",
                cam = camPose,
                time = currentTime,
            };
            return req;
        }));
    }

    void SendQualityAdjustmentAsync(bool sendSppx, bool sendBounces)
    {
        if (!isConnected || shouldStop) return;
        Task.Run(async () => await SendInternal(() =>
        {
            var req = new Dictionary<string, object>
            {
                ["type"] = "quality",
            };
            if (sendSppx) req["samples"] = targetSppx;
            if (sendBounces) req["bounces"] = targetBounces;
            return req;
        }));
    }

    void SendTransferFunctionAsync()
    {
        if (!isConnected || shouldStop) return;
        Task.Run(async () => await SendInternal(() =>
        {
            var req = new Dictionary<string, object>
            {
                ["type"] = "transferfunction",
                ["tf_path"] = transferFunction,
            };
            return req;
        }));
    }
    
    void SendPingPongAsync()
    {
        if (!isConnected || shouldStop) return;
        Task.Run(async () => await SendInternal(() =>
        {
            var t = new Dictionary<string, object>
            {
                ["type"] = "ping",
                ["time"] = currentTime
            };            
            return t;
        }));
    }
    
    async Task SendInternal<T>(Func<T> genPayload)
    {
        if (!isConnected || shouldStop) return;
        
        // Non-blocking semaphore check - drop frame if we're still sending previous one
        if (!await sendSemaphore.WaitAsync(0))
        {
            Debug.Log($"Dropping frame {Time.frameCount} - still sending previous pose frame");
            return;
        }

        try
        {
            var req = genPayload();
            
            
            byte[] data = MessagePackSerializer.Serialize((object)req);
            int dataLength = data.Length;
            int totalMessageLength = 4 + dataLength;
            
            byte[] messageBuffer = ArrayPool<byte>.Shared.Rent(totalMessageLength);
    
            try
            {
                // Write length header directly into the first 4 bytes
                if (BitConverter.IsLittleEndian)
                {
                    messageBuffer[0] = (byte)(dataLength >> 24);
                    messageBuffer[1] = (byte)(dataLength >> 16);
                    messageBuffer[2] = (byte)(dataLength >> 8);
                    messageBuffer[3] = (byte)(dataLength);
                }
                else
                {
                    messageBuffer[0] = (byte)(dataLength);
                    messageBuffer[1] = (byte)(dataLength >> 8);
                    messageBuffer[2] = (byte)(dataLength >> 16);
                    messageBuffer[3] = (byte)(dataLength >> 24);
                }
        
                // Copy serialized data after header
                Buffer.BlockCopy(data, 0, messageBuffer, 4, dataLength);
        
                // This works with async - messageBuffer is a regular array reference
                await stream.WriteAsync(messageBuffer, 0, totalMessageLength);
        
                framesSent++;
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(messageBuffer);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Send failed: {e.Message}");
            isConnected = false;
        }
        finally
        {
            sendSemaphore.Release();
        }
    }
    
    async void ReceiveData(CancellationToken cancellationToken)
    {
        while (isConnected && !cancellationToken.IsCancellationRequested && !shouldStop)
        {
            try
            {
                byte[] lengthBuffer = new byte[4];
                int lengthBytesRead = await stream.ReadAsync(lengthBuffer, 0, 4, cancellationToken);
                
                if (lengthBytesRead == 0 || cancellationToken.IsCancellationRequested || shouldStop) 
                    break;
                
                if (BitConverter.IsLittleEndian)
                    Array.Reverse(lengthBuffer);
                
                int messageLength = BitConverter.ToInt32(lengthBuffer, 0);
                
                byte[] messageBuffer = new byte[messageLength];
                int totalBytesRead = 0;
                while (totalBytesRead < messageLength && !cancellationToken.IsCancellationRequested && !shouldStop)
                {
                    int bytesRead = await stream.ReadAsync(messageBuffer, totalBytesRead, 
                        messageLength - totalBytesRead, cancellationToken);
                    if (bytesRead == 0) break;
                    totalBytesRead += bytesRead;
                }
                
                if (totalBytesRead == messageLength && !cancellationToken.IsCancellationRequested && !shouldStop)
                {
                    ProcessMessage(messageBuffer);
                }
            }
            catch (OperationCanceledException)
            {
                Debug.Log("Receive operation was cancelled");
                break;
            }
            catch (Exception e)
            {
                if (!cancellationToken.IsCancellationRequested && !shouldStop)
                {
                    Debug.LogError($"Receive failed: {e.Message}");
                }
                break;
            }
        }
    }
    
    string PeekType(ReadOnlyMemory<byte> data)
    {
        var reader = new MessagePackReader(data);
        var mapCount = reader.ReadMapHeader();
        for (int i = 0; i < mapCount; i++)
        {
            var key = reader.ReadString();
            if (key == "type")
                return reader.ReadString();
            reader.Skip();
        }
        return null;
    }
    
    void ProcessMessage(byte[] data)
    {
        if (shouldStop) return;
        
        try
        {
            framesReceived++;
            var type = PeekType(data);
            switch (type)
            {
                case "pong":
                    var pong = MessagePackSerializer.Deserialize<Envelope<PongData>>(data).payload;
                    Debug.Log($"Pong took {currentTime - pong.time}");
                    break;
                case "frame":
                    var frame = MessagePackSerializer.Deserialize<Envelope<FrameData>>(data).payload;
                    HandleFrameMessage(frame);
                    break;
                case "quality_ack":
                    var qualityAck = MessagePackSerializer.Deserialize<Envelope<QualityAck>>(data).payload;
                    HandleQualityMessage(qualityAck);
                    break;
                case "tf_ack":
                    var tfAck = MessagePackSerializer.Deserialize<Envelope<TfAck>>(data).payload;
                    HandleTfMessage(tfAck);
                    break;
                default:
                    Debug.LogError($"Unhandled message type: {type}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to process message: {e.Message}");
        }
    }

    void HandleQualityMessage(QualityAck qualityAck)
    {
        string sspx = qualityAck.samples.HasValue ? $"new sspx: {qualityAck.samples.Value.ToString()} " : "";
        string bounces = qualityAck.bounces.HasValue ? $"new bounces: {qualityAck.bounces.Value.ToString()}" : "";
        Debug.Log($"New render quality: {sspx}{bounces}");
    }

    void HandleTfMessage(TfAck tfAck)
    {
        Debug.Log($"Loaded new transfer function, success: {tfAck.status}, ML: {tfAck.message.Length} message: {tfAck.message}");
    }

    void HandleFrameMessage(FrameData message)
    {
         mainThreadActions.Enqueue(() => {
            lastFrameTime = Time.time - message.time;
            
            // Probably cache these instead of creating nex textures all the time?
            var left = new Texture2D(texSize, texSize, TextureFormat.RGBA32, false);
            var right = new Texture2D(texSize, texSize, TextureFormat.RGBA32, false);
            
            // Left is main eye, always there
            left.LoadRawTextureData(message.left);
            left.Apply();

            if (message.l_depth != null)
            {
                Span<byte> bytes = message.l_depth;
                Span<float> floats = MemoryMarshal.Cast<byte, float>(bytes);
                var depth = new Texture2D(texSize, texSize, TextureFormat.RFloat, false);
                
                BurstDepthProcessor.Cleanup(floats, texSize);
                
                var bytesAgain = MemoryMarshal.Cast<float, byte>(floats);
                depth.LoadRawTextureData(bytesAgain.ToArray());
                depth.Apply();
                
                targetMat.SetTexture(DepthMap, depth);
            }
            
            if (targetMat.mainTexture != null)
                Destroy(targetMat.mainTexture);
            targetMat.mainTexture = left;
            
            // For VR frame, also load right
            if (message.right != null)
            {
                right.LoadRawTextureData(message.right);
                right.Apply();
                
                targetMat.SetTexture(MainTextureRight, right);
                
                Span<byte> bytes = message.r_depth;
                Span<float> floats = MemoryMarshal.Cast<byte, float>(bytes);
                var depth = new Texture2D(texSize, texSize, TextureFormat.RFloat, false);

                BurstDepthProcessor.Cleanup(floats, texSize);
                
                var bytesAgain = MemoryMarshal.Cast<float, byte>(floats);
                depth.LoadRawTextureData(bytesAgain.ToArray());
                depth.Apply();
                
                targetMat.SetTexture(DepthMapRight, depth);
            }
            

            ExtendedCamPose origPose = default;
            try
            { 
                 origPose = poseCache[message.time];

                 // Undo inverse transform
                 // Not correct for continuously moving object but you get the point
                 origPose.camPose.position = targetGStf.TransformPoint(origPose.camPose.position);
                 origPose.camPose.dir = targetGStf.TransformDirection(origPose.camPose.dir);
                 origPose.camPose.up = targetGStf.TransformDirection(origPose.camPose.up);
             }
            catch (Exception e)
            {
                Debug.LogError($"Cache: {poseCache.raw.Count} {string.Join(" : ", poseCache.raw)} {string.Join(", ", poseCache.raw.Values.Select(v => v.camPose.position))}");
                Debug.LogError($"Failed to process message: {e.Message}");
            }
            
            OnImageReceived?.Invoke(origPose);
            //positioner.UpdatePosition(origPose);
            
            float aspectRatio = (float) left.width / left.height;
            Matrix4x4 pastProjectionMatrix = Matrix4x4.Perspective(positioner.targetFoV, aspectRatio, cam.nearClipPlane, cam.farClipPlane);
            Matrix4x4 pastVpInverseMatrix = (pastProjectionMatrix * origPose.worldToCamMat).inverse;
            
            targetMat.SetMatrix(PrevViewProjectionInverseID, pastVpInverseMatrix);
            targetMat.SetVector(PastCamPosID, origPose.camPose.position);
        });
    }
    
    
    void OnDestroy()
    {
        Disconnect();
    }
    
    private void Disconnect()
    {
        shouldStop = true;
        isConnected = false;
        
        cancellationTokenSource?.Cancel();
        
        try
        {
            stream?.Close();
            client?.Close();
        }
        catch (Exception e)
        {
            Debug.Log($"Exception during disconnect: {e.Message}");
        }
        
        if (receiveTask != null && !receiveTask.IsCompleted)
        {
            try
            {
                receiveTask.Wait(1000);
            }
            catch (AggregateException ex)
            {
                foreach (var innerEx in ex.InnerExceptions)
                {
                    if (!(innerEx is OperationCanceledException))
                    {
                        Debug.LogError($"Task exception: {innerEx.Message}");
                    }
                }
            }
        }
        
        cancellationTokenSource?.Dispose();
        sendSemaphore?.Dispose();
    }
}

public struct ExtendedCamPose
{
    public CamPose camPose;
    public Matrix4x4 worldToCamMat;
}
