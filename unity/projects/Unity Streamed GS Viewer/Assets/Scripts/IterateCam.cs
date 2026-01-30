using System;
using System.Collections;
using System.Collections.Generic;
using GaussianSplatting.Runtime;
using UnityEngine;

public class IterateCam : MonoBehaviour
{
    public float every = 0.5f;
    public int maxViews = 32;
    public GaussianSplatRenderer gs;
    
    private float last = 0;
    private int curActive = 0;

    private void Start()
    {
        Screen.SetResolution(4000, 2000, false);
    }

    // Update is called once per frame
    void Update()
    {
        if (Time.time > (last + every))
        {
            curActive = (curActive + 1) % maxViews;
            gs.ActivateCamera(curActive);
            last = Time.time;
        }
    }
}
