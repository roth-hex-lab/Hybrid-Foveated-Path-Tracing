using UnityEngine;

public class OrbitCamera : MonoBehaviour
{
    [Header("Target Settings")]
    [SerializeField] private Transform rotationCenter;
    [SerializeField] private Transform lookAtTarget;
    
    [Header("Orbit Settings")]
    [SerializeField] private Vector3 rotationAxis = Vector3.up;
    [SerializeField] private float orbitSpeed = 30f;
    [SerializeField] private float orbitRadius = 5f;
    
    [Header("Controls")]
    [SerializeField] private bool autoRotate = true;
    [SerializeField] private KeyCode rotateKey = KeyCode.Space;
    
    private float currentAngle = 0f;
    
    void Start()
    {
        if (rotationCenter == null)
            rotationCenter = transform;
            
        Vector3 directionFromCenter = transform.position - rotationCenter.position;
        currentAngle = Mathf.Atan2(directionFromCenter.x, directionFromCenter.z) * Mathf.Rad2Deg;
        orbitRadius = Vector3.Distance(transform.position, rotationCenter.position);
    }
    
    void Update()
    {
        /*
         * This is buggy for some axis i believe.
         * Fine though, only used for testing and visualization
         */


        if (rotationCenter == null) return;
                bool shouldRotate = autoRotate || Input.GetKey(rotateKey);
        
        if (shouldRotate)
        {
            currentAngle += orbitSpeed * Time.deltaTime;
            if (currentAngle >= 360f)
                currentAngle -= 360f;
        }
        
        Vector3 newPosition = CalculateOrbitPosition();
        transform.position = newPosition;
        if (lookAtTarget != null)
        {
            transform.LookAt(lookAtTarget);
        }
    }
    
    private Vector3 CalculateOrbitPosition()
    {
        Quaternion rotation = Quaternion.AngleAxis(currentAngle, rotationAxis.normalized);
        Vector3 offset = Vector3.forward * orbitRadius;
        Vector3 rotatedOffset = rotation * offset;
        return rotationCenter.position + rotatedOffset;
    }
    
    public void ToggleAutoRotate()
    {
        autoRotate = !autoRotate;
    }
    
    void OnDrawGizmosSelected()
    {
        if (rotationCenter == null) return;
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(rotationCenter.position, 0.2f);
        
        Gizmos.color = Color.cyan;
        Vector3 prevPoint = rotationCenter.position + Vector3.forward * orbitRadius;
        
        for (int i = 1; i <= 36; i++)
        {
            float angle = (i * 10f);
            Quaternion rot = Quaternion.AngleAxis(angle, rotationAxis.normalized);
            Vector3 newPoint = rotationCenter.position + rot * (Vector3.forward * orbitRadius);
            
            Gizmos.DrawLine(prevPoint, newPoint);
            prevPoint = newPoint;
        }
        
        Gizmos.color = Color.red;
        Vector3 axisStart = rotationCenter.position - rotationAxis.normalized * orbitRadius;
        Vector3 axisEnd = rotationCenter.position + rotationAxis.normalized * orbitRadius;
        Gizmos.DrawLine(axisStart, axisEnd);
        
        if (lookAtTarget != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawLine(transform.position, lookAtTarget.position);
        }
    }
}