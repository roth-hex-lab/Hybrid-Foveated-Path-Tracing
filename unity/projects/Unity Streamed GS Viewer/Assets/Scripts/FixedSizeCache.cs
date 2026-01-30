using System.Collections.Generic;

public class FixedSizeCache<T_Key, T_Val>
{
    private readonly int capacity;
    private readonly Dictionary<T_Key, T_Val> map;
    private readonly T_Key[] keyBuffer;
    private int head = 0;
    private int count = 0;

    public FixedSizeCache(int capacity)
    {
        this.capacity = capacity;
        map = new Dictionary<T_Key, T_Val>(capacity);
        keyBuffer = new T_Key[capacity];
    }
    
    public T_Val this[T_Key key]
    {
        get => map[key];
        set => Set(key, value);
    }

    public void Set(T_Key key, T_Val value)
    {
        if (map.ContainsKey(key))
        {
            map[key] = value;
            return;
        }

        // If full, evict the oldest key to stay in target size
        if (count == capacity)
        {
            var oldKey = keyBuffer[head];
            map.Remove(oldKey);
        }
        else
        {
            count++;
        }

        keyBuffer[head] = key;
        map[key] = value;

        head = (head + 1) % capacity;
    }

    public bool TryGetValue(T_Key key, out T_Val value) => map.TryGetValue(key, out value);
    
    public bool ContainsKey(T_Key key) => map.ContainsKey(key);

    public Dictionary<T_Key, T_Val> raw => map;
}