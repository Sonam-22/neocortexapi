using System;
using Numenta.HTM.Encoders;

class EncoderExample
{
    static void Main(string[] args)
    {
        // Create a new encoder
        var encoder = new ScalarEncoder(w=21, minval=0, maxval=100, n=50);

        // Encode a value
        var value = 42;
        var encoded = encoder.encode(value);

        // Print the encoded value
        Console.WriteLine("Encoded value: " + string.Join(",", encoded));

        // Decode the encoded value
        var decoded = encoder.decode(encoded);

        // Print the decoded value
        Console.WriteLine("Decoded value: " + decoded);
    }
}
