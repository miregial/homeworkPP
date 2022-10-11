import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class WavHeader {
    public String ChunkId; // "RIFF"
    public long ChunkSize;
    public String Format; // "WAVE"
    public String Subchunk1Id; // "fmt "
    public long Subchunk1Size;
    public int AudioFormat;
    public int NumChannels;
    public long SampleRate;
    public long ByteRate;
    public int BlockAlign;
    public int BitsPerSample;
    public String Subchunk2Id;
    public int Subchunk2Size;
    public long[] Data;

    private int getUnsignedShort(ByteBuffer bb)
    {
        return ((short) bb.getShort() & 0xFFFF);
    }

    private long getUnsignedInt(ByteBuffer bb)
    {
        return ((long) bb.getInt() & 0xFFFFFFFFL);
    }

    private long GetSample(ByteBuffer byteBuffer)
    {
        return switch (BitsPerSample) {
            case 8 -> byteBuffer.get();
            case 16 -> byteBuffer.getShort();
            case 32 -> byteBuffer.getInt();
            case 64 -> byteBuffer.getLong();
            default -> throw new IllegalStateException("Unexpected value: " + BitsPerSample);
        };
    }

    public WavHeader(byte[] fileBytes) throws Exception {
        var byteBuffer = ByteBuffer.wrap(fileBytes);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);

        byte[] buffer = new byte[4];

        byteBuffer.get(buffer, 0, 4);
        ChunkId = new String(buffer);

        ChunkSize = getUnsignedInt(byteBuffer);

        byteBuffer.get(buffer, 0, 4);
        Format = new String(buffer);

        byteBuffer.get(buffer, 0, 4);
        Subchunk1Id = new String(buffer);

        Subchunk1Size = getUnsignedInt(byteBuffer);
        AudioFormat = getUnsignedShort(byteBuffer);
        NumChannels = getUnsignedShort(byteBuffer);
        SampleRate = getUnsignedInt(byteBuffer);
        ByteRate = getUnsignedInt(byteBuffer);
        BlockAlign = getUnsignedShort(byteBuffer);
        BitsPerSample = getUnsignedShort(byteBuffer);

        var chunkId = "";
        while (byteBuffer.position() < byteBuffer.limit())
        {
            try {
                byteBuffer.get(buffer, 0, 4);
                chunkId = new String(buffer);

                if (chunkId.equals("data"))
                    break;

                var chunkSize = (int)getUnsignedInt(byteBuffer);
                var chunkBuffer = new byte[chunkSize];
                byteBuffer.get(chunkBuffer, 0, chunkSize);
            } catch (IndexOutOfBoundsException e)
            {
                throw new Exception("В вашем файле нет заголовка с данными", e);
            }
        }

        Subchunk2Id = "data";
        Subchunk2Size = byteBuffer.getInt();

        var sampleCount = Subchunk2Size / (BitsPerSample / 8);
        Data = new long[sampleCount];

        for (var i = 0; i < sampleCount; i++) {
            Data[i] = GetSample(byteBuffer);
        }
    }
}
