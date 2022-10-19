import java.util.ArrayList;

public class ArrayChunker {
    public static ArrayList<ArrayList<Long>> GetChunks(long[] array, int chunkCount)
    {
        var chunks = new ArrayList<ArrayList<Long>>();
        for (int i = 0; i < chunkCount; i++) {
            chunks.add(new ArrayList<>());
        }

        for (int i = 0; i < array.length; i++) {
            var chunkIndex = i % chunkCount; // в какой поток положить decodedData[i]
            chunks.get(chunkIndex).add(array[i]);
        }

        return chunks;
    }
}
