import java.io.File;
import java.io.FileInputStream;

public class WavReader {
    private String _filePath;
    private WavHeader _wavHeader;

    public WavReader(String filePath) {
        _filePath = filePath;
    }

    public long[] Decode() throws Exception {
        var fileBytes = GetBytes(_filePath);
        _wavHeader = new WavHeader(fileBytes);
        return _wavHeader.Data;
    }

    private byte[] GetBytes(String filePath)
    {
        try(var fileStream = new FileInputStream(filePath)){
            var fileSize = new File(filePath).length();

            var bytes = new byte[(int) fileSize];

            var bytesRead = fileStream.read(bytes);
            return bytes;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}