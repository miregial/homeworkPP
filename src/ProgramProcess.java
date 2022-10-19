import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Scanner;

public class ProgramProcess {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Введите путь к файлу wav");
            return;
        }

        if (Objects.equals(args[0], "asChild"))
        {
            childProcess(args);
            return;
        }

        // Мы родительский процесс
        var filePath = args[0];
        var THAN = 16000L;
        var PROCESS_COUNT = 4;

        if (args.length > 1) {
            THAN = Long.parseLong(args[1]);
        }
        if (args.length > 2) {
            PROCESS_COUNT = Integer.parseInt(args[2]);
        }

        mainProcess(filePath, THAN, PROCESS_COUNT);
    }

    private static void mainProcess(String filePath, long THAN, int PROCESS_COUNT) throws Exception {
        var GT = 0L;
        var LEQ = 0L;

        var wavReader = new WavReader(filePath);
        var decodedData = wavReader.Decode();
        var chunkedData = ArrayChunker.GetChunks(decodedData, PROCESS_COUNT);
        var processes = new ArrayList<Process>();
        var resultFileProcess = new ArrayList<File>();

        for (var i = 0; i < chunkedData.size(); i++) {
            // Записываем данные в файл
            var fileName = "" + i;
            SaveInFile(chunkedData.get(i), fileName);

            // Создаем процесс
            var pb = new ProcessBuilder("java", "ProgramProcess", "asChild", "" + THAN, fileName);

            // Перенаправляем stderr в родительский
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);

            // Перенаправляем stdout в файл
            var resultFile = new File("result_" + fileName);
            resultFileProcess.add(resultFile);
            pb.redirectOutput(resultFile);

            var process = pb.start();
            processes.add(process);
        }

        processes.forEach(process -> {
            try {
                process.waitFor();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        // Получаем результат
        for (var resultFile : resultFileProcess) {
            var fis = new FileInputStream(resultFile);
            var data = new byte[(int) resultFile.length()];
            fis.read(data);
            fis.close();

            var str = new String(data, StandardCharsets.UTF_8);
            var gt_lew_str = str.split(" ");
            GT += Long.parseLong(gt_lew_str[0].strip());
            LEQ += Long.parseLong(gt_lew_str[1].strip());
        }

        System.out.println("Process count: " + PROCESS_COUNT);
        System.out.println("Data length: " + decodedData.length);
        System.out.println("Count <= than " + THAN + " " + LEQ);
        System.out.println("Count > than " + THAN + " " + GT);
        System.out.println("Sum count "  + (LEQ + GT));
    }

    private static void childProcess(String[] args) throws IOException {
        // мы child процесс
        var THAN = Long.parseLong(args[1]);
        var FILEPATH = args[2];
        var LEQ = 0;
        var GT = 0;

        var scanner = new Scanner(new File(FILEPATH));
        scanner.useDelimiter(" ");

        while(scanner.hasNextLong())
        {
            var data = scanner.nextLong();
            if (Math.abs(data) > THAN) {
                GT++;
            } else {
                LEQ++;
            }
        }

        System.out.println(GT + " " + LEQ);
    }

    private static void SaveInFile(ArrayList<Long> array, String filePath)
    {
        var sb = new StringBuilder();
        for (var value: array) {
            sb.append(" ").append(value);
        }
        try(var fw = new FileWriter(filePath, false))
        {
            var fileText = sb.toString();
            fw.write(fileText);
            fw.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
