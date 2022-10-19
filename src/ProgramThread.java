import java.util.ArrayList;
import java.util.concurrent.FutureTask;

public class ProgramThread {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Введите путь к файлу wav");
            return;
        }

        var filePath = args[0];
        long THAN = 16000;
        int THREAD_COUNT = 4;

        if (args.length > 1) {
            THAN = Long.parseLong(args[1]);
        }
        if (args.length > 2) {
            THREAD_COUNT = Integer.parseInt(args[2]);
        }

        var wavReader = new WavReader(filePath);
        var decodedData = wavReader.Decode();
        // [12, 43, -56, 123, 54, -9999, 12000, ...]
        // 0 -> [12, 54]
        // 1 -> [43, -9999]
        // 2 -> [-56, 12000]
        // 3 -> [123, ]

        var tasksParams = ArrayChunker.GetChunks(decodedData, THREAD_COUNT);

        var futureTasks = new ArrayList<FutureTask<GreatAndLessOrEqualCount>>();
        for (int i = 0; i < THREAD_COUNT; i++)
        {
            var taskParam = tasksParams.get(i); // получаем параметры для передачи их в поток
            var callable = new AbsGreetAndLessCounter(taskParam, THAN); // создаем объект потока(вызываемый) и передаем ему параметры (данные)
            var futureTask = new FutureTask(callable); // создаем объект futureTask, чтобы мы могли получить результат выполнения

            futureTasks.add(futureTask); // добавляем объект futureTask в список для получения результата в будущем
            new Thread(futureTask).start(); // запускаем выполнения futureTask в потоке
        }
        var great = 0;
        var lessOrEqual = 0;
        for (var f: futureTasks)
        {
            var result = f.get(); // получаем результат выполнения

            // суммируем результат от каждого потока
            great += result.GreatCount;
            lessOrEqual += result.LessOrEqualCount;
        }

        System.out.println("Thread count: " + THREAD_COUNT);
        System.out.println("Data length: " + decodedData.length);
        System.out.println("Count <= than " + THAN + " " + lessOrEqual);
        System.out.println("Count > than " + THAN + " " + great);
        System.out.println("Sum count "  + (great + lessOrEqual));
    }
}