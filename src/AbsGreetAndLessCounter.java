import java.util.ArrayList;
import java.util.concurrent.Callable;

public class AbsGreetAndLessCounter implements Callable<GreatAndLessOrEqualCount> {
    private ArrayList<Long> _data;
    private long _than;

    public AbsGreetAndLessCounter(ArrayList<Long> data, long than){
        _than = than;
        _data = data;
    }

    @Override
    public GreatAndLessOrEqualCount call() {
        int leq = 0; // <=
        int gt = 0; // >
        for(long value: _data)
        {
            if (Math.abs(value) > _than) {
                gt++;
            } else {
                leq++;
            }
        }
        var result = new GreatAndLessOrEqualCount();
        result.LessOrEqualCount = leq;
        result.GreatCount = gt;
        return result;
    }
}
