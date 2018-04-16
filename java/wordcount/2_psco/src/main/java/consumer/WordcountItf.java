package consumer;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;
import model.Text;
import model.TextStats;


public interface WordcountItf {
    @Method(declaringClass = "consumer.WordcountImpl")
    public TextStats wordCountNewStats(
           @Parameter (type = Type.OBJECT, direction = Direction.IN)
           Text text
    );
    
    @Method(declaringClass = "consumer.WordcountImpl")
    public TextStats wordCountFillStatsIN(
           @Parameter (type = Type.OBJECT, direction = Direction.IN)
           Text text,
           @Parameter (type = Type.OBJECT, direction = Direction.IN)
           TextStats result
    );
    
    @Method(declaringClass = "consumer.WordcountImpl")
    public void wordCountFillStatsINOUT(
           @Parameter (type = Type.OBJECT, direction = Direction.IN)
           Text text,
           @Parameter (type = Type.OBJECT, direction = Direction.INOUT)
           TextStats result
    );
    
    @Method(declaringClass = "consumer.WordcountImpl")
    public TextStats reduceTaskIN(
            @Parameter (type = Type.OBJECT, direction = Direction.IN)
            TextStats m1,
            @Parameter (type = Type.OBJECT, direction = Direction.IN)
            TextStats m2
    );
    
    @Method(declaringClass = "consumer.WordcountImpl")
    public void reduceTaskINOUT(
            @Parameter (type = Type.OBJECT, direction = Direction.INOUT)
            TextStats m1,
            @Parameter (type = Type.OBJECT, direction = Direction.IN)
            TextStats m2
    );
    
    @Method(declaringClass = "model.TextStats")
	public void wordCountFillingStats(
		@Parameter (type = Type.OBJECT, direction = Direction.IN)
		Text text
	);
	
    @Method(declaringClass = "model.TextStats")
	public void reduceTask(
		@Parameter (type = Type.OBJECT, direction = Direction.IN)
		TextStats m2
	);
    
}
