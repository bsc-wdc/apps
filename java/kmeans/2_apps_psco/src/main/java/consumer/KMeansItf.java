package consumer;

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;

import model.Fragment;
import model.SumPoints;


public interface KMeansItf {

    @Method(declaringClass = "consumer.KMeansImpl")
    public SumPoints clusters_points_and_partial_sum(
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Fragment fragment,
        @Parameter(type = Type.OBJECT, direction = Direction.IN) Fragment mu,
        @Parameter(type = Type.INT, direction = Direction.IN) int k, 
        @Parameter(type = Type.INT, direction = Direction.IN) int ind
    );

    // With return
    @Method(declaringClass = "consumer.KMeansImpl") // , priority = true)
    public SumPoints reduceCentersTask(
        @Parameter(type = Type.OBJECT, direction = Direction.IN) SumPoints a,
        @Parameter(type = Type.OBJECT, direction = Direction.IN) SumPoints b
    );

    // @Method(declaringClass = "model.Fragment")
    // public SumPoints cluster_points_and_partial_sum(
    // @Parameter(type = Type.OBJECT, direction = Direction.IN) Fragment mu,
    // @Parameter(type = Type.INT, direction = Direction.IN) int k,
    // @Parameter(type = Type.INT, direction = Direction.IN) int ind);

    // @Method(declaringClass = "model.SumPoints", priority = true)
    // SumPoints reduceCentersTask(
    // @Parameter(type = Type.OBJECT, direction = Direction.IN)
    // SumPoints other
    // );

    // @Method(declaringClass = "model.Fragment")
    // Clusters clusters_points_partial(@Parameter(type = Type.OBJECT, direction
    // = Direction.IN) Fragment mu,
    // @Parameter(type = Type.INT, direction = Direction.IN) int k,
    // @Parameter(type = Type.INT, direction = Direction.IN) int ind);

    // @Method(declaringClass = "model.Fragment")
    // SumPoints partial_sum(@Parameter(type = Type.OBJECT, direction =
    // Direction.IN) Clusters cluster,
    // @Parameter(type = Type.INT, direction = Direction.IN) int k,
    // @Parameter(type = Type.INT, direction = Direction.IN) int ind);

}
