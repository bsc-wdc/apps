/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package kmeans_frag;

public class Utils {
    
    /**
     * Evaluates the convergence of the fragment mu compared with the oldmu.
     * @param mu New centers
     * @param oldmu Old centers
     * @param epsilon Convergence distance value
     * @param n Iteration number
     * @param maxIterations Maximum number of iterations
     * @return The convergence condition.
     */
    public static boolean has_converged (Fragment mu, Fragment oldmu, double epsilon, int n, int maxIterations){
        System.out.println("Iter: " + n);
        System.out.println("maxIterations: " + maxIterations);
        if (oldmu == null) {
            return false;
        } else if (n >= maxIterations) {
            return true;
        } else {
            double aux = 0;  
            for (int k = 0; k < mu.getVectors(); k++) {                 // loop over each center
                double dist = 0;
                for (int dim = 0; dim < mu.getDimensions(); dim++) { 	// loop over every dimension of a center
                        double tmp = oldmu.getPoint(k, dim) - mu.getPoint(k, dim) ;
                        if (tmp != 0 && !Double.isNaN(tmp)){
                            dist += tmp*tmp;
                        }else{
                            dist = 0;
                        }
                }
                aux += dist;
            }
            if (aux < epsilon*epsilon) {
                System.out.println("Distance_T: " + aux);
                return true;
            } else {
                System.out.println("Distance_F: " + aux);
                return false;
            }
        }
    }
}
