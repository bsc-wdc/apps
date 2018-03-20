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
package npb.nasft;

import java.io.Serializable;
import java.text.DecimalFormat;


/**
 * A representation of complex. Warning! Complex objects are mutable.
 *
 */
public class Complex implements Cloneable, Serializable {

    /**
     * 
     */
    private static final long serialVersionUID = 40L;
    private static int quantity = 0;// DEBUG
    public double real;
    public double img;

    public Complex() {
        this(0., 0.);
    }

    public Complex(double r) {
        this(r, 0.);
    }

    public Complex(double[] val) {
        this(val[0], val[1]);
    }

    public Complex(double real, double img) {
        quantity++;
        this.real = real;
        this.img = img;
    }

    public void set(double r, double i) {
        this.real = r;
        this.img = i;
    }

    public double getImg() {
        return img;
    }

    public void setImg(double img) {
        this.img = img;
    }

    public double getReal() {
        return real;
    }

    public double[] get() {
        return new double[] { real, img };
    }

    public void set(double[] value) {
        this.real = value[0];
        this.img = value[1];
    }

    public void setReal(double real) {
        this.real = real;
    }

    public Complex div(double d) {
        return new Complex(this.real / d, this.img / d);
    }

    public void divMe(double d) {
        this.real = this.real / d;
        this.img = this.img / d;
    }

    public Complex plus(Complex complex) {
        return new Complex(this.real + complex.real, this.img + complex.img);
    }

    public void plusMe(Complex complex) {
        this.real += complex.real;
        this.img += complex.img;
    }

    public void plusMe(double real, double img) {
        this.real += real;
        this.img += img;
    }

    public Complex minus(Complex c) {
        return new Complex(this.real - c.real, this.img - c.img);
    }

    public Complex mult(double d) {
        return new Complex(this.real * d, this.img * d);
    }

    public Complex mult(Complex complex) {
        return new Complex((this.real * complex.real) - (this.img * complex.img), (this.real * complex.img) +
            (this.img * complex.real));
    }

    public void multMe(double d) {
        this.real *= d;
        this.img *= d;
    }

    public void multMe(Complex complex) {
        this.real = (this.real * complex.real) - (this.img * complex.img);
        this.img = (this.real * complex.img) + (this.img * complex.real);
    }

    /**
     * A function that returns the complex conjugate of the argument.  If
     * the complex is (X,Y), its complex conjugate is (X,-Y).
     * @return the conjugate of this
     */
    public Complex conjg() {
        return new Complex(this.real, -this.img);
    }

    public String toString() {
        DecimalFormat norm = new DecimalFormat("0.0000000000");
        return "(" + norm.format(this.real) + "  \t " + norm.format(this.img) + ")";
    }

    public Object clone() {
        try {
            return super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void main(String[] args) {
        Complex c1;
        Complex c2;

        c1 = new Complex(1., 2);

        c2 = new Complex(0, 1.);

        System.out.println(" c1: " + c1 + " c2: " + c2);

        System.out.println(" c1.getReal(): " + c1.getReal());
        System.out.println(" c1.getImg(): " + c1.getImg());

        System.out.println(" c1.div(2): " + c1.div(2));
        System.out.println(" c1.mult(c2): " + c1.mult(c2));
        //        c1.divMe(2);
        //        System.out.println(" c1.divMe(2): " + c1);
        c1.plusMe(c2);
        System.out.println(" c1.plusMe(c2): " + c1);
    }

    public static int getQuantity() {
        return quantity;
    }
}
