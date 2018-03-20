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

package sparselu.files;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;


public class Block {

	public static final int BLOCK_SIZE	= 4;
	
	private int bCols, bRows;
	private double [][] data;

	public Block()
	{
		bRows = bCols = BLOCK_SIZE;
		data = new double[bRows][bCols];

		for (int i = 0; i < bRows; i++)
		{
			for(int j = 0; j < bCols; j++)
			{
				data[i][j] = 0.0;
			}
		}
	}

	public Block( int _bRows, int _bCols )
	{
		bRows = _bRows;
		bCols = _bCols;
		data = new double[bRows][bCols];

		for (int i = 0; i < bRows; i++)
		{
			for(int j = 0; j < bCols; j++)
			{
				data[i][j] = 0.0;
			}
		}
	}

	public Block( String filename )
	{
		bRows = bCols = BLOCK_SIZE;
		data = new double[bRows][bCols];
		try
		{
			FileReader filereader = new FileReader( filename );
			BufferedReader br = new BufferedReader( filereader );			// Get a buffered reader. More Efficient
			StringTokenizer tokens;
			String nextLine;

			for(int i = 0; i < bRows; i++)
			{
				nextLine = br.readLine();
				tokens = new StringTokenizer( nextLine );
				for(int j = 0; j < bCols && tokens.hasMoreTokens(); j++)
				{
					data[i][j] = Double.parseDouble( tokens.nextToken() );
				}
			}
			br.close();
			filereader.close();
		}
		catch ( FileNotFoundException fnfe )
		{
			fnfe.printStackTrace();
		}
		catch ( IOException ioe )
		{
			ioe.printStackTrace();
		}
	}

	protected void printBlock()
	{
		for(int i = 0; i < bRows; i++)
		{
			for(int j = 0; j < bCols; j++)
			{
				System.out.print( data[i][j] + " " );
			}
			System.out.println();
		}
	}

	public void blockToDisk( String filename )
		throws SparseLUAppException 
	{
		try
		{
			FileOutputStream fos = new FileOutputStream( filename );
			
			for ( int i = 0; i < bRows; i++ )
			{
				for ( int j = 0; j < bCols; j++ )
				{
					String str = ( new Double( data[i][j] ) ).toString() + " ";
					fos.write( str.getBytes() );
				}
				fos.write( "\n".getBytes() );
			}
			fos.close();
		}
		catch ( FileNotFoundException fnfe )
		{
			throw new SparseLUAppException( fnfe.getMessage() );
		}
		catch ( IOException ioe )
		{
			throw new SparseLUAppException( ioe.getMessage() );
		}
	}

	public void lu0() {
		for (int k = 0; k < BLOCK_SIZE; k++)
	    	for (int i = k+1; i < BLOCK_SIZE; i++) {
	    		this.data[i][k] = this.data[i][k] / this.data[k][k];
	            for (int j = k+1; j < BLOCK_SIZE; j++)
	            	this.data[i][j] -= this.data[i][k] * this.data[k][j];
	    	}
	}
		
	public void bdiv(Block diag) {
		for (int i = 0; i < BLOCK_SIZE; i++)
			for (int k = 0; k < BLOCK_SIZE; k++) {
				this.data[i][k] = this.data[i][k] / diag.data[k][k];
	            for (int j = k+1; j < BLOCK_SIZE; j++)
	            	this.data[i][j] -= this.data[i][k] * diag.data[k][j];
			}
	}
	
	public void bmod(Block row, Block col) {
		for (int i = 0; i < BLOCK_SIZE; i++)
	    	for (int j = 0; j < BLOCK_SIZE; j++)
	    		for (int k = 0; k < BLOCK_SIZE; k++)
	    			this.data[i][j] -= row.data[i][k] * col.data[k][j];
	}
	
	public void fwd(Block diag) {
		for (int j = 0; j < BLOCK_SIZE; j++)
			for (int k = 0; k < BLOCK_SIZE; k++) 
				for (int i = k+1; i < BLOCK_SIZE; i++)
					this.data[i][j] -= diag.data[i][k] * this.data[k][j];
	}
	
}
