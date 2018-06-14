package worker.hmmerobj.publish;

import javax.xml.ws.Endpoint;

import worker.hmmerobj.*;

public class HmmerObjectsImplServicePublisher {

	public static void main(String[] args) {

		Endpoint.publish("http://localhost:8080/HmmerObjectsImpl/hmmerobjectsimpl",
				new HmmerObjectsImpl());
		
		System.out.println("The web service is published at http://localhost:8080/HmmerObjectsImpl/hmmerobjectsimpl");
		
		System.out.println("To stop running the web service , terminate the java process");
		

	}

}
