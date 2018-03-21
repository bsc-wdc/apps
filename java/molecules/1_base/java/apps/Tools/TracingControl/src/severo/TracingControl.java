package severo.tracingcontrol;

import client.ClientManagementLib;

public class TracingControl {
    public static void main (String[] args) {
        if (args.length != 1) {
            error();
        }
        if (args[0].equals("pause")) {
            ClientManagementLib.pauseServerTraces();
        } else if (args[0].equals("resume")) {
            ClientManagementLib.resumeServerTraces();
        } else if (args[0].equals("flush")) {
            ClientManagementLib.flushServerTraces();
        } else {
            error();
        }
    }

    private static void error() {
        System.err.println("Bad arguments, expected: <pause | resume | flush>");
        System.exit(-1);
    }
}
