#!/usr/bin/env python3
#
# FDSN-WebService Listener
# This module is part of the Automatic Alert System (AAS) solution
#
# Author:  Juan Esteban Rodríguez, Josep de la Puente
# Contact: juan.rodriguez@bsc.es, josep.delapuente@bsc.es
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ###############################################################################

import os, json
import sys
import traceback
import argparse
from ucis4eq.aas import FDSNClient

# ###############################################################################

# Methods and classes


def parser():
    # Parse the arguments
    parser = argparse.ArgumentParser(
        prog='listener',
        description='FDSN-WS Listener')
    parser.add_argument('config', help='Remote JSON configuration file')
    parser.add_argument('-p', dest='opid',
                        help='Wait for a given PID before start')
    args = parser.parse_args()
    # Return them
    return args


def main():
    try:
        # Call the parser
        args = parser()
        if (args.config == "config_events"):
            ws = FDSNClient.WSEvents().start(args.config)
        else:
            ws = FDSNClient.WSGeneral().start(args.config)
    except Exception as error:
        print("Exception in code:")
        print('-' * 80)
        traceback.print_exc(file=sys.stdout)
        print('-' * 80)


# ###############################################################################
# Run the program
if __name__ == "__main__":
    main()
