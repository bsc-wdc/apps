#!/usr/bin/env python3
#
# Encode and parse the query string params
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


################################################################################
# Module imports
from urllib.parse import urlparse

################################################################################
# Methods and classes
def parse_query_params(query_string):
    """
    Function to parse the query parameter string.
    """
    # Parse the query param string
    query_params = dict(urlparse.parse_qs(query_string))

    # Get the value from the list
    query_params = {k: v[0] for k, v in query_params.items()}

    return query_params
