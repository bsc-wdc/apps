#!/usr/bin/env python3
#
# FDSN-WebService Listener
# This module is part of the Automatic Alert System (AAS) solution
#
# Author:  Juan Esteban Rodríguez
# Contributor: Josep de la Puente <josep.delapuente@bsc.es>
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
# Module imports
import os
import uuid
import sys
import json
import requests
import sched, time
import multiprocessing
import obspy
import datetime
import psutil

from ucis4eq.dal import staticDataMap
from ucis4eq.misc import config

# ###############################################################################
# Methods and classes


# Interface class
class WSGeneral:

    # Methods
    def __init__(self):
        # Wait for the process to finish
        #if( 'opid' in config['listener'] ):
        #    while psutil.pid_exists(config['listener']['opid']):
        #        print(config['listener']['opid'], "already exists")
        #        time.sleep(10)
        #else:
        #    os.system('python %s' % ' '.join(sys.argv[0:]), "-p ", str(os.getpid()))

        # Attributes
        self.results = {}
        self.outdir = "./"
        self.tmpdata = ""
        self.query = ""

        # Object UUID
        #self.uuid = uuid.uuid1()

        # Initialize DAL
        r = requests.post("http://127.0.0.1:5000/dal", json={})
        config.checkPostRequest(r)

        # Index documents with regard this component
        self.fileMapping = staticDataMap.StaticDataMap(self.__class__.__name__)

    # Start client
    def start(self, file):

        # Read the configuration file
        with open(self.fileMapping[file], 'r') as f:
            self.config = json.load(f)

        print(self.config, flush=True)

        # Check if an output directory was provided
        if ("outputdir" not in self.config["listener"]):
            self.config["listener"]["outputdir"] = self.outdir
        self.outdir = self.config["listener"]["outputdir"] + "/"
        self.tmpdata = self.outdir + "temporary_data/"

        # Create data directories
        os.makedirs(self.tmpdata, exist_ok=True)

        # Create the recursive scheduler
        self.s = sched.scheduler(time.time, time.sleep)

        # Config such scheduler
        self.s.enter(0, 1, self._exploreRepositories, ())

        # Start the AAS service
        self.s.run()

    # Explore the set of input repositories
    def _exploreRepositories(self, pool=None):
        # Variables
        output = {}
        dump = {}

        # Prepare the pool of processes
        np = len(self.config['repositories'])
        ws = self.config['webservice']

        # Create the processes pool just one
        if not pool:
            pool = multiprocessing.Pool(np)

        query = ws['interface'] + "/" + ws['majorversion'] + "/"
        query = query + ws["application"] + "?"
        for p in ws["parameters"]:
            query = query + p + "=" + ws["parameters"][p] + "&"

        self.query = query.strip('&')

        # For each provided repository
        tasks = [(self._requestRepository,
                 (r, self.config['repositories'][r]+self.query))
                 for r in self.config['repositories'].keys()]
        imap_unordered_it = pool.imap_unordered(self._runtask, tasks)

        #print("All ready!", flush=True)

        # Wait the tasks to finish
        for name, nelems, result in imap_unordered_it:
            self.results[name] = result
            if (nelems):
                dump[name] = nelems

        # Store the current timestamp
        output = self._postprocess_actions()

        # If there are elements, write them and trigger the set action
        if (output):
            #time2 = datetime.datetime.now(datetime.timezone.utc).strftime('%d%m%Y_%H%M%S')+"_"
            #file = self.outdir + time2 + self.config['listener']['results_name']

            #print(json.dumps(output), flush=True)

            # Do a trigger for each event received
            event = {}
            for e in output['events']:

                event['alerts'] = output['events'][e]
                event['uuid'] = e
                event['sources'] = {}

                for a in event['alerts']:
                    agency = a['agency']
                    event['sources'][agency] = output['sources'][agency]
                #print(json.dumps(event), flush=True)

                file = self.outdir + e + "." + self.config['listener']['results_name']
                with open(file, "w") as f:
                    json.dump(event, f, indent=4)

                # Start running the triggering system
                os.system((self.config['listener']['trigger'] + "&").replace("%s", file))

        # Queue the following job
        interval = self.config['listener']['interval']
        if( interval > 0 ):
            #print("Next report in ", interval, "seconds")
            self.s.enter(interval, 1, self._exploreRepositories, kwargs={'pool': pool})


    # Do the REST-GET petition
    def _requestRepository(self, name, url):
        # Request data
        #print("Requesting info from '" + url + "'")
        self.r = requests.get(url, stream=True)
        nelems, result = self._process_data(self.r, name)
        # Process data
        return (name, nelems, result)


    # Process obtained data
    def _process_data(self, results, name):
        # Variables
        file = self.tmpdata+name+self.config['listener']['data_ext']
        # Write file to disk
        with open(file, 'wb') as f:
            f.write(results.content)
        # Return the complete filename
        return file


    # Post-process actions
    def _postprocess_actions(self):
        pass


    # For parallel computing
    def __execute(self,func, args):
        result = func(*args)
        return result

    def _runtask(self, args):
        return self.__execute(*args)


class WSEvents(WSGeneral):

    def __init__(self):
        # Attributes
        self.currenttime = None
        self.events = []
        #WSGeneral.__init__(self,config)
        super(WSEvents, self).__init__()
        #self.uuid = uuid.uuid1()

    def _process_data(self, results, name):
        # Variables
        params = self.config['webservice']['parameters']

        # Write the original file to disk
        file = super(WSEvents, self)._process_data(results, name)
        # Check the requested format
        if( 'format' in params.keys() and params['format'] == "xml"):
            return self._process_xml_data(results, file, name)
        return 0

    # Process a QuakeML data
    def _process_xml_data(self, results, file, name):
        # Variables
        now = None
        delay = None
        cat = None
        events = []

        # Set the time threshold
        if( name in self.results.keys() and self.results[name] ):
            currenttime = self.results[name]['timestamp']
        else:
            currenttime = 0.0

        try:
            # Read the set of events
            cat = obspy.read_events(file, "QUAKEML")
        except Exception as error:
            # MPC printing information for debugging
            # MPC the listener pings the FDSN services every 60s and updates the files in
            # /tmp/ucis4eq-aas-listener/temporary_data
            # If one of the agencies does not give a response, the file AGENCY_NAME.response
            # in the tmp directory is going to be empty, which is what this exception catches.
            # print("WARNING: reading the events from file %s failed." %file)
            return False, {}

        #print("[", name, "] --> ", cat.count(),"events found")

        for e in cat:
            if(currenttime < e.origins[0]['time'].timestamp):
                # Calculate the elapsed time from last event occurred
                now = datetime.datetime.now(datetime.timezone.utc)
                delay = now.timestamp() - e.origins[0]['time'].timestamp

                # DEBUG pursoses message
                #print("[", name, "] --> ",  e.origins[0]['time'].datetime,
                #e.origins[0]['latitude'], e.origins[0]['longitude'],
                #e.origins[0]['depth'], e.magnitudes[0]['mag'],
                #e.event_descriptions[0].text if e.event_descriptions else "",
                #"Delay (secs):", delay, flush=True)

                # Don't add the event if a deadline time was reached
                # SPRUCE [P.Beckman 2006]
                if(delay > self.config['listener']['deadline']):
                    print("WARNING: The event occurred at " +  str(e.origins[0]['time'].datetime) +
                    " in (" + str(e.origins[0]['latitude']) + ", " + str(e.origins[0]['longitude'])
                    + ") and magnitude " + str(e.magnitudes[0]['mag'])
                    + " overpassed the set deadline and will be not triggered", flush=True)
                    continue

                # Obtain information about the event
                event = {}
                #event['time'] = e.origins[0]['time'].datetime.strftime("%d/%m/%Y, %H:%M:%S")
                event['time'] = e.origins[0]['time'].timestamp
                event['latitude'] = e.origins[0]['latitude']
                event['longitude'] = e.origins[0]['longitude']
                event['depth'] =  e.origins[0]['depth']
                event['magnitude'] = e.magnitudes[0]['mag']
                event['elapsedtime'] = int(delay)
                event['description'] = e.event_descriptions[0].text if e.event_descriptions else ""
                events.append(event)

        #print(name, ":", self.a)
        #print( cat.__str__(print_all=True) )
        #print("[", name, "] --> ",  cat[0].origins[0]['time'],
        #    cat[0].origins[0]['latitude'], cat[0].origins[0]['longitude'],
        #    cat[0].origins[0]['depth'], cat[0].magnitudes[0]['mag'])
            #self.currenttime = cat[1]['creation_info']['creation_time']
        return len(events), {'timestamp': cat[0].origins[0]['time'].timestamp,
                             'events' : events}

    # Post-process actions
    def _postprocess_actions(self):
        # Variables
        tth = self.config['listener']['timethreshold']
        output = {'sources':{}, 'events':{}}
        source = {}
        events = {}
        id = ""

        # Generate the set of events
        for r in self.results.keys():
            # Check if the current agency didn't registered events
            if( not self.results[r] ):
                continue

            # Define sources
            source = {}
            source['query'] = self.config['repositories'][r]+self.query
            source['data'] = self.tmpdata+r+self.config['listener']['data_ext']
            source['timestamp'] = self.results[r]['timestamp']
            output['sources'][r] = source

            # Obtain the events found for each source
            for e in self.results[r]['events']:
                # Check if the event was already registered
                id = ""
                for he in self.events:
                    # TODO: Try to do something more 'smart'
                    if( abs(e['time'] - he['time']) <= float(tth) ):
                        id = he['event']

                # Create event
                if( id == "" ):
                    id = str(uuid.uuid1())
                    #print("Creating ID:", id)
                    events[id] = []

                # Insert the event on the sorted list
                self.events.append({'time': e['time'], 'event': id})

                # Convert event timestamp in a readable format
                #['time'] = datetime.datetime.utcfromtimestamp(e['time']).strftime('%d/%m/%Y, %H:%M:%S')
                e['time'] = datetime.datetime.utcfromtimestamp(e['time']).strftime('%Y/%m/%d, %H:%M:%S')
                # Add the agency information
                e['agency'] = r

                # Set the information provided by the specific agency
                # TODO: Add mechanisms able to detect if an event was already
                #       triggered.
                if( id in events.keys() ):
                    events[id].append(e)

        # Just check if some event was registered by any agency
        if( events.keys() ):
            output['events'] = events
        else:
            output = None

        # Return the set of events found
        return(output)
