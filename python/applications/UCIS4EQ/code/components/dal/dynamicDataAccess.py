#!/usr/bin/env python3
#
# Abstract factory for accessing storage repositories
# This module is part of the Smart Center Control (SSC) solution
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

from pymongo import MongoClient
import os
import json

from flask import jsonify

import ucis4eq
import ucis4eq.dal as dal
from ucis4eq.dal import staticDataAccess
from ucis4eq.misc import config, microServiceABC

################################################################################
# Methods and classes

class DAL(microServiceABC.MicroServiceABC):
    
    # Initialization method
    def __init__(self):
        """
        Initialize the DAL service
        """

    # Service's entry point definition
    @config.safeRun
    def entryPoint(self, repositoryName = dal.repository):
            
        repository = staticDataAccess.repositories.create(repositoryName, **dal.config)
        
        # Create a directory for DAL documents 
        workSpace = ucis4eq.workSpace + "/DAL/"
        os.makedirs(workSpace, exist_ok=True)    
        
        # Download the file from the repository
        rpath = dal.config[repositoryName]["datamapping"]
        rfile = os.path.basename(rpath)
        lpath = workSpace + rfile
        
        repository.downloadFile(rpath, lpath)

        # Alias for the static data mapping and DB
        db = dal.database
        
        # Create collection (Remove previous one)
        db[dal.StaticDataMappingDocument].drop()
        col = self._createCollection(db, lpath,
                dal.StaticDataMappingDocument)
        
        # Obtain the set of documents needed by the UCIS4EQ before start serving
        dalDocs = { "used_by": {"$eq": "DALService"} }
        docs = col.find(dalDocs)
        
        for doc in docs:
            repo, repoSettings = staticDataAccess.repositories.selectFrom(doc['repositories'],
                    repositoryName)
            rpath = repoSettings['path']
            rfile = os.path.basename(rpath)
            lpath = workSpace + rfile
                
                
            # Download the file from the repository
            if not os.path.exists(lpath):
                print("Downloading file: " + lpath, flush=True)
                repository.downloadFile(rpath, lpath)

            # Create collection
            col = self._createCollection(db, lpath)

        # Return success
        return jsonify(result = {}, response = 201)

    def _createCollection( self, db, documentsPath, collectionName=None):
        """
        This method will create a collection of documents in the given MongoDB
        """
        
        # Obtain the collection name if not provided
        if not collectionName:
            collectionName = os.path.splitext(os.path.basename(documentsPath))[0]
            
        # Define the collection
        collection = db[collectionName]
        
        # Insert all registries
        if documentsPath.endswith(".json"):

            try:
                with open(documentsPath, "r", encoding="utf-8") as f:
                    fdata = json.load(f)
            except ValueError:
                raise Exception('Decoding JSON file ' + documentsPath + ' has failed')
                        
            # Get the set of collections already set in DAL
            list = collection.distinct( "id" )

            if "documents" in fdata:
                ## Insert the new ones
                for item in fdata['documents']:
                    if not item['id'] in list:
                        collection.insert_one(item)
            else:
                ## Insert the new ones
                for item in fdata['docFuments']:
                    if not item['id'] in list:
                        collection.insert_one(item)

        return collection
