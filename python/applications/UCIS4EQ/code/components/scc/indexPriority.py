#!/usr/bin/env python3
#
# Events dispatcher
# This module is part of the Smart Center Control (SSC) solution
#
# Author:  Marisol Monterrubio Velasco, Juan Esteban Rodríguez
# Contact: marisol.monterrubio@bsc.es, juan.rodriguez@bsc.es
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
# along with this program. If not, see <https://www.gnu.org/licenses/>.

################################################################################
# Module imports
# System
import sys
import traceback
import json

from bson.json_util import dumps
from bson import ObjectId

# Third parties
from flask import jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC  
from haversine import haversine

# Internal
from ucis4eq.misc import config, principalComponents, microServiceABC
import ucis4eq.dal as dal
from ucis4eq.dal import staticDataMap

################################################################################
# Methods and classes

@staticDataMap.build
class IndexPriority(microServiceABC.MicroServiceABC):

    # Initialization method
    def __init__(self):
        """
        Initialize the eventDispatcher component implementation    
        """

    # Service's entry point definition
    @microServiceABC.MicroServiceABC.runRegistration
    def entryPoint(self, body):
        """
        This method will figure out the index priority for a given event
        """
        
        alertPriority = []
            
        alert = body['alerts'][0]
        self.country = self._getPlace(alert['latitude'], alert['longitude'])
        
        # TODO: Loop for each agency 
        self.latitude = alert['latitude']
        self.longitude = alert['longitude']
        self.depth = alert['depth']
        self.magnitude = alert['magnitude']
        self.hour = 0.0   
        self.intensity = -1000        
        alertPriority.append(self._indexPriorityCalculation())        
        
        # TODO: Unify the decision in one value considering the different agencies 
        
        # Return list of Id of the newly created item
        return jsonify(result = alertPriority[0], response = 201)   

    def _indexPriorityCalculation(self):
        """
        This method will compute the index priority for a given event
        """
        
        # TODO: Put your code here @Marisol
        
        # load databases
        classifierSVM_file = "Classifier_SVM1990-6"
        classifierRF_file = "Classifier_RF1990-6"
        classifierXGB_file = "Classifier_XGB1990-6"
        dataPGA = "dataPGA"
        # dataPGA_folder = dataPGAPath
        fullDB_file = "Full_DataBase"
        fullPGA_file = "VecKPGA_FULL_DataBase"
        countryDB_file = "UnionFiles_Total_File_Filled"
        latlonDB_file = "DataBase_LatLon"
                
        dfNewEarthquake = pd.DataFrame()
        dfNewEarthquake['Country'] = [self.country]
        dfNewEarthquake['Hour'] = [self.hour]
        dfNewEarthquake['Intensity'] = [self.intensity]
        dfNewEarthquake['FocalDepth'] = [self.depth/1000]   
        dfNewEarthquake['Magnitude'] = [self.magnitude]
        dfNewEarthquake['Latitude'] = [self.latitude]
        dfNewEarthquake['Longitude'] = [self.longitude]        
        CountryNewEQ = dfNewEarthquake['Country']
        
        latVec = np.arange(-90,100,1)
        for i in np.arange(len(latVec)):
            if latVec[i] <= dfNewEarthquake['Latitude'][0] < latVec[i+1]:
                nflag = latVec[i]
                nflagP = latVec[i-1]
                nflagL = latVec[i+1]                
                break
                
        dataPGAPath = self.fileMapping[dataPGA] + "/"
        
        dataPGAlocal = np.loadtxt(dataPGAPath +'1Degree-PGA_db'+str(nflag)+'.DAT')
        print(dataPGAPath+'1Degree-PGA_db-'+str(nflag)+'.DAT')
        df2 = pd.DataFrame()
        df2['Longitude'] = dataPGAlocal[:,1]
        df2['Latitude'] = dataPGAlocal[:,0]
        df2['PGA'] = dataPGAlocal[:,2]        
        print('df2', df2)
        dataPGAlocalP = np.loadtxt(dataPGAPath+'1Degree-PGA_db'+str(nflagP)+'.DAT')
        print(dataPGAPath+'1Degree-PGA_db-'+str(nflagP)+'.DAT')
        df3 = pd.DataFrame()
        df3['Longitude'] = dataPGAlocalP[:,1]
        df3['Latitude'] = dataPGAlocalP[:,0]
        df3['PGA'] = dataPGAlocalP[:,2] 
        print('df3', df3)
        dataPGAlocalL = np.loadtxt(dataPGAPath+'1Degree-PGA_db'+str(nflagL)+'.DAT')
        print(dataPGAPath+'1Degree-PGA_db-'+str(nflagL)+'.DAT')
        df4 = pd.DataFrame()
        df4['Longitude'] = dataPGAlocalL[:,1]
        df4['Latitude'] = dataPGAlocalL[:,0]
        df4['PGA'] = dataPGAlocalL[:,2]
        print('df4', df4)
        frames = [df3, df2, df4]
        df1 = pd.concat(frames)  
        print(df1, flush=True)
        
        k_neigh = 40    # number of neighbors to consider their PGA value
        PGAneigbors = np.zeros(k_neigh)
        Threshold = 20  # in km
        A = [1]         # one event 
        VecKPGA = np.zeros((len(A),2))
        for j in np.arange(len(A)):           
            PGAneigbors = np.zeros(k_neigh)
            NeighborsID = np.zeros(k_neigh)
            cont = 0
            for i in np.arange(len(df1)):        
                dataCoordinatesNew = (dfNewEarthquake['Latitude'],dfNewEarthquake['Longitude'])
                dataCoordinates = (df1['Latitude'].iloc[i], df1['Longitude'].iloc[i])
                data_Dist = haversine(dataCoordinates, dataCoordinatesNew) #Haversine distance in km
                if data_Dist <= Threshold:
                    PGAneigbors[cont] = df1['PGA'].iloc[i]
                    NeighborsID[cont] = i
                    cont += 1                    
                    if cont == k_neigh-1:
                        break
                            
            VecKPGA[j,0] = np.mean(PGAneigbors[0:cont])
            VecKPGA[j,1] = np.std(PGAneigbors[0:cont])
                          
        
        print('VecKPGA', VecKPGA, flush = True)
        
        dfFull = pd.read_pickle(self.fileMapping[fullDB_file])
        PGAFull = np.loadtxt(self.fileMapping[fullPGA_file])
        df = pd.DataFrame()
        df['PGAMean'] = PGAFull[:,0]
        df['PGAStd'] = PGAFull[:,1]
        dfFullNew = pd.concat([dfFull, df], axis=1)
        
        print(dfFullNew['Country'], flush=True)
        print('CountryNewEQ', CountryNewEQ, flush=True)
        
        Z = dfFullNew.index[dfFullNew['Country'].values == CountryNewEQ.values].tolist()
        SimilarEarthquakes = []
        for i in np.arange(len(Z)):
            index = Z[i]
            SimilarEarthquakes.append(dfFullNew.iloc[Z[i]])
          
          
        print('Z', Z, flush = True)  
        dfCountry = pd.read_table(self.fileMapping[countryDB_file])
        print(dfCountry['Country'])
        A = dfCountry.index[dfCountry['Country'].values == CountryNewEQ.values].tolist()
        print('A', A)
        
        j = 0
        dfNewEarthquake['HumanSeat'] = [dfCountry.iloc[A[0],1]]
        dfNewEarthquake['IncomingSal'] = [dfCountry.iloc[A[0],2]]
        dfNewEarthquake['InfIndex'] = [dfCountry.iloc[A[0],3]]
        dfNewEarthquake['PopSize'] = [dfCountry.iloc[A[0],4]]
        dfNewEarthquake['IndexQuality'] = [dfCountry.iloc[A[0],5]]
        dfNewEarthquake['PGAMean'] = [VecKPGA[j,0]]
        dfNewEarthquake['PGAStd'] = [VecKPGA[j,1]]
        print('dfNewEarthquake', dfNewEarthquake, flush = True)
        dfLatLon = pd.read_pickle(self.fileMapping[latlonDB_file])
        dfLatLon = dfLatLon.append(dfNewEarthquake,sort=False)
      
        dfLatLon = dfLatLon.drop(['Country'],axis=1)
        scaler = MinMaxScaler(feature_range=(0,1))
        df_Sca = pd.DataFrame(scaler.fit_transform(dfLatLon), columns=dfLatLon.columns)
        
        
        data = [df_Sca['Latitude'],df_Sca['Longitude']] #.iloc[:,5]]
        print('data', data, flush=True)
        headers = ["Latitude", "Longitude"] 
        df3 = pd.concat(data, axis=1, keys=headers)
        df_Sca = df_Sca.drop(['PGAStd'],axis=1)
        num_components=1
        var_threshold=0.99
        
        print('entre_a_PCA')
        pca = PCA()
        pca.fit(df3.T)
	
        pcomp = pca.components_
        pcomp = pcomp.T # take transopose in order to get as (n_features, n_components)
	
        if num_components is None:
          num_components = pd.Series(pca.explained_variance_ratio_).cumsum()
          num_components = num_components[num_components < var_threshold].size
    
        pcomp = pd.DataFrame(pcomp[:,:num_components])
        pcomp.index = df3.index
        indice = pcomp     
        
        print('indice', indice, flush=True)
        
        df_Sca['indexLocation'] = indice
        cols = ['indexLocation']  + [col for col in df_Sca if col != 'indexLocation']
        df_Sca = df_Sca[cols]
        cols = ['PGAMean']  + [col for col in df_Sca if col != 'PGAMean']
        df_Sca = df_Sca[cols]
        df_Sca = df_Sca.drop(['Latitude'],axis = 1)
        df_Sca = df_Sca.drop(['Longitude'],axis = 1)
        df_Sca = df_Sca.drop(['Hour'],axis = 1)
        df_Sca = df_Sca.drop(['IncomingSal'],axis = 1)
        df_Sca.iloc[-1]['FocalDepth']
        print(df_Sca, flush=True)
        dfTest = pd.DataFrame()
        dfTest['PGAMean'] = [df_Sca.iloc[-1]['PGAMean']]
        dfTest['indexLocation'] = [df_Sca.iloc[-1]['indexLocation']*-1.0]
        dfTest['FocalDepth'] = [df_Sca.iloc[-1]['FocalDepth']]
        dfTest['Magnitude'] = [df_Sca.iloc[-1]['Magnitude']]
        dfTest['InfIndex'] = [df_Sca.iloc[-1]['InfIndex']]
        dfTest['IndexQuality'] = [df_Sca.iloc[-1]['IndexQuality']]
        print(dfTest, flush=True)
        
        modelRF = pickle.load(open(self.fileMapping[classifierRF_file], 'rb'))
        UrgencyIndexRF = modelRF.predict(dfTest)
        PredRF = modelRF.predict_proba(dfTest)
        
        modelXGB = pickle.load(open(self.fileMapping[classifierXGB_file], 'rb'))
        UrgencyIndexXGB = modelXGB.predict(dfTest)
        PredXGB = modelXGB.predict_proba(dfTest)
        
        modelSVM = pickle.load(open(self.fileMapping[classifierSVM_file], 'rb'))
        UrgencyIndexSVM = modelSVM.predict(dfTest)
        PredSVM = modelSVM.predict_proba(dfTest)
        
        dfSE = pd.DataFrame(SimilarEarthquakes) 
        highPriorityIndex = 0
        if UrgencyIndexRF[0] + UrgencyIndexXGB[0] + UrgencyIndexSVM[0] >= 2:
          highPriorityIndex = 1
        elif UrgencyIndexRF[0] + UrgencyIndexXGB[0] + UrgencyIndexSVM[0] <= 1:
          highPriorityIndex = 0
          
        print('highPriorityIndex', highPriorityIndex, flush=True)
        print('dfSE',dfSE.to_dict())

                      
        return {'priorityIndex':highPriorityIndex, 'similarEarthquakes': dfSE.to_dict()} 

    def _getPlace(self, lat, lon):

      dfFull = pd.read_pickle(self.fileMapping["Full_DataBase"])
      dfFullVec = dfFull["Country"].unique()
      country_Name = ""

    
    # data = requests.get("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson").json()
      with open(self.fileMapping["Countries_OnShore"], 'r', encoding='utf-8') as f:
          data = json.load(f)
          countries = {}
          for feature in data["features"]:
              geom = feature["geometry"]
              country = feature["properties"]["ADMIN"] # countries.geojson
              countries[country] = prep(shape(geom))
            
          def get_country(lon, lat):
              point = Point(lon, lat)
              for country, geom in countries.items():
                  if geom.contains(point):
                      return country    
                  
          country_Name = get_country(lon, lat)        
          print("country_Name_OnShore", country_Name)
          if country_Name and country_Name.upper() in dfFullVec:                          
              return country_Name.upper()
                        
          if not country_Name:
              with open(self.fileMapping["Countries_OffShore"], 'r', encoding='utf-8') as f:
                  data = json.load(f)               

                  properties = ["TERRITORY1", "SOVEREIGN1", "SOVEREIGN2", "TERRITORY2", "GEONAME"]
                  for proper in properties:
                      print("proper",proper)
                      countries = {}   
                      for feature in data["features"]:                            
                          geom = feature["geometry"]
                          country = feature["properties"][proper] # eez_v11.json  #JAPAN
                          countries[country] = prep(shape(geom))        
                      def get_country(lon, lat):
                          point = Point(lon, lat)
                          for country, geom in countries.items():
                              if geom.contains(point):
                                  return country             
                      country_Name = get_country(lon, lat)                   
                      print("country_Name",country_Name)
                      if country_Name and country_Name.upper() in dfFullVec:                          
                          return country_Name.upper()
                          break        
