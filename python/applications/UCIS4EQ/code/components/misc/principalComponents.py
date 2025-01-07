#!/usr/bin/env python3
#
# Encode and parse the query string params
# This module is part of the Automatic Alert System (AAS) solution
#
# Author:  Jose Carlos Carrasco 
# Contact: XXXXX # TODO
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

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

################################################################################
# Methods and classes

def extract_pca_components(data_df, num_components=None, var_threshold=0.99):
    """ Extract PCA components
    Extract principal components from PCA

    Parameters
    ----------
    data_df: pd.DataFrame
        data frame containg data
    num_components: int
        number of components to be returned
    
    Returns
    -------
    pcomp: pd.DataFrame
        data frame with selected number of principal components
    """
    
    print('entre_a_PCA')
    pca = PCA()
    pca.fit(data_df.T)
	
    pcomp = pca.components_
    pcomp = pcomp.T # take transopose in order to get as (n_features, n_components)
	
    if num_components is None:
    	num_components = pd.Series(pca.explained_variance_ratio_).cumsum()
    	num_components = num_components[num_components < var_threshold].size
	
    pcomp = pd.DataFrame(pcomp[:,:num_components])
    pcomp.index = data_df.index
	
    return pcomp
