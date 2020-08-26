import numpy as np


class Linearizer():
    
    def __init__(self,rounded=None,sample=None):
        self.rounded = rounded
        self.sample = sample   
    
    
    def fit(self,fit_array):
        array = fit_array.copy()
        array = array[~np.isnan(array)]
        
        self.contains_zero = (0.0 in array) and (0.0 != array.max()) and (0.0 != array.min())
                
        if self.contains_zero:
            self.d2z = (array[array < 0.].max()/2,array[array > 0.].min()/2)
            
        if self.sample:
            array = np.random.choice(array,min(self.sample,len(array)),replace=False)            
        
        if self.rounded != None:
            _d = .1**(self.rounded+1)
        else:
            unique_array = np.unique(array)
            if len(array) - len(unique_array):
                sorted_array = np.sort(unique_array)
                array_diff = sorted_array[1:] - sorted_array[:-1]
                _d = array_diff.min()/10
            else:
                _d = 0.

        array = np.sort(array + np.random.uniform(-_d,_d,size=array.size))
                
        distance = array[1:] - array[:-1]
        unit_distance = distance.mean()
        
        self.space = array[:-1]
        self.new_space = self.space[0] + (unit_distance * np.arange(len(self.space)))

        self.scalers = unit_distance / distance
        self.inverse_scalers = 1 / self.scalers

        return self
        
    
    def _map_points(self,arr,space,new_space,scalers):
        
        mapped_location = (arr.reshape(-1,1) >= space[1:].reshape(1,-1)).sum(axis=1)
        mapped_array = new_space[mapped_location] + ((arr - space[mapped_location]) * scalers[mapped_location])
        
        return mapped_array
        
        
    def transform(self,arr):
        
        nan_arr = arr.astype(float).copy()
        arr = nan_arr[~np.isnan(nan_arr)].copy()
        
        mapped_arr = self._map_points(arr,self.space,self.new_space,self.scalers)
        nan_arr[~np.isnan(nan_arr)] = mapped_arr
        
        return nan_arr
    
    
    def fit_transform(self,arr):
        
        nan_arr = arr.astype(float).copy()
        arr = nan_arr[~np.isnan(nan_arr)].copy()
        
        self.fit(arr)
        
        mapped_arr = self._map_points(arr,self.space,self.new_space,self.scalers)
                
        nan_arr[~np.isnan(nan_arr)] = mapped_arr
        
        return nan_arr

    
    def inverse_transform(self,arr):
        
        nan_arr = arr.astype(float).copy()
        arr = nan_arr[~np.isnan(nan_arr)].copy()
        
        inv_arr = self._map_points(arr,self.new_space,self.space,self.inverse_scalers)
        
        if self.contains_zero:
            inv_arr[(self.d2z[0] < inv_arr) & (inv_arr < self.d2z[1])] = 0.
        
        if type(self.rounded) == int:
            inv_arr = inv_arr.round(self.rounded)
            if self.rounded == 0:
                inv_arr = inv_arr.astype(int)
                
        nan_arr[~np.isnan(nan_arr)] = inv_arr
                
        return nan_arr