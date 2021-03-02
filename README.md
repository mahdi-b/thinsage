# Thin Sage

* Repo for data thinning code base used in SAGE 3.


### Notes

sklearn groups methods into classes and classes into files/folders,
utilize other libraries such as scipy
docs go in all methods at the beginning
lots of decorators

sklearn model Selection
  train_test_split
  split is analogous to subsampling?
  
random:
  given
    collection/list/array-like of OBJECTS
    percentage/fraction/size

stratified (w known class assignments)
	balanced and imbalanced
  collection of OBJECTS
  list tells class/category assigned to each object


explore sample from time series
  LTTB: Largest Triangle Three Bucket
  "english" explanation: https://medium.com/@hayley.morrison/sampling-time-series-data-sets-fc16caefff1b
  python code: https://github.com/devoxi/lttb-py/blob/master/lttb/lttb.py
  takes in data: list of tuples (x, y) representing points in the time series
           threshold: desired size (int) of subsample to be returned

# We need to list example algorithms and

from subsample library/package
  (only does rows of text delim='\n')

  random reservoir: 
    create empty "reservoir" with parameter size. 
    iterate through data
    for each item generate random number r between 0 and iteration i 
    if r < size 
      add item to reserrvoir

  approx sample:
    parameter fraction gives estimated fraction of original data that is desired
    for each item
      if random.Random() < fraction
        yield item
      
yield is like return but does not end the function. also no need to create a list in the function. save space?

