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

