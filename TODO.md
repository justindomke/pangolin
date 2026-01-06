# Before 0.0.1

* Bring sanity to indexing

  * Outer indexing by default
  * Offer vectorized indexing `A.v[x,y,z]` obey scalar broadcasting

* Offer optional OCD array indexing mode

* Add super().__init__ to all Op

# After 0.0.1

* Figure out what to do about vectorized indexing.
  * Maybe allow special slices? Maybe do something else?

* Special Slice type that's a subtype of Constant
  * Possibly other Constant types?

* Make VMap generic so you can have VMap[VMap] etc.

* Generalize numpy broadcasting
  * Unify with logic for indexing

* Offer field for vmap

  ```python
  i = Index(5)
  j = Index(10)
  x = Field()
  x[i, j] = a[i] * B[i,j] * c[j]
  
  x = field(
    (5,10),
    lambda i, j : a[i] * B[i,j] * c[j]
  )
  ```

  I guess `Field` should be a subclass of `RV` 

* Offer field for scan/vmap

  ```python
  i = Index(5)
  j = Index(10)
  x = Field()
  x[start,j] = a[j]
  x[i,j] = x[i-1,j] + B[i,j]

  x = field(
    (5, 10),
    rule = lambda i, j, x: x[i-1,j] + B[i,j]
    init = lambda i, j: a[j]
  )
  ```

* Fakeloops

*  Many Op like Softmax or Sum would often be mapped over arrays. Should these take axis arguments? Or should that be the perogative of VMap and dealt with at the interface level?
    * Lean towards a "ReductionOp" class in the IR

* `vmap(normal)([0,1,2], [3,4,5])` should raise a better error message

* Improve `get_shape` functions / docs for scalar Op to better document number of args etc

* More complete unified backend tests

* Offer constraints:
  * Possibly just for scalar dists
  * Possibly offer an unnormalized_log_prob function (tricky because matters if constraints are constant)