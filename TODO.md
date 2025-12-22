* Bring sanity to indexing

* Offer pointwise / scalar indexing `A._[x,y,z]` (obey scalar broadcasting rules?)

* Offer optional OCD array indexing mode

* Fakeloops

*  Many Op like Softmax or Sum would often be mapped over arrays. Should these take axis arguments? Or should that be the perogative of VMap and dealt with at the interface level?
    * Lean towards a "ReductionOp" class in the IR

* `vmap(normal)([0,1,2], [3,4,5])` should raise a better error message

* Improve `get_shape` functions / docs for scalar functions to better document number of args etc

* Offer constraints:
  * Possibly just for scalar dists
  * Possibly offer an unnormalized_log_prob function (tricky because matters if constraints are constant)