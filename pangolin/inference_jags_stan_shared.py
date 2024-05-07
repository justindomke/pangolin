from . import interface, dag, util, inference

import textwrap


class Reference:
    def __init__(self, id, shape, loop_indices=None):
        if loop_indices is None:
            loop_indices = [None] * len(shape)
        self.id = id
        self.shape = shape
        self.loop_indices = loop_indices

    def __str__(self):
        if all(i is None for i in self.loop_indices):
            return self.id
        ret = self.id + "["
        for n, i in enumerate(self.loop_indices):
            if i:
                ret += i
            if n < len(self.loop_indices) - 1:
                ret += ","
        ret += "]"
        return ret

    def nth_open_axis(self, axis):
        return util.nth_index(self.loop_indices, None, axis)

    def index(self, axis, i):
        if axis is None:
            return self

        # i should go in the axis-th empty slot
        where = self.nth_open_axis(axis)

        new_loop_indices = self.loop_indices.copy()
        new_loop_indices[where] = i
        return Reference(self.id, self.shape, new_loop_indices)

    def index_mult(self, indices):
        rez = self
        for i in indices:
            rez = rez.index(0, i)
        return rez

    # def index_shifted(self, axis, i, where_to_place):
    #     tmp_indices = self.index(axis, i).loop_indices
    #     where = self.nth_open_axis(axis)
    #     new_loop_indices = util.swapped_list(tmp_indices, where, where_to_place)
    #     return Reference(self.id, self.shape, new_loop_indices)

    @property
    def num_empty(self):
        return sum(x is None for x in self.loop_indices)

    @property
    def ndim(self):
        return len(self.shape)


class Helper:
    def __init__(self, backend, assignment_operator, full_slice_operator):
        self.backend = backend  # "JAGS" / "Stan"
        self.assignment_operator = assignment_operator  # "<-" / "="
        self.full_slice_operator = full_slice_operator  # "" / ":"

    def indent(self, code, n):
        return textwrap.indent(code, "    " * n)

    def gencode_infix_factory(self, infix_str):
        def gencode_infix(cond_dist, loopdepth, ref, *parent_refs):
            return (
                f"{ref} {self.assignment_operator} ({parent_refs[0]}) {infix_str} "
                f"({parent_refs[1]});\n"
            )

        return gencode_infix

    def gencode_deterministic_factory(self, fun_str):
        def gencode_deterministic(cond_dist, loopdepth, ref, *parent_refs):
            return (
                f"{ref} {self.assignment_operator} {fun_str}"
                f"{util.comma_separated(parent_refs,str)};\n"
            )

        return gencode_deterministic

    def slice_to_str(self, my_slice, ref):
        start = my_slice.start
        stop = my_slice.stop
        step = my_slice.step
        if start is None:
            start = 0
        if stop is None:
            # use next unused axis
            axis = ref.nth_open_axis(0)
            stop = ref.shape[axis]
        if step:
            raise NotImplementedError(
                f"{self.backend} doesn't support step in slices " f":("
            )

        # JAGS 1-indexed and inclusive so start increase but not stop
        loop_index_str = f"{start + 1}:{stop}"
        loop_shape_str = f"1:{stop - start}"
        return loop_index_str, loop_shape_str

    def gencode_index(self, cond_dist, loopdepth, ref, parent_ref, *index_refs):
        check_gencode_index_inputs(cond_dist, ref, parent_ref, *index_refs)

        # currently can only index with 1d or 2d arrays

        loop_code = ""
        end_code = ""

        index_ndim = num_index_dims(index_refs)

        ref_loop_index_needed = True
        idx_loop_indices = []
        if index_ndim is not None:
            first_shape = index_refs[0].shape

            # idx_loop_indices = []
            for n in range(index_ndim):
                idx_loop_index = f"l{loopdepth}"
                loop_code += f"for ({idx_loop_index} in 1:{first_shape[n]})" + "{" "\n"
                end_code = "}\n" + end_code
                idx_loop_indices.append(idx_loop_index)
                loopdepth += 1

            # add loop indices for LHS if should go at start
            if cond_dist.advanced_at_start:
                ref = ref.index_mult(idx_loop_indices)
                ref_loop_index_needed = False

        index_refs_iter = iter(index_refs)
        # go through all slots in cond_dist (Index object)
        for my_slice in cond_dist.slices:
            if my_slice:
                loop_index_str, loop_shape_str = self.slice_to_str(my_slice, parent_ref)
                parent_ref = parent_ref.index(0, loop_index_str)
                ref = ref.index(0, loop_shape_str)
            else:
                # grab next index parent and add index loop indices
                my_index_ref = next(index_refs_iter).index_mult(idx_loop_indices)

                # add loop indices for LHS if should go here
                if ref_loop_index_needed:
                    ref = ref.index_mult(idx_loop_indices)
                    ref_loop_index_needed = False

                parent_loop_index = "1+" + str(my_index_ref)  # JAGS 1 indexed
                parent_ref = parent_ref.index(0, parent_loop_index)

        middle_code = str(ref) + f"{self.assignment_operator}" + str(parent_ref) + ";\n"
        middle_code = middle_code
        code = loop_code + middle_code + end_code
        return code

    def gencode_dist_factory(self, name, perm=None):
        """
        Create a code generator.
        perm is an optional permutation of the indices (can drop or reverse etc.)
        """

        def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
            if perm:
                new_parent_refs = tuple(parent_refs[p] for p in perm)
            else:
                new_parent_refs = parent_refs
            return f"{ref} ~ {name}" + util.comma_separated(new_parent_refs, str) + ";\n"
            # return f"{ref} ~ {name}" + util.comma_separated(parent_refs, str) + ";\n"

        return gencode_dist

    # def gencode_dist_factory_swapargs(self, name):
    #     def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
    #         assert len(parent_refs) == 2
    #         new_parent_refs = (parent_refs[1], parent_refs[0])
    #         return f"{ref} ~ {name}" + util.comma_separated(new_parent_refs, str) + "\n"
    #
    #     return gencode_dist

    # def gencode_dist_factory_permute_args(self, name, perm):
    #     def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
    #         assert len(parent_refs) == 2
    #         new_parent_refs = tuple(parent_refs[p] for p in perm)
    #         return f"{ref} ~ {name}" + util.comma_separated(new_parent_refs, str) + "\n"
    #
    #     return gencode_dist

    def gencode_sum(self, cond_dist, loopdepth, ref, parent_ref):
        assert isinstance(cond_dist, interface.Sum)
        axis = cond_dist.axis
        assert axis is not None, "Stan/JAGs only support sum along single integer axis"
        assert isinstance(
            axis, int
        ), "Stan/JAGs only support sum along single integer axis"
        loop_code = ""
        end_code = ""
        for n in range(ref.ndim):
            if n == axis:
                parent_ref = parent_ref.index(0, f"{self.full_slice_operator}")
            else:
                loop_index = f"l{loopdepth}"
                open_axis = ref.nth_open_axis(0)
                loop_code += f"for ({loop_index} in 1:{ref.shape[open_axis]})" + "{" "\n"
                end_code = "}\n" + end_code
                loopdepth += 1
                ref = ref.index(0, loop_index)
                parent_ref = parent_ref.index(0, loop_index)
        middle_code = f"{ref} {self.assignment_operator} sum({parent_ref});\n"
        code = loop_code + middle_code + end_code
        return code

    def gencode_categorical_factory(self, name):
        def gencode_categorical(cond_dist, loopdepth, ref, *parent_refs):
            """
            special code needed since Stan is 1-indexed
            """
            assert cond_dist == interface.categorical
            assert len(parent_refs) == 1
            code1 = f"tmp_{ref} ~ {name}({parent_refs[0]})\n"
            code2 = f"{ref} = tmp_{ref}-1\n"
            return code1 + code2

    def gencode_vmapdist_factory(self, gencode):
        def gencode_vmapdist(cond_dist, loopdepth, ref, *parent_refs):
            loop_index = f"i{loopdepth}"

            new_ref = ref.index(0, loop_index)
            new_parent_refs = [
                p_ref.index(axis, loop_index)
                for p_ref, axis in zip(parent_refs, cond_dist.in_axes)
            ]

            # must update cond_dist.axis size in case cond_dist.axis_size==None
            # this block of code is pretty new
            if cond_dist.axis_size is None:
                axis_size = None
                assert len(cond_dist.in_axes) == len(parent_refs)
                for in_axis, parent_ref in zip(cond_dist.in_axes, parent_refs):
                    if in_axis is not None:
                        axis_size = parent_ref.shape[in_axis]
                        break
                if axis_size is None:
                    assert False, "should be impossible"
            else:
                axis_size = cond_dist.axis_size

            loop_code = f"for ({loop_index} in 1:" + str(axis_size) + "){\n"
            middle_code = gencode(
                cond_dist.base_cond_dist, loopdepth + 1, new_ref, *new_parent_refs
            )
            end_code = "}\n"

            middle_code = self.indent(middle_code, 1)

            code = loop_code + middle_code + end_code

            return code

        return gencode_vmapdist


def num_index_dims(index_refs):
    """
    get the number of dimensions in the index refs, or None if empty
    """
    # assert all dimensions same
    for index_ref1 in index_refs:
        for index_ref2 in index_refs:
            assert (
                index_ref1.shape == index_ref2.shape
            ), "all indices must have same dimensions"
    if index_refs:
        return index_refs[0].ndim
    else:
        return None


def check_gencode_index_inputs(cond_dist, ref, parent_ref, *index_refs):
    """
    1. that the number of empty slots in `parent_ref` is the number of slots in
    `cond_dist` (each can be either a slice of a scalar or an array)
    2. that the number of empty slots in `ref` is equal to the number of *slice*s in
    `cond_dist` plus the number of dims on index (if it exists)
    """
    expected_parent_empty_slots = len(cond_dist.slices)
    assert parent_ref.num_empty == expected_parent_empty_slots

    num_actual_slices = sum(1 for my_slice in cond_dist.slices if my_slice)
    if index_refs:
        index_dims = index_refs[0].ndim
        assert ref.num_empty == num_actual_slices + index_dims
