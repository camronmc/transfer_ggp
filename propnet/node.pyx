# cython: profile=True
from .constants import (AND, OR, PROPOSITION, TRANSITION, NOT, CONSTANT,
                        UNKNOWN, base, input, legal, goal, terminal,
                        other)
import re
# import numpy as np
# cimport numpy as np  # Has magic cython compile time information

# DTYPE = np.int32
# ctypedef np.int32_t DTYPE_t

LEGAL_RE = re.compile(r'\( *legal *(\w+) *(.+) *\)')
GOAL_RE = re.compile(r'\( *goal *(\w+) *(\d+) *\)')


cdef class Node:
    cdef public int id
    cdef public int cons
    cdef public int node_type
    cdef public list inputs
    cdef public list outputs
    def __init__(self, id, cons, node_type, inputs, outputs):
        self.id = id
        self.cons = cons
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = outputs

        #  if node_type == OR:
            #  self._eval = lambda data, inputs: any(data[i] for i in inputs)
        #  elif node_type == AND:
            #  self._eval = lambda data, inputs: all(data[i] for i in inputs)
        #  elif node_type == CONSTANT:
            #  self._eval = lambda data, inputs: not data[inputs[0]]

    def update_inputs(self, old, new):
        for i, x in enumerate(self.inputs):
            if x == old:
                self.inputs[i] = new

    #  def eval(self, data, init, actions):
        #  return self._eval(data, self.inputs)

    def __str__(self):
        return f'{self.__class__.__name__}(id={self.id}, node_type={self.node_type})'

    __repr__ = __str__


cdef class Or(Node):
    cpdef int eval(self, list data, int init, set actions):
        for i in self.inputs:
            if data[i]:
                return 1
        return 0


cdef class And(Node):
    cpdef int eval(self, list data, int init, set actions):
        for i in self.inputs:
            if not data[i]:
                return 0
        return 1


cdef class Not(Node):
    cpdef int eval(self, list data, int init, set actions):
        return not data[self.inputs[0]]


cdef class Constant(Node):
    cpdef int eval(self, list data, int init, set actions):
        return self.cons


cdef class TransitionPre(Node):
    cdef public list _outputs
    cdef public int post_id
    def __init__(self, post_id, *args):
        super().__init__(*args)
        self._outputs = self.outputs
        self.outputs = []
        self.post_id = post_id

    cpdef int eval(self, list data, int init, set actions):
        return data[self.inputs[0]]


cdef class TransitionPost(Node):
    #  cdef list _inputs
    cdef public int pre_id
    def __init__(self, pre_id, *args):
        super().__init__(*args)
        #  self._inputs = self.inputs
        self.inputs = []
        self.pre_id = pre_id

    cpdef int eval(self, list data, int init, set actions):
        return data[self.pre_id]


def transition_split(args, new_id):
    id, *args = args
    pre = TransitionPre(new_id, id, *args)
    post = TransitionPost(id, new_id, *args)
    return pre, post


cdef class Proposition:
    cdef public int id
    cdef public int node_type
    cdef public list inputs
    cdef public list outputs
    cdef public str prop_type
    cdef public str gdl
    cdef public str move_role
    cdef public str move_gdl
    cdef public str role
    cdef public double score
    cdef public str normalised_gdl
    cdef public int input_id
    def __cinit__(self, int id, int cons, int node_type, inputs, outputs, str prop_type, str gdl):
        self.id = id
        self.node_type = node_type
        self.outputs = outputs
        self.inputs = inputs
        self.prop_type = prop_type
        self.gdl = gdl

        if prop_type == legal:
            self.move_role, self.move_gdl = LEGAL_RE.search(gdl).groups()
            assert(len(inputs) == 1)
        elif prop_type == goal:
            self.role, score = GOAL_RE.match(gdl).groups()
            self.score = int(score)/100
            assert(len(inputs) == 1)
        elif prop_type == input:
            self.normalised_gdl = gdl.replace(' ', '')
            assert(len(inputs) == 0)

    def update_inputs(self, old, new):
        for i, x in enumerate(self.inputs):
            if x == old:
                self.inputs[i] = new

    def set_input(self, data, val):
        assert(self.prop_type == input)
        data[self.id] = val

    cpdef eval(self, list data, int init, set actions):
        if self.prop_type == 'init':
            return init
        if self.prop_type == input:
            return self.id in actions
        if not self.inputs:
            import pdb; pdb.set_trace()
        return data[self.inputs[0]]

    def __str__(self):
        return '%s(id=%d, type=%s, gdl=%s)' % (
            self.__class__.__name__,
            self.id,
            self.prop_type,
            self.gdl,
        )

    __repr__ = __str__


node_types = {
    AND: And,
    OR: Or,
    NOT: Not,
    PROPOSITION: Proposition,
    #  TRANSITION: Transition,
    CONSTANT: Constant,
    # UNKNOWN: Node
}
