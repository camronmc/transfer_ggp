# cython: profile=True
import copy


cdef class PersistentArray:
    cdef public int maxlen
    cdef public int id
    cdef public TreeNode tree
    def __init__(self, maxlen):
        cdef int x = 1
        while x < maxlen:
            x *= 2
        self.maxlen = x
        self.tree = TreeNode(0, self.maxlen, 0)
        self.id = 0

    def copy(self):
        new = copy.copy(self)
        new.id += 1
        return new

    cdef int get(self, int i):
        return self.tree.query(i, 0, self.maxlen)

    cdef void set(self, int i, int value):
        self.tree = self.tree.update(i, 0, self.maxlen, self.id, value)

    def setAll(self, arr):
        self.tree = self.tree.updateAll(0, self.maxlen, self.id, arr)

    def __len__(self):
        return self.maxlen

    def __getitem__(self, key):
        if key >= self.maxlen or key < 0:
            raise IndexError(f'{key} out of bounds for PersistentArray of size {self.maxlen}')
        return self.get(key)

    def __setitem__(self, key, value):
        if key >= self.maxlen or key < 0:
            raise IndexError(f'{key} out of bounds for PersistentArray of size {self.maxlen}')
        self.set(key, value)

    cpdef values(self):
        cdef list ret = []
        self.tree.values(0, self.maxlen, ret)
        return ret

    cpdef items(self):
        cdef list ret = []
        self.tree.items(0, self.maxlen, ret)
        return ret


cdef class TreeNode:
    cdef public int id
    cdef public int value
    cdef public TreeNode left
    cdef public TreeNode right

    def __cinit__(self, int start, int end, int id, int create=True):
        self.id = id
        cdef int mid = (start+end) // 2
        if start + 1 == end:
            self.value = False
        elif create:
            self.left = TreeNode(start, mid, id)
            self.right = TreeNode(mid, end, id)

    cdef int query(self, int i, int start, int end):
        if start + 1 == end:
            return self.value
        cdef int mid = (start+end) // 2
        if i < mid:
            return self.left.query(i, start, mid)
        else:
            return self.right.query(i, mid, end)

    cdef TreeNode update(self, int i, int start, int end, int id, int value):
        cdef TreeNode new = self
        cdef int mid = (start+end) // 2
        if start + 1 == end:
            if value == self.value:
                return self
            if self.id != id:
                new = TreeNode(start, end, id)
            new.value = value
            return new
        else:
            if i < mid:
                new = self.left.update(i, start, mid, id, value)
                if self.id == id:
                    self.left = new
                    return self
                else:
                    me = TreeNode(start, end, id, create=False)
                    me.left = new
                    me.right = self.right
                    return me
            else:
                new = self.right.update(i, mid, end, id, value)
                if self.id == id:
                    self.right = new
                    return self
                else:
                    me = TreeNode(start, end, id, create=False)
                    me.left = self.left
                    me.right = new
                    return me

    cdef void values(self, int start, int end, list ret):
        if start + 1 == end:
            ret.append(self.value)
        else:
            mid = (start+end) // 2
            self.left.values(start, mid, ret)
            self.right.values(mid, end, ret)

    cdef void items(self, int start, int end, list ret):
        if start + 1 == end:
            ret.append((start, self.value))
        else:
            mid = (start+end) // 2
            self.left.values(start, mid, ret)
            self.right.values(mid, end, ret)
