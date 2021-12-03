use crate::ChildIndex;

use slab::Slab;

/// Allocates `T` values and blocks of `CHILDREN` [`AllocPtr`]s at a time. One [`AllocPtr`] corresponds to a single node at the
/// given [`Level`](crate::Level).
#[derive(Clone, Debug)]
pub struct NodeAllocator<T, const CHILDREN: usize> {
    /// A slab of pointer blocks. Empty at level 0.
    pointers: Slab<[AllocPtr; CHILDREN]>,
    /// A slab of values.
    values: Slab<T>,
}

/// Points to a node owned by an internal allocator.
pub type AllocPtr = u32;

/// An [`AllocPtr`] that doesn't point to anything.
pub const EMPTY_PTR: AllocPtr = AllocPtr::MAX;

impl<T, const CHILDREN: usize> Default for NodeAllocator<T, CHILDREN> {
    fn default() -> Self {
        Self {
            pointers: Default::default(),
            values: Default::default(),
        }
    }
}

impl<T, const CHILDREN: usize> NodeAllocator<T, CHILDREN> {
    #[inline]
    pub fn insert_leaf(&mut self, value: T) -> AllocPtr {
        self.values.insert(value) as AllocPtr
    }

    #[inline]
    pub fn insert_branch(&mut self, value: T) -> (AllocPtr, &mut [AllocPtr; CHILDREN]) {
        let ptr = self.values.insert(value);
        let pointer_entry = self.pointers.vacant_entry();
        debug_assert_eq!(ptr, pointer_entry.key());

        (ptr as AllocPtr, pointer_entry.insert([EMPTY_PTR; CHILDREN]))
    }

    #[inline]
    pub fn remove(&mut self, ptr: AllocPtr) -> (Option<T>, Option<[AllocPtr; CHILDREN]>) {
        (
            self.values.try_remove(ptr as usize),
            self.pointers.try_remove(ptr as usize),
        )
    }

    #[inline]
    pub fn contains_node(&self, ptr: AllocPtr) -> bool {
        // This assumes that every node has a value.
        self.values.contains(ptr as usize)
    }

    #[inline]
    pub unsafe fn get_value_unchecked(&self, ptr: AllocPtr) -> &T {
        self.values.get_unchecked(ptr as usize)
    }

    #[inline]
    pub unsafe fn get_value_unchecked_mut(&mut self, ptr: AllocPtr) -> &mut T {
        self.values.get_unchecked_mut(ptr as usize)
    }

    #[inline]
    pub fn get_value(&self, ptr: AllocPtr) -> Option<&T> {
        self.values.get(ptr as usize)
    }

    #[inline]
    pub fn get_value_mut(&mut self, ptr: AllocPtr) -> Option<&mut T> {
        self.values.get_mut(ptr as usize)
    }

    #[inline]
    pub fn get_children(&self, ptr: AllocPtr) -> Option<&[AllocPtr; CHILDREN]> {
        self.pointers.get(ptr as usize)
    }

    #[inline]
    pub fn get_children_mut(&mut self, ptr: AllocPtr) -> Option<&mut [AllocPtr; CHILDREN]> {
        self.pointers.get_mut(ptr as usize)
    }

    #[inline]
    pub fn unlink_child(&mut self, parent_ptr: AllocPtr, child_index: ChildIndex) {
        if let Some(children) = self.pointers.get_mut(parent_ptr as usize) {
            children[child_index as usize] = EMPTY_PTR;
        }
    }
}
