use crate::allocator::{AllocPtr, NodeAllocator, EMPTY_PTR};
use crate::{BranchShape, ChildIndex, Level, SmallKeyHashMap, VectorKey};

use smallvec::SmallVec;
use std::collections::{hash_map, VecDeque};
use std::marker::PhantomData;
use std::mem;

/// Uniquely identifies a node slot in the [`Tree`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeKey<V> {
    pub level: Level,
    pub coordinates: V,
}

/// Uniquely and stably identifies an occupied node in the [`Tree`] (until the node is removed).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodePtr {
    pub(crate) level: Level,
    pub(crate) alloc_ptr: AllocPtr,
}

impl NodePtr {
    #[inline]
    pub fn level(&self) -> Level {
        self.level
    }

    /// Null pointers can only be gotten by manually calling `Tree::child_pointers`.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.alloc_ptr == EMPTY_PTR
    }
}

/// Info about a node's parent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Parent<V> {
    pub ptr: NodePtr,
    pub coordinates: V,
    pub child_index: ChildIndex,
}

/// A child [`NodePtr`] and its [`Parent`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildRelation<V> {
    pub child: NodePtr,
    pub parent: Option<Parent<V>>,
}

/// All children pointers for some branch node. Some may be [`EMPTY_PTR`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildPointers<'a, const CHILDREN: usize> {
    pub level: Level,
    pub pointers: &'a [AllocPtr; CHILDREN],
}

/// A generic "grid tree" which can be either a quadtree or an octree depending on the type parameters.
#[derive(Clone, Debug)]
pub struct Tree<V, S, T, const CHILDREN: usize> {
    /// 2x2 square in 2D or 2x2x2 cube in 3D.
    branch_shape: PhantomData<S>,
    /// Every node at the highest LOD is a root.
    root_nodes: SmallKeyHashMap<V, AllocPtr>,
    /// An allocator for each level.
    allocators: Vec<NodeAllocator<T, CHILDREN>>,
}

impl<V, S, T, const CHILDREN: usize> Tree<V, S, T, CHILDREN>
where
    V: VectorKey,
    S: BranchShape<V>,
{
    /// This constructor is only necessary if you need to use a custom shape `S` or vector `V`. Otherwise use the constructor
    /// for [`OctreeI32`](crate::OctreeI32) or [`QuadtreeI32`](crate::QuadtreeI32).
    ///
    /// # Safety
    ///
    /// You must guarantee that `S` correctly linearizes `V` vectors so that they don't go out of array bounds in a block.
    /// Refer to [`QuadtreeShapeI32`](crate::QuadtreeShapeI32) and [`OctreeShapeI32`](crate::OctreeShapeI32) for examples.
    pub unsafe fn new_generic(height: Level) -> Self {
        assert!(height > 1);
        Self {
            branch_shape: PhantomData,
            root_nodes: Default::default(),
            allocators: (0..height).map(|_| NodeAllocator::default()).collect(),
        }
    }

    /// The number of [`Level`]s in this tree.
    #[inline]
    pub fn height(&self) -> Level {
        self.allocators.len() as Level
    }

    /// The [`Level`] where root nodes live.
    #[inline]
    pub fn root_level(&self) -> Level {
        self.height() - 1
    }

    /// Returns the unique root that is an ancestor of `descendant_key`.
    #[inline]
    pub fn ancestor_root_key(&self, descendant_key: NodeKey<V>) -> NodeKey<V> {
        assert!(descendant_key.level <= self.root_level());
        let levels_up = self.root_level() - descendant_key.level;
        let coordinates = S::ancestor_key(descendant_key.coordinates, levels_up as u32);
        NodeKey {
            level: self.root_level(),
            coordinates,
        }
    }

    /// Returns true iff this tree contains a node for `ptr`.
    #[inline]
    pub fn contains_node(&self, ptr: NodePtr) -> bool {
        self.allocator(ptr.level).contains_node(ptr.alloc_ptr)
    }

    /// Returns true iff this tree contains a root node at `coords`.
    #[inline]
    pub fn contains_root(&self, coordinates: V) -> bool {
        self.root_nodes.contains_key(&coordinates)
    }

    #[inline]
    pub fn get_value(&self, ptr: NodePtr) -> Option<&T> {
        self.allocator(ptr.level).get_value(ptr.alloc_ptr)
    }

    #[inline]
    pub fn get_value_mut(&mut self, ptr: NodePtr) -> Option<&mut T> {
        self.allocator_mut(ptr.level).get_value_mut(ptr.alloc_ptr)
    }

    /// # Safety
    ///
    /// The node for `ptr` must exist.
    #[inline]
    pub unsafe fn get_value_unchecked(&self, ptr: NodePtr) -> &T {
        self.allocator(ptr.level).get_value_unchecked(ptr.alloc_ptr)
    }

    /// # Safety
    ///
    /// The node for `ptr` must exist.
    #[inline]
    pub unsafe fn get_value_unchecked_mut(&mut self, ptr: NodePtr) -> &mut T {
        self.allocator_mut(ptr.level)
            .get_value_unchecked_mut(ptr.alloc_ptr)
    }

    /// Inserts `value` at the root node at `coords`. Returns the old value.
    #[inline]
    pub fn insert_root(&mut self, coordinates: V, new_value: T) -> (NodePtr, Option<T>) {
        let level = self.root_level();
        let Self {
            root_nodes,
            allocators,
            ..
        } = self;
        let mut old_value = None;
        let alloc = &mut allocators[level as usize];
        let root_ptr = match root_nodes.entry(coordinates) {
            hash_map::Entry::Occupied(occupied) => {
                let root_ptr = *occupied.get();
                let current_value = unsafe { alloc.get_value_unchecked_mut(root_ptr) };
                old_value = Some(mem::replace(current_value, new_value));
                root_ptr
            }
            hash_map::Entry::Vacant(vacant) => {
                let (root_ptr, _children) = alloc.insert_branch(new_value);
                vacant.insert(root_ptr);
                root_ptr
            }
        };
        (
            NodePtr {
                level,
                alloc_ptr: root_ptr,
            },
            old_value,
        )
    }

    /// Inserts a child node of `parent_ptr` storing `child_value`. Returns the old child value if one exists.
    #[inline]
    pub fn insert_child(
        &mut self,
        parent_ptr: NodePtr,
        child_index: ChildIndex,
        child_value: T,
    ) -> (NodePtr, Option<T>) {
        assert!(parent_ptr.level > 0);
        let child_level = parent_ptr.level - 1;
        let (left, right) = self.allocators.split_at_mut(parent_ptr.level as usize);
        let child_alloc = left.last_mut().unwrap();
        let parent_alloc = right.first_mut().unwrap();

        let mut old_value = None;
        if let Some(children) = parent_alloc.get_children_mut(parent_ptr.alloc_ptr) {
            let child_ptr = &mut children[child_index as usize];
            if *child_ptr == EMPTY_PTR {
                if child_level > 0 {
                    let (new_child_ptr, _) = child_alloc.insert_branch(child_value);
                    *child_ptr = new_child_ptr;
                } else {
                    *child_ptr = child_alloc.insert_leaf(child_value);
                }
            } else {
                let current_value = unsafe { child_alloc.get_value_unchecked_mut(*child_ptr) };
                old_value = Some(mem::replace(current_value, child_value));
            }
            (
                NodePtr {
                    level: child_level,
                    alloc_ptr: *child_ptr,
                },
                old_value,
            )
        } else {
            panic!(
                "Tried inserting child of {:?} which has no child pointers",
                parent_ptr
            );
        }
    }

    /// Same as `insert_child` but `child_offset` is linearized into a [`ChildIndex`] based on the [`BranchShape`].
    ///
    /// This means any given coordinate in `child_offset` can only be 0 or 1!
    #[inline]
    pub fn insert_child_at_offset(
        &mut self,
        parent_ptr: NodePtr,
        child_offset: V,
        child_value: T,
    ) -> (NodePtr, Option<T>) {
        self.insert_child(parent_ptr, S::linearize_child(child_offset), child_value)
    }

    /// Looks up the root pointer for `coords` in the top-level hash map.
    #[inline]
    pub fn find_root(&self, coordinates: V) -> Option<NodePtr> {
        self.root_nodes.get(&coordinates).map(|&ptr| NodePtr {
            level: self.root_level(),
            alloc_ptr: ptr,
        })
    }

    /// Starting from the ancestor root, searches for the corresponding descendant node at `key`.
    ///
    /// A [`ChildRelation`] is returned because it contains some extra useful info that is conveniently accessible during the
    /// search.
    #[inline]
    pub fn find_node(&self, key: NodeKey<V>) -> Option<ChildRelation<V>> {
        if key.level == self.root_level() {
            self.find_root(key.coordinates)
                .map(|root_ptr| ChildRelation {
                    child: root_ptr,
                    parent: None,
                })
        } else {
            let root_coords = self.ancestor_root_key(key).coordinates;
            self.find_root(root_coords)
                .and_then(|root_ptr| self.find_descendant(root_ptr, root_coords, key))
        }
    }

    /// Starting from the node at `ancestor_ptr`, searches for the corresponding descendant node at `descendant_key`.
    pub fn find_descendant(
        &self,
        ancestor_ptr: NodePtr,
        ancestor_coordinates: V,
        descendant_key: NodeKey<V>,
    ) -> Option<ChildRelation<V>> {
        assert!(ancestor_ptr.level > descendant_key.level);
        let level_diff = ancestor_ptr.level - descendant_key.level;

        self.child_pointers(ancestor_ptr).and_then(|children| {
            let child_coords = if level_diff == 1 {
                descendant_key.coordinates
            } else {
                S::ancestor_key(descendant_key.coordinates, level_diff as u32 - 1)
            };
            let min_sibling_coords = S::min_child_key(ancestor_coordinates);
            let offset = child_coords - min_sibling_coords;
            let child_index = S::linearize_child(offset);
            let child_ptr = children.pointers[child_index as usize];

            (child_ptr != EMPTY_PTR)
                .then(|| child_ptr)
                .and_then(|child_ptr| {
                    let child_ptr = NodePtr {
                        level: children.level,
                        alloc_ptr: child_ptr,
                    };
                    if level_diff == 1 {
                        // Ancestor is the parent.
                        Some(ChildRelation {
                            child: child_ptr,
                            parent: Some(Parent {
                                ptr: ancestor_ptr,
                                coordinates: ancestor_coordinates,
                                child_index,
                            }),
                        })
                    } else {
                        self.find_descendant(child_ptr, child_coords, descendant_key)
                    }
                })
        })
    }

    /// Visits pointers for all non-empty children of the node at `parent_ptr`.
    ///
    /// If `parent_ptr` does not exist or does not have any children, nothing happens.
    #[inline]
    pub fn visit_children(
        &self,
        parent_ptr: NodePtr,
        mut visitor: impl FnMut(ChildIndex, NodePtr),
    ) {
        if let Some(children) = self.child_pointers(parent_ptr) {
            for (child_index, &child_ptr) in children.pointers.iter().enumerate() {
                if child_ptr != EMPTY_PTR {
                    visitor(child_index as ChildIndex, NodePtr { level: children.level, alloc_ptr: child_ptr });
                }
            }
        }
    }

    /// Visits pointers and coordinates for all non-empty children of the node at `parent_ptr` with coordinates `parent_coords`.
    ///
    /// If `parent_ptr` does not exist or does not have any children, nothing happens.
    #[inline]
    pub fn visit_children_with_coordinates(
        &self,
        parent_ptr: NodePtr,
        parent_coordinates: V,
        mut visitor: impl FnMut(NodePtr, V),
    ) {
        self.visit_children(parent_ptr, |child_index, child_ptr| {
            let child_offset = S::delinearize_child(child_index as ChildIndex);
            let child_coords = S::min_child_key(parent_coordinates) + child_offset;
            visitor(child_ptr, child_coords)
        })
    }

    /// Visit `ancestor_ptr` and all descendants in depth-first order.
    ///
    /// If `visitor` returns `false`, descendants of that node will not be visited.
    #[inline]
    pub fn visit_tree_depth_first(
        &self,
        ancestor_ptr: NodePtr,
        ancestor_coordinates: V,
        mut visitor: impl FnMut(NodePtr, V) -> bool,
    ) {
        let mut stack = SmallVec::<[(NodePtr, V); 32]>::new();
        stack.push((ancestor_ptr, ancestor_coordinates));
        while let Some((parent_ptr, parent_coords)) = stack.pop() {
            let keep_going = visitor(parent_ptr, parent_coords);
            if keep_going {
                self.visit_children_with_coordinates(
                    parent_ptr,
                    parent_coords,
                    |child_ptr, child_coords| {
                        stack.push((child_ptr, child_coords));
                    },
                );
            }
        }
    }

    /// Visit `ancestor_ptr` and all descendants in breadth-first order.
    ///
    /// If `visitor` returns `false`, descendants of that node will not be visited.
    #[inline]
    pub fn visit_tree_breadth_first(
        &self,
        ancestor_ptr: NodePtr,
        ancestor_coordinates: V,
        mut visitor: impl FnMut(NodePtr, V) -> bool,
    ) {
        let mut queue = VecDeque::new();
        queue.push_back((ancestor_ptr, ancestor_coordinates));
        while let Some((parent_ptr, parent_coords)) = queue.pop_front() {
            let keep_going = visitor(parent_ptr, parent_coords);
            if keep_going {
                self.visit_children_with_coordinates(
                    parent_ptr,
                    parent_coords,
                    |child_ptr, child_coords| {
                        queue.push_back((child_ptr, child_coords));
                    },
                );
            }
        }
    }

    /// Returns an array of pointers to the children of `parent_ptr`.
    ///
    /// Returns `None` if `parent_ptr` is at level 0. Otherwise, only the occupied children will have `NodePtr::is_null` be
    /// `false`.
    #[inline]
    pub fn child_pointers(&self, parent_ptr: NodePtr) -> Option<ChildPointers<'_, CHILDREN>> {
        self.allocator(parent_ptr.level)
            .get_children(parent_ptr.alloc_ptr)
            .map(|children| {
                ChildPointers {
                    level: parent_ptr.level - 1,
                    pointers: children,
                }
            })
    }

    /// Drops the child node of `relation` and all descendants.
    #[inline]
    pub fn drop_tree(&mut self, relation: &ChildRelation<V>) {
        self.unlink_child(relation);

        // Drop node and all descendants.
        let mut to_drop = SmallVec::<[NodePtr; 32]>::new();
        to_drop.push(relation.child);
        while let Some(ptr) = to_drop.pop() {
            let (_value, children) = self.allocator_mut(ptr.level).remove(ptr.alloc_ptr);
            if let Some(children) = children {
                let child_level = ptr.level - 1;
                for child_ptr in children.into_iter() {
                    if child_ptr != EMPTY_PTR {
                        to_drop.push(NodePtr {
                            level: child_level,
                            alloc_ptr: child_ptr,
                        });
                    }
                }
            }
        }
    }

    /// Moves the node at `coordinates` and all descendants into `consumer`.
    #[inline]
    pub fn remove_tree(
        &mut self,
        relation: &ChildRelation<V>,
        coordinates: V,
        mut consumer: impl FnMut(NodeKey<V>, T),
    ) {
        self.unlink_child(relation);

        let mut to_remove = SmallVec::<[(NodePtr, V); 32]>::new();
        to_remove.push((relation.child, coordinates));
        while let Some((ptr, coordinates)) = to_remove.pop() {
            let (value, children) = self.allocator_mut(ptr.level).remove(ptr.alloc_ptr);
            if let Some(value) = value {
                let node_key = NodeKey {
                    level: ptr.level,
                    coordinates,
                };
                consumer(node_key, value);
                if let Some(children) = children {
                    let child_level = ptr.level - 1;
                    let min_child = S::min_child_key(coordinates);
                    for (child_index, child_ptr) in children.into_iter().enumerate() {
                        if child_ptr != EMPTY_PTR {
                            let child_coords =
                                min_child + S::delinearize_child(child_index as ChildIndex);
                            to_remove.push((
                                NodePtr {
                                    level: child_level,
                                    alloc_ptr: child_ptr,
                                },
                                child_coords,
                            ));
                        }
                    }
                }
            }
        }
    }

    fn unlink_child(&mut self, relation: &ChildRelation<V>) {
        if let Some(parent) = relation.parent.as_ref() {
            if parent.ptr.level == self.root_level() {
                self.root_nodes.remove(&parent.coordinates);
            }
            self.allocator_mut(parent.ptr.level)
                .unlink_child(parent.ptr.alloc_ptr, parent.child_index);
        }
    }

    fn allocator(&self, level: Level) -> &NodeAllocator<T, CHILDREN> {
        &self.allocators[level as usize]
    }

    fn allocator_mut(&mut self, level: Level) -> &mut NodeAllocator<T, CHILDREN> {
        &mut self.allocators[level as usize]
    }
}

// ████████╗███████╗███████╗████████╗
// ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
//    ██║   █████╗  ███████╗   ██║
//    ██║   ██╔══╝  ╚════██║   ██║
//    ██║   ███████╗███████║   ██║
//    ╚═╝   ╚══════╝╚══════╝   ╚═╝

#[cfg(test)]
mod test {
    use super::*;

    use crate::glam::IVec3;
    use crate::OctreeI32;

    #[test]
    fn insert_root() {
        let mut tree = OctreeI32::new(3);

        let key = IVec3::ZERO;

        assert_eq!(tree.find_root(key), None);

        let (ptr, old_value) = tree.insert_root(key, "val1");
        assert_eq!(old_value, None);
        assert!(tree.contains_node(ptr));
        assert_eq!(tree.get_value(ptr), Some(&"val1"));
        assert_eq!(tree.find_root(key), Some(ptr));

        let (ptr, old_value) = tree.insert_root(key, "val2");
        assert_eq!(old_value, Some("val1"));
        assert!(tree.contains_node(ptr));
        assert_eq!(tree.get_value(ptr), Some(&"val2"));
    }

    #[test]
    fn insert_child_of_root() {
        let mut tree = OctreeI32::new(3);

        let root_key = IVec3::ZERO;
        let (root_ptr, _) = tree.insert_root(root_key, "val1");
        let (child_ptr, old_val) = tree.insert_child(root_ptr, 0, "val2");
        assert_eq!(old_val, None);
        assert!(tree.contains_node(child_ptr));
        assert_eq!(tree.get_value(child_ptr), Some(&"val2"));
    }

    #[test]
    fn find_descendant_none() {
        let mut tree = OctreeI32::new(3);

        let root_coords = IVec3::ZERO;
        let (root_ptr, _) = tree.insert_root(root_coords, ());

        let descendant_key = NodeKey {
            level: 1,
            coordinates: IVec3::ZERO,
        };
        let found = tree.find_descendant(root_ptr, root_coords, descendant_key);
        assert_eq!(found, None);
        let found = tree.find_node(descendant_key);
        assert_eq!(found, None);
    }

    #[test]
    fn find_descendant_child() {
        let mut tree = OctreeI32::new(3);

        let root_coords = IVec3::ZERO;
        let (root_ptr, _) = tree.insert_root(root_coords, ());

        let child_key = NodeKey {
            level: 1,
            coordinates: IVec3::new(1, 1, 1),
        };
        let (child_ptr, _) = tree.insert_child_at_offset(root_ptr, child_key.coordinates, ());

        let expected_find = Some(ChildRelation {
            child: child_ptr,
            parent: Some(Parent {
                ptr: root_ptr,
                coordinates: IVec3::ZERO,
                child_index: 7,
            }),
        });
        let found = tree.find_descendant(root_ptr, root_coords, child_key);
        assert_eq!(found, expected_find);
        let found = tree.find_node(child_key);
        assert_eq!(found, expected_find);
    }

    #[test]
    fn find_descendant_grandchild() {
        let mut tree = OctreeI32::new(3);

        let root_coords = IVec3::ZERO;
        let (root_ptr, _) = tree.insert_root(root_coords, ());

        let grandchild_key = NodeKey {
            level: 0,
            coordinates: IVec3::new(3, 3, 3),
        };
        let (child_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild_ptr, _) = tree.insert_child_at_offset(child_ptr, IVec3::new(1, 1, 1), ());

        let expected_find = Some(ChildRelation {
            child: grandchild_ptr,
            parent: Some(Parent {
                ptr: child_ptr,
                coordinates: IVec3::new(1, 1, 1),
                child_index: 7,
            }),
        });
        let found = tree.find_descendant(root_ptr, root_coords, grandchild_key);
        assert_eq!(found, expected_find);
        let found = tree.find_node(grandchild_key);
        assert_eq!(found, expected_find);
    }

    #[test]
    fn visit_children_of_root() {
        let mut tree = OctreeI32::new(3);

        let root_coords = IVec3::new(1, 1, 1);
        let (root_ptr, _) = tree.insert_root(root_coords, ());
        let (child1_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(0, 0, 0), ());
        let (child2_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(1, 1, 1), ());

        let mut visited = Vec::new();
        tree.visit_children_with_coordinates(root_ptr, root_coords, |child_ptr, child_coords| {
            visited.push((child_coords, child_ptr));
        });

        assert_eq!(
            visited.as_slice(),
            &[
                (IVec3::new(2, 2, 2), child1_ptr),
                (IVec3::new(3, 3, 3), child2_ptr)
            ]
        );
    }

    #[test]
    fn visit_tree_of_root() {
        let mut tree = OctreeI32::new(3);

        let root_coords = IVec3::new(1, 1, 1);
        let (root_ptr, _) = tree.insert_root(root_coords, ());
        let (child1_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(0, 0, 0), ());
        let (child2_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild1_ptr, _) = tree.insert_child_at_offset(child1_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild2_ptr, _) = tree.insert_child_at_offset(child2_ptr, IVec3::new(0, 0, 0), ());

        let mut visited = Vec::new();
        tree.visit_tree_depth_first(root_ptr, root_coords, |child_ptr, child_coords| {
            visited.push((child_coords, child_ptr));
            true
        });
        assert_eq!(
            visited.as_slice(),
            &[
                (IVec3::new(1, 1, 1), root_ptr),
                (IVec3::new(3, 3, 3), child2_ptr),
                (IVec3::new(6, 6, 6), grandchild2_ptr),
                (IVec3::new(2, 2, 2), child1_ptr),
                (IVec3::new(5, 5, 5), grandchild1_ptr),
            ]
        );

        let mut visited = Vec::new();
        tree.visit_tree_breadth_first(root_ptr, root_coords, |child_ptr, child_coords| {
            visited.push((child_coords, child_ptr));
            true
        });
        assert_eq!(
            visited.as_slice(),
            &[
                (IVec3::new(1, 1, 1), root_ptr),
                (IVec3::new(2, 2, 2), child1_ptr),
                (IVec3::new(3, 3, 3), child2_ptr),
                (IVec3::new(5, 5, 5), grandchild1_ptr),
                (IVec3::new(6, 6, 6), grandchild2_ptr),
            ]
        );
    }

    #[test]
    fn drop_tree() {
        let mut tree = OctreeI32::new(3);

        let root_coords = IVec3::new(1, 1, 1);
        let (root_ptr, _) = tree.insert_root(root_coords, ());
        let (child1_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(0, 0, 0), ());
        let (child2_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild1_ptr, _) = tree.insert_child_at_offset(child1_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild2_ptr, _) = tree.insert_child_at_offset(child2_ptr, IVec3::new(0, 0, 0), ());

        let child1_relation = tree
            .find_node(NodeKey {
                level: 1,
                coordinates: IVec3::new(2, 2, 2),
            })
            .unwrap();
        tree.drop_tree(&child1_relation);

        assert!(!tree.contains_node(child1_ptr));
        assert!(!tree.contains_node(grandchild1_ptr));

        let mut visited = Vec::new();
        tree.visit_tree_breadth_first(root_ptr, root_coords, |ptr, coords| {
            visited.push((ptr, coords));
            true
        });

        assert_eq!(
            visited.as_slice(),
            &[
                (root_ptr, root_coords),
                (child2_ptr, IVec3::new(3, 3, 3)),
                (grandchild2_ptr, IVec3::new(6, 6, 6))
            ]
        );
    }

    #[test]
    fn remove_tree() {
        let mut tree = OctreeI32::new(3);

        let root_coords = IVec3::new(1, 1, 1);
        let (root_ptr, _) = tree.insert_root(root_coords, ());
        let (child1_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(0, 0, 0), ());
        let (child2_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild1_ptr, _) = tree.insert_child_at_offset(child1_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild2_ptr, _) = tree.insert_child_at_offset(child2_ptr, IVec3::new(0, 0, 0), ());

        let child1_relation = tree
            .find_node(NodeKey {
                level: 1,
                coordinates: IVec3::new(2, 2, 2),
            })
            .unwrap();
        let mut removed = Vec::new();
        let child1_coords = IVec3::new(2, 2, 2);
        tree.remove_tree(&child1_relation, child1_coords, |key, value| {
            removed.push((key, value))
        });

        assert_eq!(
            removed.as_slice(),
            &[
                (
                    NodeKey {
                        level: 1,
                        coordinates: IVec3::new(2, 2, 2)
                    },
                    ()
                ),
                (
                    NodeKey {
                        level: 0,
                        coordinates: IVec3::new(5, 5, 5)
                    },
                    ()
                ),
            ]
        );

        assert!(!tree.contains_node(child1_ptr));
        assert!(!tree.contains_node(grandchild1_ptr));

        let mut visited = Vec::new();
        tree.visit_tree_breadth_first(root_ptr, root_coords, |ptr, coords| {
            visited.push((ptr, coords));
            true
        });

        assert_eq!(
            visited.as_slice(),
            &[
                (root_ptr, root_coords),
                (child2_ptr, IVec3::new(3, 3, 3)),
                (grandchild2_ptr, IVec3::new(6, 6, 6))
            ]
        );
    }
}
