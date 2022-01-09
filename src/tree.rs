use crate::allocator::{AllocPtr, NodeAllocator, EMPTY_ALLOC_PTR};
use crate::{BranchShape, ChildIndex, Level, SmallKeyHashMap, VectorKey};

use smallvec::SmallVec;
use std::collections::{hash_map, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;

/// Uniquely identifies a node slot in the [`Tree`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NodeKey<V> {
    pub level: Level,
    pub coordinates: V,
}

impl<V> NodeKey<V> {
    #[inline]
    pub fn new(level: Level, coordinates: V) -> Self {
        Self { level, coordinates }
    }
}

/// Uniquely and stably identifies an occupied node in the [`Tree`] (until the node is removed).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodePtr {
    pub(crate) level: Level,
    pub(crate) alloc_ptr: AllocPtr,
}

impl NodePtr {
    pub const NULL: Self = NodePtr {
        level: 0,
        alloc_ptr: EMPTY_ALLOC_PTR,
    };

    pub fn new(level: Level, alloc_ptr: AllocPtr) -> Self {
        Self { level, alloc_ptr }
    }

    #[inline]
    pub fn alloc_ptr(&self) -> AllocPtr {
        self.alloc_ptr
    }

    #[inline]
    pub fn level(&self) -> Level {
        self.level
    }

    /// Null pointers can only be gotten by manually calling `Tree::child_pointers`.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.alloc_ptr == EMPTY_ALLOC_PTR
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

/// All children pointers for some branch node. Some may be empty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildPointers<'a, const CHILDREN: usize> {
    level: Level,
    pointers: &'a [AllocPtr; CHILDREN],
}

impl<'a, const CHILDREN: usize> ChildPointers<'a, CHILDREN> {
    /// Returns the child pointer at the given `child` linear index.
    #[inline]
    pub fn get_child(&self, child: ChildIndex) -> Option<NodePtr> {
        let alloc_ptr = self.pointers[child as usize];
        (alloc_ptr != EMPTY_ALLOC_PTR).then(|| NodePtr::new(self.level, alloc_ptr))
    }

    #[inline]
    pub fn level(&self) -> Level {
        self.level
    }
}

pub enum NodeEntry<'a, T, const CHILDREN: usize> {
    Occupied(OccupiedNodeEntry<'a, T, CHILDREN>),
    Vacant(VacantNodeEntry<'a, T, CHILDREN>),
}

impl<'a, T, const CHILDREN: usize> NodeEntry<'a, T, CHILDREN> {
    pub fn or_insert_with(&mut self, mut filler: impl FnMut() -> T) -> (AllocPtr, &mut T) {
        match self {
            Self::Occupied(o) => (o.ptr, o.get_mut()),
            Self::Vacant(v) => {
                let ptr = v.insert(filler());
                (ptr, unsafe { v.alloc.get_value_unchecked_mut(ptr) })
            }
        }
    }

    pub fn pointer(&self) -> AllocPtr {
        // PERF: this extra branch seems unnecessary when both variants have a pointer
        match self {
            Self::Occupied(o) => o.ptr,
            Self::Vacant(v) => *v.ptr,
        }
    }
}

pub struct OccupiedNodeEntry<'a, T, const CHILDREN: usize> {
    alloc: &'a mut NodeAllocator<T, CHILDREN>,
    ptr: AllocPtr,
}

impl<'a, T, const CHILDREN: usize> OccupiedNodeEntry<'a, T, CHILDREN> {
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.alloc.get_value_unchecked_mut(self.ptr) }
    }
}

pub struct VacantNodeEntry<'a, T, const CHILDREN: usize> {
    alloc: &'a mut NodeAllocator<T, CHILDREN>,
    ptr: &'a mut AllocPtr,
    level: Level,
}

impl<'a, T, const CHILDREN: usize> VacantNodeEntry<'a, T, CHILDREN> {
    #[inline]
    pub fn insert(&mut self, value: T) -> AllocPtr {
        let new_ptr = if self.level == 0 {
            self.alloc.insert_leaf(value)
        } else {
            let (ptr, _child_pointers) = self.alloc.insert_branch(value);
            ptr
        };
        *self.ptr = new_ptr;
        new_ptr
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RootNode {
    pub self_ptr: AllocPtr,
    /// Roots may optionally point back to a node in some other [`Tree`]'s allocator.
    pub parent_ptr: Option<AllocPtr>,
}

impl RootNode {
    pub fn new(self_ptr: AllocPtr, parent_ptr: Option<AllocPtr>) -> Self {
        Self {
            self_ptr,
            parent_ptr,
        }
    }

    pub fn new_without_parent(self_ptr: AllocPtr) -> Self {
        Self::new(self_ptr, None)
    }
}

/// A generic "grid tree" which can be either a quadtree or an octree depending on the type parameters.
#[derive(Clone, Debug)]
pub struct Tree<V, S, T, const CHILDREN: usize> {
    /// 2x2 square in 2D or 2x2x2 cube in 3D.
    branch_shape: PhantomData<S>,
    /// Every node at the highest LOD is a root.
    root_nodes: SmallKeyHashMap<NodeKey<V>, RootNode>,
    /// An allocator for each level.
    allocators: Vec<NodeAllocator<T, CHILDREN>>,
}

impl<V, S, T, const CHILDREN: usize> Tree<V, S, T, CHILDREN>
where
    V: VectorKey,
    NodeKey<V>: Hash,
    S: BranchShape<V>,
{
    /// The maximum number of children a branch node can have. 4 for a quadtree and 8 for an octree.
    pub const CHILDREN: ChildIndex = CHILDREN as ChildIndex;

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

    /// Iterate over all root keys.
    pub fn iter_root_keys(&self) -> impl Iterator<Item = &NodeKey<V>> {
        self.root_nodes.keys()
    }

    /// Iterate over all root nodes.
    pub fn iter_roots(&self) -> impl Iterator<Item = (&NodeKey<V>, &RootNode)> {
        self.root_nodes.iter()
    }

    /// Returns true iff this tree contains a node for `ptr`.
    #[inline]
    pub fn contains_node(&self, ptr: NodePtr) -> bool {
        self.allocator(ptr.level).contains_node(ptr.alloc_ptr)
    }

    /// Returns true iff this tree contains a root node at `coords`.
    #[inline]
    pub fn contains_root(&self, key: NodeKey<V>) -> bool {
        self.root_nodes.contains_key(&key)
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

    /// Inserts `value` at the root node at `key`. Returns the old value.
    #[inline]
    pub fn insert_root(
        &mut self,
        key: NodeKey<V>,
        parent_ptr: Option<AllocPtr>,
        new_value: T,
    ) -> (RootNode, Option<T>) {
        let Self {
            root_nodes,
            allocators,
            ..
        } = self;
        let mut old_value = None;
        let alloc = &mut allocators[key.level as usize];
        let root_node = match root_nodes.entry(key) {
            hash_map::Entry::Occupied(occupied) => {
                let root_node = *occupied.get();
                let current_value = unsafe { alloc.get_value_unchecked_mut(root_node.self_ptr) };
                old_value = Some(mem::replace(current_value, new_value));
                root_node
            }
            hash_map::Entry::Vacant(vacant) => {
                let (root_ptr, _children) = alloc.insert_branch(new_value);
                let node = RootNode::new(root_ptr, parent_ptr);
                vacant.insert(node);
                node
            }
        };
        (root_node, old_value)
    }

    /// Gets the root pointer or calls `filler` to insert a value first.
    #[inline]
    pub fn get_or_create_root(
        &mut self,
        key: NodeKey<V>,
        mut filler: impl FnMut() -> T,
    ) -> RootNode {
        let Self {
            root_nodes,
            allocators,
            ..
        } = self;
        let alloc = &mut allocators[key.level as usize];
        *root_nodes.entry(key).or_insert_with(|| {
            let (root_ptr, _children) = alloc.insert_branch(filler());
            RootNode::new_without_parent(root_ptr)
        })
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

        let [parent_alloc, child_alloc] =
            Self::parent_and_child_allocators_mut(&mut self.allocators, parent_ptr.level)
                .unwrap_or_else(|| {
                    panic!("Tried inserting child of invalid parent: {:?}", parent_ptr)
                });
        let child_level = parent_ptr.level - 1;

        let mut old_value = None;
        let children = parent_alloc.get_children_mut_or_panic(parent_ptr.alloc_ptr);
        let child_ptr = &mut children[child_index as usize];
        if *child_ptr == EMPTY_ALLOC_PTR {
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
        (NodePtr::new(child_level, *child_ptr), old_value)
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

    /// May return the [`RootNode`] for convenience.
    #[inline]
    pub fn fill_root(
        &mut self,
        key: NodeKey<V>,
        mut filler: impl FnMut(&mut NodeEntry<'_, T, CHILDREN>) -> VisitCommand,
    ) -> (Option<RootNode>, VisitCommand) {
        let Self {
            allocators,
            root_nodes,
            ..
        } = self;
        let root_alloc = &mut allocators[key.level as usize];

        match root_nodes.entry(key) {
            hash_map::Entry::Occupied(occupied_node) => {
                let root_node = *occupied_node.get();
                let mut entry = NodeEntry::Occupied(OccupiedNodeEntry {
                    alloc: root_alloc,
                    ptr: root_node.self_ptr,
                });
                (Some(root_node), filler(&mut entry))
            }
            hash_map::Entry::Vacant(vacant_node) => {
                let mut new_ptr = EMPTY_ALLOC_PTR;
                let mut entry = NodeEntry::Vacant(VacantNodeEntry {
                    alloc: root_alloc,
                    ptr: &mut new_ptr,
                    level: key.level,
                });
                let command = filler(&mut entry);
                if new_ptr == EMPTY_ALLOC_PTR {
                    (None, command)
                } else {
                    let root_node = RootNode::new_without_parent(new_ptr);
                    vacant_node.insert(root_node);
                    (Some(root_node), command)
                }
            }
        }
    }

    /// # Panics
    ///
    /// - If `parent_ptr` is at level 0 and hence cannot have children.
    /// - If no node exists for `parent_ptr`.
    #[inline]
    pub fn child_entry(
        &mut self,
        parent_ptr: NodePtr,
        child_index: ChildIndex,
    ) -> NodeEntry<'_, T, CHILDREN> {
        let [parent_alloc, child_alloc] =
            Self::parent_and_child_allocators_mut(&mut self.allocators, parent_ptr.level)
                .unwrap_or_else(|| {
                    panic!("Tried getting child of invalid parent: {:?}", parent_ptr)
                });

        let children = parent_alloc.get_children_mut_or_panic(parent_ptr.alloc_ptr);
        let child_ptr = &mut children[child_index as usize];
        if *child_ptr == EMPTY_ALLOC_PTR {
            NodeEntry::Vacant(VacantNodeEntry {
                alloc: child_alloc,
                ptr: child_ptr,
                level: parent_ptr.level - 1,
            })
        } else {
            NodeEntry::Occupied(OccupiedNodeEntry {
                alloc: child_alloc,
                ptr: *child_ptr,
            })
        }
    }

    /// Inserts the data from `filler()` in every descendant "slot" of `ancestor_ptr` where `filler` returns `Some`.
    ///
    /// Any node N is skipped if `predicate` returns false for any ancestor of N.
    #[inline]
    pub fn fill_descendants(
        &mut self,
        ancestor_ptr: NodePtr,
        ancestor_coordinates: V,
        min_level: Level,
        mut filler: impl FnMut(NodeKey<V>, &mut NodeEntry<'_, T, CHILDREN>) -> VisitCommand,
    ) {
        assert!(min_level < ancestor_ptr.level());
        let mut stack = SmallVec::<[(NodePtr, V); 32]>::new();
        stack.push((ancestor_ptr, ancestor_coordinates));
        while let Some((parent_ptr, parent_coords)) = stack.pop() {
            if parent_ptr.level > 0 {
                let has_grandchildren = parent_ptr.level > min_level + 1;
                let child_level = parent_ptr.level - 1;
                for child_index in 0..Self::CHILDREN {
                    let mut child_entry = self.child_entry(parent_ptr, child_index);
                    let child_coords = parent_coords + S::delinearize_child(child_index);
                    let child_key = NodeKey::new(child_level, child_coords);
                    let command = filler(child_key, &mut child_entry);
                    if let VisitCommand::Continue = command {
                        if has_grandchildren {
                            let child_ptr = child_entry.pointer();
                            if child_ptr != EMPTY_ALLOC_PTR {
                                stack.push((NodePtr::new(child_level, child_ptr), child_coords));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Call `filler` on all nodes from the root ancestor to `target_key`.
    pub fn fill_path_to_node(
        &mut self,
        target_key: NodeKey<V>,
        mut filler: impl FnMut(NodeKey<V>, &mut NodeEntry<'_, T, CHILDREN>) -> VisitCommand,
    ) {
        // We need to start from the top to avoid allocating nodes that already exist.
        let root_key = self.ancestor_root_key(target_key);
        let (root_node, command) =
            self.fill_root(root_key, |root_entry| filler(root_key, root_entry));

        if let (Some(root_node), VisitCommand::Continue) = (root_node, command) {
            let mut parent_ptr = NodePtr::new(root_key.level, root_node.self_ptr);
            let mut parent_coords = root_key.coordinates;
            for child_level in (target_key.level..root_key.level).rev() {
                // Get the child index of the ancestor at this level.
                let level_diff = child_level - target_key.level;
                let ancestor_coords = S::ancestor_key(target_key.coordinates, level_diff as u32);
                let child_index =
                    S::linearize_child(ancestor_coords - S::min_child_key(parent_coords));
                let child_coords = parent_coords + S::delinearize_child(child_index);
                let node_key = NodeKey::new(child_level, child_coords);
                let mut child_entry = self.child_entry(parent_ptr, child_index);
                let command = filler(node_key, &mut child_entry);
                if command == VisitCommand::SkipDescendants {
                    break;
                }
                let child_ptr = child_entry.pointer();
                if child_ptr == EMPTY_ALLOC_PTR {
                    break;
                }
                parent_ptr = NodePtr::new(child_level, child_ptr);
                parent_coords = ancestor_coords;
            }
        }
    }

    /// Looks up the root pointer for `key` in the top-level hash map.
    #[inline]
    pub fn find_root(&self, key: NodeKey<V>) -> Option<RootNode> {
        self.root_nodes.get(&key).cloned()
    }

    /// Starting from the ancestor root, searches for the corresponding descendant node at `key`.
    ///
    /// A [`ChildRelation`] is returned because it contains some extra useful info that is conveniently accessible during the
    /// search.
    #[inline]
    pub fn find_node(&self, key: NodeKey<V>) -> Option<ChildRelation<V>> {
        if key.level == self.root_level() {
            self.find_root(key).map(|root_node| ChildRelation {
                child: NodePtr::new(key.level, root_node.self_ptr),
                parent: None,
            })
        } else {
            let root_key = self.ancestor_root_key(key);
            self.find_root(root_key).and_then(|root_node| {
                self.find_descendant(
                    NodePtr::new(root_key.level, root_node.self_ptr),
                    root_key.coordinates,
                    key,
                )
            })
        }
    }

    /// Starting from the node at `ancestor_ptr`, searches for the corresponding descendant node at `descendant_key`.
    pub fn find_descendant(
        &self,
        ancestor_ptr: NodePtr,
        ancestor_coordinates: V,
        descendant_key: NodeKey<V>,
    ) -> Option<ChildRelation<V>> {
        assert!(
            ancestor_ptr.level > descendant_key.level,
            "{} > {}",
            ancestor_ptr.level,
            descendant_key.level
        );
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

            (child_ptr != EMPTY_ALLOC_PTR)
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
        mut visitor: impl FnMut(NodePtr, ChildIndex),
    ) {
        if let Some(children) = self.child_pointers(parent_ptr) {
            for (child_index, &child_ptr) in children.pointers.iter().enumerate() {
                if child_ptr != EMPTY_ALLOC_PTR {
                    visitor(
                        NodePtr {
                            level: children.level,
                            alloc_ptr: child_ptr,
                        },
                        child_index as ChildIndex,
                    );
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
        self.visit_children(parent_ptr, |child_ptr, child_index| {
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
        min_level: Level,
        mut visitor: impl FnMut(NodePtr, V) -> VisitCommand,
    ) {
        let mut stack = SmallVec::<[(NodePtr, V); 32]>::new();
        stack.push((ancestor_ptr, ancestor_coordinates));
        while let Some((parent_ptr, parent_coords)) = stack.pop() {
            let command = visitor(parent_ptr, parent_coords);
            if let VisitCommand::Continue = command {
                if parent_ptr.level > min_level {
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
    }

    /// Visit `ancestor_ptr` and all descendants in breadth-first order.
    ///
    /// If `visitor` returns `false`, descendants of that node will not be visited.
    #[inline]
    pub fn visit_tree_breadth_first(
        &self,
        ancestor_ptr: NodePtr,
        ancestor_coordinates: V,
        min_level: Level,
        mut visitor: impl FnMut(NodePtr, V) -> VisitCommand,
    ) {
        let mut queue = VecDeque::new();
        queue.push_back((ancestor_ptr, ancestor_coordinates));
        while let Some((parent_ptr, parent_coords)) = queue.pop_front() {
            let command = visitor(parent_ptr, parent_coords);
            if command == VisitCommand::Continue {
                if parent_ptr.level > min_level {
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
    }

    /// Returns an array of pointers to the children of `parent_ptr`.
    ///
    /// Returns `None` if `parent_ptr` is at level 0.
    #[inline]
    pub fn child_pointers(&self, parent_ptr: NodePtr) -> Option<ChildPointers<'_, CHILDREN>> {
        self.allocator(parent_ptr.level)
            .get_children(parent_ptr.alloc_ptr)
            .map(|children| ChildPointers {
                level: parent_ptr.level - 1,
                pointers: children,
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
                    if child_ptr != EMPTY_ALLOC_PTR {
                        to_drop.push(NodePtr {
                            level: child_level,
                            alloc_ptr: child_ptr,
                        });
                    }
                }
            }
        }
    }

    /// Moves the child node of `relation` (with `coordinates`) and all descendants into `consumer`.
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
                        if child_ptr != EMPTY_ALLOC_PTR {
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
            self.root_nodes
                .remove(&NodeKey::new(relation.child.level + 1, parent.coordinates));
            self.allocator_mut(parent.ptr.level)
                .unlink_child(parent.ptr.alloc_ptr, parent.child_index);
        }
    }

    fn parent_and_child_allocators_mut(
        allocators: &mut [NodeAllocator<T, CHILDREN>],
        parent_level: Level,
    ) -> Option<[&mut NodeAllocator<T, CHILDREN>; 2]> {
        if parent_level == 0 {
            return None;
        }

        let (left, right) = allocators.split_at_mut(parent_level as usize);
        let child_alloc = left.last_mut().unwrap();
        let parent_alloc = right.first_mut().unwrap();

        Some([parent_alloc, child_alloc])
    }

    fn allocator(&self, level: Level) -> &NodeAllocator<T, CHILDREN> {
        &self.allocators[level as usize]
    }

    fn allocator_mut(&mut self, level: Level) -> &mut NodeAllocator<T, CHILDREN> {
        &mut self.allocators[level as usize]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisitCommand {
    Continue,
    SkipDescendants,
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

        let key = NodeKey::new(2, IVec3::ZERO);

        assert_eq!(tree.find_root(key), None);

        let (node, old_value) = tree.insert_root(key, None, "val1");
        assert_eq!(old_value, None);
        assert_eq!(node.parent_ptr, None);
        let root_ptr = NodePtr::new(2, node.self_ptr);
        assert!(tree.contains_node(root_ptr));
        assert_eq!(tree.get_value(root_ptr), Some(&"val1"));
        assert_eq!(tree.find_root(key), Some(node));

        let (_node, old_value) = tree.insert_root(key, None, "val2");
        assert_eq!(old_value, Some("val1"));
        assert!(tree.contains_node(root_ptr));
        assert_eq!(tree.get_value(root_ptr), Some(&"val2"));
    }

    #[test]
    fn get_or_create_root() {
        let mut tree = OctreeI32::new(3);

        let key = NodeKey::new(2, IVec3::ZERO);

        let ptr = tree.get_or_create_root(key, || ());
        assert_eq!(ptr, tree.get_or_create_root(key, || ()));

        let (ptr, _old_value) = tree.insert_root(key, None, ());
        assert_eq!(ptr, tree.get_or_create_root(key, || ()));
    }

    #[test]
    fn insert_child_of_root() {
        let mut tree = OctreeI32::new(3);

        let root_key = NodeKey::new(2, IVec3::ZERO);
        let (root_node, _) = tree.insert_root(root_key, None, "val1");
        let root_ptr = NodePtr::new(2, root_node.self_ptr);
        let (child_ptr, old_val) = tree.insert_child(root_ptr, 0, "val2");
        assert_eq!(old_val, None);
        assert!(tree.contains_node(child_ptr));
        assert_eq!(tree.get_value(child_ptr), Some(&"val2"));
    }

    #[test]
    fn find_descendant_none() {
        let mut tree = OctreeI32::new(3);

        let root_key = NodeKey::new(2, IVec3::ZERO);
        let (root_node, _) = tree.insert_root(root_key, None, ());
        let root_ptr = NodePtr::new(2, root_node.self_ptr);

        let descendant_key = NodeKey {
            level: 1,
            coordinates: IVec3::ZERO,
        };
        let found = tree.find_descendant(root_ptr, root_key.coordinates, descendant_key);
        assert_eq!(found, None);
        let found = tree.find_node(descendant_key);
        assert_eq!(found, None);
    }

    #[test]
    fn find_descendant_child() {
        let mut tree = OctreeI32::new(3);

        let root_key = NodeKey::new(2, IVec3::ZERO);
        let (root_node, _) = tree.insert_root(root_key, None, ());
        let root_ptr = NodePtr::new(2, root_node.self_ptr);

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
        let found = tree.find_descendant(root_ptr, root_key.coordinates, child_key);
        assert_eq!(found, expected_find);
        let found = tree.find_node(child_key);
        assert_eq!(found, expected_find);
    }

    #[test]
    fn find_descendant_grandchild() {
        let mut tree = OctreeI32::new(3);

        let root_key = NodeKey::new(2, IVec3::ZERO);
        let (root_node, _) = tree.insert_root(root_key, None, ());
        let root_ptr = NodePtr::new(2, root_node.self_ptr);

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
        let found = tree.find_descendant(root_ptr, root_key.coordinates, grandchild_key);
        assert_eq!(found, expected_find);
        let found = tree.find_node(grandchild_key);
        assert_eq!(found, expected_find);
    }

    #[test]
    fn visit_children_of_root() {
        let mut tree = OctreeI32::new(3);

        let root_key = NodeKey::new(2, IVec3::new(1, 1, 1));
        let (root_node, _) = tree.insert_root(root_key, None, ());
        let root_ptr = NodePtr::new(2, root_node.self_ptr);
        let (child1_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(0, 0, 0), ());
        let (child2_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(1, 1, 1), ());

        let mut visited = Vec::new();
        tree.visit_children_with_coordinates(
            root_ptr,
            root_key.coordinates,
            |child_ptr, child_coords| {
                visited.push((child_coords, child_ptr));
            },
        );

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

        let root_key = NodeKey::new(2, IVec3::new(1, 1, 1));
        let (root_node, _) = tree.insert_root(root_key, None, ());
        let root_ptr = NodePtr::new(2, root_node.self_ptr);
        let (child1_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(0, 0, 0), ());
        let (child2_ptr, _) = tree.insert_child_at_offset(root_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild1_ptr, _) = tree.insert_child_at_offset(child1_ptr, IVec3::new(1, 1, 1), ());
        let (grandchild2_ptr, _) = tree.insert_child_at_offset(child2_ptr, IVec3::new(0, 0, 0), ());

        let mut visited = Vec::new();
        tree.visit_tree_depth_first(
            root_ptr,
            root_key.coordinates,
            0,
            |child_ptr, child_coords| {
                visited.push((child_coords, child_ptr));
                VisitCommand::Continue
            },
        );
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
        tree.visit_tree_breadth_first(
            root_ptr,
            root_key.coordinates,
            0,
            |child_ptr, child_coords| {
                visited.push((child_coords, child_ptr));
                VisitCommand::Continue
            },
        );
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

        let root_key = NodeKey::new(2, IVec3::new(1, 1, 1));
        let (root_node, _) = tree.insert_root(root_key, None, ());
        let root_ptr = NodePtr::new(2, root_node.self_ptr);
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
        tree.visit_tree_breadth_first(root_ptr, root_key.coordinates, 0, |ptr, coords| {
            visited.push((ptr, coords));
            VisitCommand::Continue
        });

        assert_eq!(
            visited.as_slice(),
            &[
                (root_ptr, root_key.coordinates),
                (child2_ptr, IVec3::new(3, 3, 3)),
                (grandchild2_ptr, IVec3::new(6, 6, 6))
            ]
        );
    }

    #[test]
    fn remove_tree() {
        let mut tree = OctreeI32::new(3);

        let root_key = NodeKey::new(2, IVec3::new(1, 1, 1));
        let (root_node, _) = tree.insert_root(root_key, None, ());
        let root_ptr = NodePtr::new(2, root_node.self_ptr);
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
        tree.visit_tree_breadth_first(root_ptr, root_key.coordinates, 0, |ptr, coords| {
            visited.push((ptr, coords));
            VisitCommand::Continue
        });

        assert_eq!(
            visited.as_slice(),
            &[
                (root_ptr, root_key.coordinates),
                (child2_ptr, IVec3::new(3, 3, 3)),
                (grandchild2_ptr, IVec3::new(6, 6, 6))
            ]
        );
    }

    #[test]
    fn fill_some_descendants() {
        let mut tree = OctreeI32::new(3);

        let root_key = NodeKey::new(2, IVec3::new(1, 1, 1));
        let (root_node, _) = tree.insert_root(root_key, None, ());
        let root_ptr = NodePtr::new(2, root_node.self_ptr);
        tree.fill_descendants(root_ptr, root_key.coordinates, 0, |_child_coords, entry| {
            entry.or_insert_with(|| ());
            VisitCommand::Continue
        });

        let mut visited_lvl0 = SmallKeyHashMap::new();
        tree.visit_tree_depth_first(
            root_ptr,
            root_key.coordinates,
            0,
            |child_ptr, child_coords| {
                if child_ptr.level() == 0 && child_coords % 2 == IVec3::ZERO {
                    visited_lvl0.insert(child_coords, child_ptr);
                }
                VisitCommand::Continue
            },
        );

        let mut expected_lvl0 = SmallKeyHashMap::new();
        for z in 4..8 {
            for y in 4..8 {
                for x in 4..8 {
                    let p = IVec3::new(x, y, z);
                    if p % 2 == IVec3::ZERO {
                        let relation = tree
                            .find_node(NodeKey {
                                level: 0,
                                coordinates: p,
                            })
                            .unwrap();
                        expected_lvl0.insert(p, relation.child);
                    }
                }
            }
        }
        assert_eq!(visited_lvl0, expected_lvl0);
    }

    #[test]
    fn fill_root() {
        let mut tree = OctreeI32::new(5);

        let root_key = NodeKey::new(2, IVec3::new(1, 1, 1));
        let (root_node, _) = tree.fill_root(root_key, |entry| {
            match entry {
                NodeEntry::Occupied(_) => {
                    panic!("Unexpected occupied entry");
                }
                NodeEntry::Vacant(v) => {
                    v.insert(());
                }
            }
            VisitCommand::Continue
        });
        assert!(tree.contains_root(root_key));
        assert_eq!(tree.find_root(root_key).unwrap(), root_node.unwrap());
    }

    #[test]
    fn fill_path_to_node() {
        let mut tree = OctreeI32::new(5);

        let target_key = NodeKey::new(1, IVec3::new(1, 1, 1));
        let mut path = Vec::new();
        tree.fill_path_to_node(target_key, |key, entry| {
            match entry {
                NodeEntry::Occupied(_) => {
                    panic!("Unexpected occupied entry");
                }
                NodeEntry::Vacant(v) => {
                    let new_ptr = v.insert(());
                    path.push((NodePtr::new(key.level, new_ptr), key.coordinates));
                }
            }
            VisitCommand::Continue
        });

        assert_eq!(path.len(), 4);

        for (ptr, coords) in path.into_iter() {
            let key = NodeKey::new(ptr.level(), coords);
            assert!(tree.contains_node(ptr));
            let relation = tree.find_node(key).unwrap();
            assert_eq!(relation.child, ptr);
        }
    }
}
