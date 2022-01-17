//! [![Crates.io](https://img.shields.io/crates/v/grid-tree.svg)](https://crates.io/crates/grid-tree)
//! [![Docs.rs](https://docs.rs/grid-tree/badge.svg)](https://docs.rs/grid-tree)
//!
//! Pixel quadtrees and voxel octrees.
//!
//! Store any type in an [`OctreeI32`](crate::OctreeI32), [`OctreeU32`](crate::OctreeU32), [`QuadtreeI32`](crate::QuadtreeI32),
//! or [`QuadtreeU32`](crate::QuadtreeU32), all of which are specific instances of the generic [`Tree`](crate::Tree). A
//! [`Tree`](crate::Tree) represents a map from `(Level, Integer Coordinates)` to `T`. Thus it is useful for storing pixel or
//! voxel data with level-of-detail. The tree also requires that if a node slot is occupied (has data), then all ancestor slots
//! are also filled.
//!
//! # Design Advantages
//!
//! - Since a [`Tree`](crate::Tree) has its own internal allocators, any pointers are completely local to the data structure. In
//!   principle, this makes it easy to clone the tree for e.g. uploading to a GPU (although we haven't tried it for ourselves).
//! - The level 0 allocator does not store pointers, only values. Pointer overhead at higher levels can be amortized using
//!   chunked data, i.e. `[T; CHUNK_SIZE]`. The alternative "pointerless" octrees take up less memory, but are also more complex
//!   to edit and traverse.
//! - By using a hash map of root nodes, the addressable space is not limited by the height of the tree, and it is not necessary
//!   to "translate" the octree as it follows a focal point.
//! - By having a very simple data layout, access using a [`NodePtr`](crate::NodePtr) is simply an array lookup.
//! - The [`NodeEntry`](crate::NodeEntry) and `Tree::child_entry` APIs allow for very simple code that fills entire trees with a
//!   single visitor closure.
//! - By implementing [`VectorKey`](crate::VectorKey) for a custom key type, the addressable range can be extended to
//!   coordinates of arbitrary precision.
//!
//! # Performance
//!
//! This structure is optimized for iteration speed and spatial queries that benefit from a bounding volume hierarchy (like
//! raycasting). Finding a single node by [`NodeKey`](crate::NodeKey) starting from the root should be minimized as much as
//! possible, so you might find it useful to cache [`NodePtr`](crate::NodePtr)s or amortize the search with a full tree
//! traversal. Memory usage is decent given the simplicity of the implementation, and the pointer overhead is easily amortized
//! by using dense chunk values.
//!
//! - random access with [`NodeKey`](crate::NodeKey): O(depth)
//! - random access with [`NodePtr`](crate::NodePtr): O(1)
//! - iteration: O(nodes)
//! - memory usage per node:
//!   - **level 0**: `size_of::<T>()` bytes
//!   - **level N > 0**: `size_of::<T>() + CHILDREN * 4` bytes
//!   - *where* `CHILDREN=4` for a quadtree and `CHILDREN=8` for an octree

mod allocator;
mod shape;
mod tree;
mod vector_key;

pub use allocator::{AllocPtr, EMPTY_ALLOC_PTR};
pub use shape::*;
pub use tree::*;
pub use vector_key::*;

#[cfg(feature = "glam")]
mod impl_glam;

#[cfg(feature = "glam")]
pub use impl_glam::*;

#[cfg(feature = "glam")]
pub use glam;

/// A "level of detail" in a [`Tree`].
pub type Level = u8;

/// A linear index of a node relative to its parent.
pub type ChildIndex = u8;

use ahash::AHashMap;

type SmallKeyHashMap<K, V> = AHashMap<K, V>;
