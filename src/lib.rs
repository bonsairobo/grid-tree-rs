//! Voxel octrees and quadtrees.
//!
//! Store any type in an [`OctreeI32`](crate::OctreeI32), [`OctreeU32`](crate::OctreeU32), [`QuadtreeI32`](crate::QuadtreeI32),
//! or [`QuadtreeU32`](crate::QuadtreeU32), all of which are specific instances of the generic [`Tree`](crate::Tree). A
//! [`Tree`](crate::Tree) represents a map from `(Level, Integer Coordinates)` to `T`. Thus it is useful for storing pixel or
//! voxel data with level-of-detail.
//!
//! # Performance
//!
//! This structure is optimized for iteration speed and spatial queries like raycasting. Finding a single node by
//! [`NodeKey`](crate::NodeKey) starting from the root should be minimized as much as possible, so you might find it useful to
//! cache [`NodePtr`](crate::NodePtr)s or amortize the search with a full tree traversal. Memory usage is decent given the
//! simplicity of the implementation, and the pointer overhead is easily amortized by using dense chunk values.
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
