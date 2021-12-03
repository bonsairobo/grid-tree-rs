use crate::{BranchShape, ChildIndex, Level, Tree, VectorKey};

use glam::{const_ivec2, const_ivec3, const_uvec2, const_uvec3, IVec2, IVec3, UVec2, UVec3};
use ndshape::{
    ConstPow2Shape2i32, ConstPow2Shape2u32, ConstPow2Shape3i32, ConstPow2Shape3u32, ConstShape,
};

/// A [`BranchShape`] for signed quadtrees.
pub type QuadtreeShapeI32 = ConstPow2Shape2i32<1, 1>;
/// A [`BranchShape`] for unsigned quadtrees.
pub type QuadtreeShapeU32 = ConstPow2Shape2u32<1, 1>;

/// A [`BranchShape`] for signed octrees.
pub type OctreeShapeI32 = ConstPow2Shape3i32<1, 1, 1>;
/// A [`BranchShape`] for unsigned octrees.
pub type OctreeShapeU32 = ConstPow2Shape3u32<1, 1, 1>;

/// The default quadtree with `i32` coordinates.
pub type QuadtreeI32<T> = Tree<IVec2, QuadtreeShapeI32, T, 4>;
/// The default quadtree with `u32` coordinates.
pub type QuadtreeU32<T> = Tree<UVec2, QuadtreeShapeU32, T, 4>;

/// The default octree with `i32` coordinates.
pub type OctreeI32<T> = Tree<IVec3, OctreeShapeI32, T, 8>;
/// The default octree with `u32` coordinates.
pub type OctreeU32<T> = Tree<UVec3, OctreeShapeU32, T, 8>;

impl<T> QuadtreeI32<T> {
    pub fn new(height: Level) -> Self {
        unsafe { Self::new_generic(height) }
    }
}
impl<T> QuadtreeU32<T> {
    pub fn new(height: Level) -> Self {
        unsafe { Self::new_generic(height) }
    }
}
impl<T> OctreeI32<T> {
    pub fn new(height: Level) -> Self {
        unsafe { Self::new_generic(height) }
    }
}
impl<T> OctreeU32<T> {
    pub fn new(height: Level) -> Self {
        unsafe { Self::new_generic(height) }
    }
}

macro_rules! impl_signed_branch_shape {
    ($name:ty, $vector:ty, $shifter:expr) => {
        impl BranchShape<$vector> for $name {
            const SHAPE_SHIFTER: $vector = $shifter;

            #[inline]
            fn linearize_child(v: $vector) -> ChildIndex {
                <$name>::linearize(v.into()) as ChildIndex
            }

            #[inline]
            fn delinearize_child(i: ChildIndex) -> $vector {
                <$name>::delinearize(i as i32).into()
            }
        }
    };
}

macro_rules! impl_unsigned_branch_shape {
    ($name:ty, $vector:ty, $shifter:expr) => {
        impl BranchShape<$vector> for $name {
            const SHAPE_SHIFTER: $vector = $shifter;

            #[inline]
            fn linearize_child(v: $vector) -> ChildIndex {
                <$name>::linearize(v.into()) as ChildIndex
            }

            #[inline]
            fn delinearize_child(i: ChildIndex) -> $vector {
                <$name>::delinearize(i as u32).into()
            }
        }
    };
}

impl_unsigned_branch_shape!(QuadtreeShapeU32, UVec2, const_uvec2!([1; 2]));
impl_unsigned_branch_shape!(OctreeShapeU32, UVec3, const_uvec3!([1; 3]));
impl_signed_branch_shape!(QuadtreeShapeI32, IVec2, const_ivec2!([1; 2]));
impl_signed_branch_shape!(OctreeShapeI32, IVec3, const_ivec3!([1; 3]));

impl VectorKey for IVec2 {
    #[inline]
    fn mul_u32(self, rhs: u32) -> Self {
        self * rhs as i32
    }
}
impl VectorKey for IVec3 {
    #[inline]
    fn mul_u32(self, rhs: u32) -> Self {
        self * rhs as i32
    }
}
impl VectorKey for UVec2 {
    #[inline]
    fn mul_u32(self, rhs: u32) -> Self {
        self * rhs
    }
}
impl VectorKey for UVec3 {
    #[inline]
    fn mul_u32(self, rhs: u32) -> Self {
        self * rhs
    }
}
