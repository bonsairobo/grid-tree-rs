use crate::{ChildIndex, VectorKey};

/// The shape of a single node's children. E.g. 2<sup>2</sup> for a quadtree and 2<sup>3</sup> for an octree.
pub trait BranchShape<V>
where
    V: VectorKey,
{
    /// The number of bits to shift each dimension in order to translate coordinates between adjacent [`Level`](crate::Level)s.
    const SHAPE_SHIFTER: V;

    fn linearize_child(offset: V) -> ChildIndex;
    fn delinearize_child(i: ChildIndex) -> V;

    #[inline]
    fn parent_key(key: V) -> V {
        key >> Self::SHAPE_SHIFTER
    }

    #[inline]
    fn ancestor_key(key: V, levels_up: u32) -> V {
        key >> Self::SHAPE_SHIFTER.mul_u32(levels_up)
    }

    #[inline]
    fn min_child_key(key: V) -> V {
        key << Self::SHAPE_SHIFTER
    }

    #[inline]
    fn min_descendant_key(key: V, levels_down: u32) -> V {
        key << Self::SHAPE_SHIFTER.mul_u32(levels_down)
    }
}
