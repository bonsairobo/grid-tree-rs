use std::hash::Hash;
use std::ops::{Add, Shl, Shr, Sub};

/// An integer vector that can be used as a key for [`Tree`](crate::Tree).
pub trait VectorKey:
    Sized
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Shl<Output = Self>
    + Shr<Output = Self>
    + Eq
    + Hash
{
    // Need this because we can't impl `Mul<u32>` for foreign types.
    fn mul_u32(self, rhs: u32) -> Self;
}
