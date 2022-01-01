use std::fmt::{Display, Formatter};

pub enum TestSize {
    Small,
    Medium,
    Large,
}

impl Display for TestSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TestSize::Small => write!(f, "Small"),
            TestSize::Medium => write!(f, "Medium"),
            TestSize::Large => write!(f, "Large"),
        }
    }
}
