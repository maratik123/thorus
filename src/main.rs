use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::VulkanLibrary;

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library");
    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");
}
