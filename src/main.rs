use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::VulkanLibrary;

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library");

    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");

    let physical_devices = instance
        .enumerate_physical_devices()
        .expect("could not enumerate physical devices")
        .next()
        .expect("no devices available");
}
