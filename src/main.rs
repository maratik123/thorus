use tracing::debug;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::VulkanLibrary;

fn main() {
    tracing_subscriber::fmt::init();

    let library = VulkanLibrary::new().expect("no local Vulkan library");
    debug!("initialized library: {library:?}");

    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");
    debug!("initialized instance: {instance:?}");

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate physical devices")
        .next()
        .expect("no devices available");
    debug!("chosen physical device: {physical_device:?}");

    for family in physical_device.queue_family_properties() {
        debug!(
            "found a queue family with {:?} queue(s)",
            family.queue_count
        );
    }
}
