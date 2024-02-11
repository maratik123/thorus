use tracing::{debug, enabled, Level};
use vulkano::device::QueueFlags;
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

    if enabled!(Level::DEBUG) {
        for (pos, family) in physical_device.queue_family_properties().iter().enumerate() {
            debug!("found a queue family at index {pos}: {family:?}");
        }
    }

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_property| {
            queue_family_property
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphical queue family");
    debug!("selected queue family index: {queue_family_index}");
}
