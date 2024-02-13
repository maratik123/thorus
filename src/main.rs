use std::sync::Arc;
use tracing::{debug, enabled, Level};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfoTyped};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::GpuFuture;
use vulkano::{sync, VulkanLibrary};

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
            debug!("found a queue family with property at index {pos}: {family:?}");
        }
    }

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphical queue family") as u32;
    debug!("selected queue family index: {queue_family_index}");

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..QueueCreateInfo::default()
            }],
            ..DeviceCreateInfo::default()
        },
    )
    .expect("failed to create device");
    debug!("created device: {device:?}");

    let queue = queues
        .next()
        .expect("at least one queue expected for device");
    debug!("device queue: {queue:?}");

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    debug!("created memory allocator: {memory_allocator:?}");

    let source_content: Vec<i32> = (0..1024).collect();

    let source = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..BufferCreateInfo::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        },
        source_content,
    )
    .expect("failed to create source buffer");
    debug!("create source buffer: {source:?}");

    let destination_content: Vec<i32> = (0..1024).map(|_| 0i32).collect();
    let destination = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..BufferCreateInfo::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..AllocationCreateInfo::default()
        },
        destination_content,
    )
    .expect("failed to create destination buffer");
    debug!("create destination buffer {destination:?}");

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    debug!("create command buffer allocator: {command_buffer_allocator:?}");

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("failed to create auto command buffer builder");

    builder
        .copy_buffer(CopyBufferInfoTyped::buffers(
            source.clone(),
            destination.clone(),
        ))
        .expect("failed to copy buffers");

    let command_buffer = builder.build().expect("failed to create command buffer");
    debug!("create auto command buffer");

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .expect("can not execute command buffer")
        .then_signal_fence_and_flush()
        .expect("can not flush");
    debug!("executing command buffer");

    future.wait(None).expect("can not wait execution");
    debug!("executed command buffer");

    let src_content = source.read().expect("can not read source");
    let dst_content = destination.read().expect("can not read destination");
    assert_eq!(&*src_content, &*dst_content);
    debug!("Everything ok");
}
