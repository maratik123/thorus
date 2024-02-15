use image::{ImageBuffer, Rgba};
use std::sync::Arc;
use thorus::shader;
use tracing::{debug, enabled, Level};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Version, VulkanLibrary};

fn main() {
    const PICTURE_SIZE: u32 = 4096;

    tracing_subscriber::fmt::init();

    let library = VulkanLibrary::new().expect("no local Vulkan library");
    debug!("initialized library: {library:?}");

    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");
    debug!("initialized instance: {instance:?}");

    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate physical devices")
        .find(|d| d.api_version() >= Version::V1_3)
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
                .contains(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a compute and graphics queue family")
        as u32;
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

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    debug!("create command buffer allocator: {command_buffer_allocator:?}");

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [PICTURE_SIZE, PICTURE_SIZE, 1],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..ImageCreateInfo::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..AllocationCreateInfo::default()
        },
    )
    .expect("image creation failed");
    debug!("image created: {image:?}");

    let view = ImageView::new_default(image.clone()).unwrap();
    debug!("created image view: {view:?}");

    let shader = shader::load(device.clone()).expect("failed to create shader module");
    debug!("loaded shader: {shader:?}");

    let cs = shader
        .entry_point("main")
        .expect("failed to create entry point");
    debug!("found entry point: {cs:?}");

    let stage = PipelineShaderStageCreateInfo::new(cs);
    debug!("created pipeline shader stage: {stage:?}");

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .expect("failed to convert to pipeline layout create info"),
    )
    .expect("failed to create pipeline layout");
    debug!("created pipeline layout: {layout:?}");

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("failed to create compute pipeline");
    debug!("created compute pipeline: {compute_pipeline:?}");

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
        device.clone(),
        StandardDescriptorSetAllocatorCreateInfo::default(),
    );
    debug!("created descriptor set allocator: {descriptor_set_allocator:?}");

    let layout = compute_pipeline.layout().set_layouts().first().unwrap();

    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())],
        [],
    )
    .expect("can not create persistent descriptor set");

    let buf = Buffer::from_iter(
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
        (0..PICTURE_SIZE * PICTURE_SIZE * 4).map(|_| 0u8),
    )
    .unwrap();
    debug!("buffer created: {buf:?}");

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .expect("failed to bind compute pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .expect("failed to bind descriptor set")
        .dispatch([PICTURE_SIZE / 8, PICTURE_SIZE / 8, 1])
        .expect("failed to dispatch")
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();
    debug!("command buffer created");

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    debug!("command buffer executing");

    future.wait(None).unwrap();
    debug!("command buffer executed");

    let buffer_content = buf.read().unwrap();
    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(PICTURE_SIZE, PICTURE_SIZE, &buffer_content[..])
            .unwrap();
    debug!("image created");

    image.save("image.png").unwrap();
    debug!("image saved");

    debug!("Everything ok");
}
