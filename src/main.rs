use std::sync::Arc;
use tracing::{debug, enabled, Level};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
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

    let data_iter = 0u32..65536;

    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..BufferCreateInfo::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        },
        data_iter,
    )
    .expect("failed to create data buffer");
    debug!("create data buffer: {data_buffer:?}");

    let shader = cs::load(device.clone()).expect("failed to create shader module");
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

    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .expect("can not get descriptor set layout");
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        [],
    )
    .expect("can not create persistent descriptor set");

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    debug!("create command buffer allocator: {command_buffer_allocator:?}");

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("failed to create auto command buffer builder");

    let work_group_counts = [1024, 1, 1];

    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .expect("failed to bind compute pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )
        .expect("failed to bind descriptor set")
        .dispatch(work_group_counts)
        .expect("failed to dispatch");

    let command_buffer = command_buffer_builder
        .build()
        .expect("failed to create command buffer");
    debug!("create auto command buffer");

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .expect("can not execute command buffer")
        .then_signal_fence_and_flush()
        .expect("can not flush");
    debug!("executing command buffer");

    future.wait(None).expect("can not wait execution");
    debug!("executed command buffer");

    let content = data_buffer.read().expect("can not read data buffer");
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }
    debug!("Everything ok");
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shader/shader.comp"
    }
}
