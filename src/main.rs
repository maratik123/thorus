use image::{ImageBuffer, Rgba};
use std::ffi::c_long;
use std::sync::Arc;
use thorus::shader::{load_fragment, load_vertex};
use thorus::vertex::MyVertex;
use tracing::debug;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::swapchain::{Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Version, VulkanLibrary};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new();
    debug!("event loop created");

    let library = VulkanLibrary::new().expect("no local Vulkan library");
    debug!("initialized library: {library:?}");

    let required_extensions = Surface::required_extensions(&event_loop);

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..InstanceCreateInfo::default()
        },
    )
    .expect("failed to create instance");
    debug!("initialized instance: {instance:?}");

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    debug!("window created");

    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
    debug!("surface created");

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::default()
    };

    let (physical_device, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);
    debug!("chosen physical device: {physical_device:?}");
    debug!("selected queue family index: {queue_family_index}");

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..QueueCreateInfo::default()
            }],
            enabled_extensions: device_extensions,
            ..DeviceCreateInfo::default()
        },
    )
    .expect("failed to create device");
    debug!("created device: {device:?}");

    let caps = physical_device
        .surface_capabilities(&surface, SurfaceInfo::default())
        .expect("failed to get surface capabilities");
    debug!("surface capabilities: {caps:?}");

    let dimensions = window.inner_size();
    debug!("dimensions: {dimensions:?}");

    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    debug!("composite alpha: {composite_alpha:?}");

    let image_format = physical_device
        .surface_formats(&surface, SurfaceInfo::default())
        .unwrap()
        .first()
        .unwrap()
        .0;
    debug!("image format: {image_format:?}");

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..SwapchainCreateInfo::default()
        },
    )
    .unwrap();

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

    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..BufferCreateInfo::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        },
        [
            MyVertex {
                position: [-0.5, -0.5],
            },
            MyVertex {
                position: [0.0, 0.5],
            },
            MyVertex {
                position: [0.5, -0.25],
            },
        ],
    )
    .unwrap();
    debug!("vertex buffer: {vertex_buffer:?}");

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: Format::R8G8B8A8_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap();
    debug!("render_pass: {render_pass:?}");

    event_loop.run(|event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        _ => (),
    });

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();
    debug!("image: {image:?}");

    let view = ImageView::new_default(image.clone()).unwrap();
    debug!("image view: {view:?}");

    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..FramebufferCreateInfo::default()
        },
    )
    .unwrap();
    debug!("framebuffer: {framebuffer:?}");

    let vs = load_vertex(device.clone()).unwrap();
    debug!("vertex shader: {vs:?}");

    let fs = load_fragment(device.clone()).unwrap();
    debug!("fragment shader: {fs:?}");

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [1024.0, 1024.0],
        depth_range: 0.0..=1.0,
    };
    debug!("viewport: {viewport:?}");

    let pipeline = {
        let vs = vs.entry_point("main").unwrap();
        debug!("vertex shader entry point: {vs:?}");

        let fs = fs.entry_point("main").unwrap();
        debug!("fragment shader entry point: {fs:?}");

        let vertex_input_state = MyVertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();
        debug!("vertex input state: {vertex_input_state:?}");

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        debug!("stages: {stages:?}");

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        debug!("pipeline layout: {layout:?}");

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        debug!("subpass: {subpass:?}");

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..ViewportState::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };
    debug!("graphics pipeline: {pipeline:?}");

    let buf = Buffer::new_sized::<[u8; 1024 * 1024 * 4]>(
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
    )
    .unwrap();
    debug!("buf: {buf:?}");

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..SubpassBeginInfo::default()
            },
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .unwrap()
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .unwrap()
        .draw(3, 1, 0, 0)
        .unwrap()
        .end_render_pass(SubpassEndInfo::default())
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buf.clone()))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    debug!("buffer content gathered");

    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    debug!("image created");

    image.save("image.png").unwrap();

    debug!("Everything ok");
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate physical devices")
        .filter(|d| d.api_version() >= Version::V1_3)
        .filter(|d| d.supported_extensions().contains(&device_extensions))
        .filter_map(|d| {
            d.queue_family_properties()
                .iter()
                .enumerate()
                .filter_map(|(i, q)| {
                    Some(i as u32).filter(|_| q.queue_flags.contains(QueueFlags::GRAPHICS))
                })
                .find(|&i| d.surface_support(i, &surface).unwrap_or(false))
                .map(|i| (d, i))
        })
        .max_by_key(|(d, _)| match d.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no devices available")
}
