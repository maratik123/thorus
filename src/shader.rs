pub mod vs {
    vulkano_shaders::shader! {
        vulkan_version: "1.2",
        spirv_version: "1.6",
        ty: "vertex",
        path: "shader/shader.vert"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        vulkan_version: "1.2",
        spirv_version: "1.6",
        ty: "fragment",
        path: "shader/shader.frag"
    }
}
