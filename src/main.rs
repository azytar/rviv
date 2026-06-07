use anyhow::{bail, Context, Result};
use ash::{vk, Device, Entry, Instance};
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use image::GenericImageView;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    ffi::CStr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    event::{ElementState, Event, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::Fullscreen,
    window::WindowBuilder,
};

#[derive(Parser, Debug)]
#[command(
    name = "rviv",
    about = "Optimized Vulkan Image Viewer & Wallpaper Setter"
)]
struct Cli {
    images: Vec<PathBuf>,
    #[arg(short = 'F', long)]
    fullscreen: bool,
    #[arg(long)]
    bg: Option<PathBuf>,
    #[arg(short = 'r', long)]
    recursive: bool,
    #[arg(short = 'S', long, default_value = "0.0")]
    slideshow: f32,
    #[arg(long)]
    sort: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PushConst {
    offset: [f32; 2],
    scale: [f32; 2],
}

struct Texture {
    image: vk::Image,
    mem: vk::DeviceMemory,
    view: vk::ImageView,
    dims: (u32, u32),
}

struct VkCtx {
    _entry: Entry,
    instance: Instance,
    device: Device,
    pdev: vk::PhysicalDevice,
    queue: vk::Queue,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    sc_images: Vec<vk::Image>,
    sc_views: Vec<vk::ImageView>,
    sc_format: vk::Format,
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    image_available: vk::Semaphore,
    render_done: vk::Semaphore,
    fence: vk::Fence,
    cache: std::collections::HashMap<PathBuf, Texture>,
    cache_order: std::collections::VecDeque<PathBuf>,
    max_cache_size: usize,
    tex_sampler: vk::Sampler,
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
}

fn cstr(s: &[u8]) -> &CStr {
    CStr::from_bytes_with_nul(s).unwrap()
}

impl VkCtx {
    unsafe fn find_memory_type(
        inst: &Instance,
        pdev: vk::PhysicalDevice,
        type_filter: u32,
        props: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_props = inst.get_physical_device_memory_properties(pdev);
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
            {
                return i;
            }
        }
        panic!("Memory type not found");
    }

    unsafe fn create_buffer(
        inst: &Instance,
        device: &Device,
        pdev: vk::PhysicalDevice,
        size: u64,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buf = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .unwrap();
        let req = device.get_buffer_memory_requirements(buf);
        let mem = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(Self::find_memory_type(
                        inst,
                        pdev,
                        req.memory_type_bits,
                        props,
                    )),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(buf, mem, 0).unwrap();
        (buf, mem)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn transition_image(
        device: &Device,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)],
        );
    }

    unsafe fn one_shot_begin(device: &Device, pool: vk::CommandPool) -> vk::CommandBuffer {
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];
        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
        cmd
    }

    unsafe fn one_shot_end(
        device: &Device,
        pool: vk::CommandPool,
        queue: vk::Queue,
        cmd: vk::CommandBuffer,
    ) {
        device.end_command_buffer(cmd).unwrap();
        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                vk::Fence::null(),
            )
            .unwrap();
        device.queue_wait_idle(queue).unwrap();
        device.free_command_buffers(pool, &[cmd]);
    }

    unsafe fn new(window: &winit::window::Window) -> Result<Self> {
        let entry = Entry::load().context("Vulkan entry fail")?;
        let display = window.display_handle()?.as_raw();
        let exts = ash_window::enumerate_required_extensions(display)?;
        let inst = entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0))
                .enabled_extension_names(exts),
            None,
        )?;
        let surface = ash_window::create_surface(
            &entry,
            &inst,
            display,
            window.window_handle()?.as_raw(),
            None,
        )?;
        let surface_loader = ash::khr::surface::Instance::new(&entry, &inst);
        let (pdev, qfam) = inst
            .enumerate_physical_devices()?
            .into_iter()
            .find_map(|pd| {
                inst.get_physical_device_queue_family_properties(pd)
                    .iter()
                    .enumerate()
                    .find_map(|(i, q)| {
                        if q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && surface_loader
                                .get_physical_device_surface_support(pd, i as u32, surface)
                                .unwrap_or(false)
                        {
                            Some((pd, i as u32))
                        } else {
                            None
                        }
                    })
            })
            .context("No GPU")?;
        let device = inst.create_device(
            pdev,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(qfam)
                    .queue_priorities(&[1.0])])
                .enabled_extension_names(&[ash::khr::swapchain::NAME.as_ptr()]),
            None,
        )?;
        let queue = device.get_device_queue(qfam, 0);
        let swapchain_loader = ash::khr::swapchain::Device::new(&inst, &device);
        let (swapchain, sc_images, sc_views, sc_format, extent) = Self::create_swapchain_resources(
            &surface_loader,
            &swapchain_loader,
            pdev,
            surface,
            &device,
            window.inner_size().width,
            window.inner_size().height,
        )?;
        let render_pass = Self::create_render_pass(&device, sc_format)?;
        let framebuffers = Self::create_framebuffers(&device, render_pass, &sc_views, extent)?;
        let descriptor_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ]),
            None,
        )?;
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[descriptor_layout])
                .push_constant_ranges(&[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    offset: 0,
                    size: std::mem::size_of::<PushConst>() as u32,
                }]),
            None,
        )?;
        let pipeline = Self::create_pipeline(&device, render_pass, pipeline_layout, extent)?;
        let cmd_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(qfam)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let cmd_bufs = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(sc_images.len() as u32),
        )?;
        let descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&[vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: sc_images.len() as u32,
                }])
                .max_sets(sc_images.len() as u32),
            None,
        )?;
        let descriptor_sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&vec![descriptor_layout; sc_images.len()]),
        )?;
        let image_available = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
        let render_done = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
        let fence = device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )?;
        let tex_sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR),
            None,
        )?;
        let (staging_buf, staging_mem) = Self::create_buffer(
            &inst,
            &device,
            pdev,
            4,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        Ok(Self {
            _entry: entry,
            instance: inst,
            device,
            pdev,
            queue,
            surface_loader,
            swapchain_loader,
            surface,
            swapchain,
            sc_images,
            sc_views,
            sc_format,
            extent,
            render_pass,
            framebuffers,
            pipeline_layout,
            pipeline,
            cmd_pool,
            cmd_bufs,
            descriptor_pool,
            descriptor_layout,
            descriptor_sets,
            image_available,
            render_done,
            fence,
            cache: std::collections::HashMap::new(),
            cache_order: std::collections::VecDeque::new(),
            max_cache_size: 10,
            tex_sampler,
            staging_buf,
            staging_mem,
        })
    }

    #[allow(clippy::type_complexity)]
    unsafe fn create_swapchain_resources(
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        pdev: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        device: &Device,
        width: u32,
        height: u32,
    ) -> Result<(
        vk::SwapchainKHR,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        vk::Format,
        vk::Extent2D,
    )> {
        let caps = surface_loader.get_physical_device_surface_capabilities(pdev, surface)?;
        let formats = surface_loader.get_physical_device_surface_formats(pdev, surface)?;
        let surf_fmt = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
            .unwrap_or(&formats[0]);
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        let img_count = (caps.min_image_count + 1).min(if caps.max_image_count > 0 {
            caps.max_image_count
        } else {
            u32::MAX
        });
        let swapchain = swapchain_loader.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(img_count)
                .image_format(surf_fmt.format)
                .image_color_space(surf_fmt.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true),
            None,
        )?;
        let sc_images = swapchain_loader.get_swapchain_images(swapchain)?;
        let sc_views = sc_images
            .iter()
            .map(|&img| {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(img)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surf_fmt.format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((swapchain, sc_images, sc_views, surf_fmt.format, extent))
    }

    unsafe fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass> {
        let attachment = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }]);
        let dep = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        Ok(device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&[attachment])
                .subpasses(&[subpass])
                .dependencies(&[dep]),
            None,
        )?)
    }

    unsafe fn create_framebuffers(
        device: &Device,
        render_pass: vk::RenderPass,
        views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>> {
        views
            .iter()
            .map(|&v| {
                Ok(device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&[v])
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )?)
            })
            .collect()
    }

    unsafe fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
        extent: vk::Extent2D,
    ) -> Result<vk::Pipeline> {
        let v_spv = include_bytes!("shaders/vert.spv");
        let f_spv = include_bytes!("shaders/frag.spv");

        let v_mod = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default()
                .code(bytemuck::cast_slice(v_spv)),
            None,
        )?;
        let f_mod = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default()
                .code(bytemuck::cast_slice(f_spv)),
            None,
        )?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(v_mod)
                .name(cstr(b"main\0")),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(f_mod)
                .name(cstr(b"main\0")),
        ];
        let pipeline = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                    .input_assembly_state(
                        &vk::PipelineInputAssemblyStateCreateInfo::default()
                            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP),
                    )
                    .viewport_state(
                        &vk::PipelineViewportStateCreateInfo::default()
                            .viewports(&[vk::Viewport {
                                x: 0.0,
                                y: 0.0,
                                width: extent.width as f32,
                                height: extent.height as f32,
                                min_depth: 0.0,
                                max_depth: 1.0,
                            }])
                            .scissors(&[vk::Rect2D {
                                offset: vk::Offset2D::default(),
                                extent,
                            }]),
                    )
                    .rasterization_state(
                        &vk::PipelineRasterizationStateCreateInfo::default()
                            .polygon_mode(vk::PolygonMode::FILL)
                            .cull_mode(vk::CullModeFlags::NONE)
                            .front_face(vk::FrontFace::CLOCKWISE)
                            .line_width(1.0),
                    )
                    .multisample_state(
                        &vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                    )
                    .color_blend_state(
                        &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
                            vk::PipelineColorBlendAttachmentState::default()
                                .color_write_mask(vk::ColorComponentFlags::RGBA)
                                .blend_enable(true)
                                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                                .color_blend_op(vk::BlendOp::ADD)
                                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                                .alpha_blend_op(vk::BlendOp::ADD),
                        ]),
                    )
                    .layout(layout)
                    .render_pass(render_pass)
                    .subpass(0)],
                None,
            )
            .map_err(|e| e.1)?[0];
        device.destroy_shader_module(v_mod, None);
        device.destroy_shader_module(f_mod, None);
        Ok(pipeline)
    }

    unsafe fn prepare_texture(&mut self, path: &PathBuf) -> Result<(u32, u32)> {
        if let Some(tex) = self.cache.get(path) {
            if let Some(pos) = self.cache_order.iter().position(|p| p == path) {
                self.cache_order.remove(pos);
            }
            self.cache_order.push_back(path.clone());
            self.update_descriptors(tex.view);
            return Ok(tex.dims);
        }
        let img = image::open(path)?;
        let dims = img.dimensions();
        let rgba = img.to_rgba8();
        let px = rgba.into_raw();
        self.device.queue_wait_idle(self.queue)?;
        self.device.destroy_buffer(self.staging_buf, None);
        self.device.free_memory(self.staging_mem, None);
        let (s_buf, s_mem) = Self::create_buffer(
            &self.instance,
            &self.device,
            self.pdev,
            px.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let ptr = self
            .device
            .map_memory(s_mem, 0, px.len() as u64, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(px.as_ptr(), ptr as *mut u8, px.len());
        self.device.unmap_memory(s_mem);
        self.staging_buf = s_buf;
        self.staging_mem = s_mem;
        let tex_image = self.device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .extent(vk::Extent3D {
                    width: dims.0,
                    height: dims.1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )?;
        let req = self.device.get_image_memory_requirements(tex_image);
        let tex_mem = self.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(Self::find_memory_type(
                    &self.instance,
                    self.pdev,
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )),
            None,
        )?;
        self.device.bind_image_memory(tex_image, tex_mem, 0)?;
        let cmd = Self::one_shot_begin(&self.device, self.cmd_pool);
        Self::transition_image(
            &self.device,
            cmd,
            tex_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );
        self.device.cmd_copy_buffer_to_image(
            cmd,
            s_buf,
            tex_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_extent: vk::Extent3D {
                    width: dims.0,
                    height: dims.1,
                    depth: 1,
                },
                ..Default::default()
            }],
        );
        Self::transition_image(
            &self.device,
            cmd,
            tex_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        Self::one_shot_end(&self.device, self.cmd_pool, self.queue, cmd);
        let tex_view = self.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(tex_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )?;
        self.update_descriptors(tex_view);
        if self.cache_order.len() >= self.max_cache_size {
            if let Some(old_path) = self.cache_order.pop_front() {
                if let Some(old_tex) = self.cache.remove(&old_path) {
                    self.device.destroy_image_view(old_tex.view, None);
                    self.device.destroy_image(old_tex.image, None);
                    self.device.free_memory(old_tex.mem, None);
                }
            }
        }
        self.cache.insert(
            path.clone(),
            Texture {
                image: tex_image,
                mem: tex_mem,
                view: tex_view,
                dims,
            },
        );
        self.cache_order.push_back(path.clone());
        Ok(dims)
    }

    unsafe fn update_descriptors(&self, view: vk::ImageView) {
        for &ds in &self.descriptor_sets {
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&[vk::DescriptorImageInfo {
                        sampler: self.tex_sampler,
                        image_view: view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    }])],
                &[],
            );
        }
    }

    unsafe fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        self.device.queue_wait_idle(self.queue)?;
        for &v in &self.sc_views {
            self.device.destroy_image_view(v, None);
        }
        for &f in &self.framebuffers {
            self.device.destroy_framebuffer(f, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
        let (sc, imgs, views, fmt, ext) = Self::create_swapchain_resources(
            &self.surface_loader,
            &self.swapchain_loader,
            self.pdev,
            self.surface,
            &self.device,
            width,
            height,
        )?;
        self.swapchain = sc;
        self.sc_images = imgs;
        self.sc_views = views;
        self.sc_format = fmt;
        self.extent = ext;
        self.framebuffers =
            Self::create_framebuffers(&self.device, self.render_pass, &self.sc_views, self.extent)?;
        self.device.destroy_pipeline(self.pipeline, None);
        self.pipeline = Self::create_pipeline(
            &self.device,
            self.render_pass,
            self.pipeline_layout,
            self.extent,
        )?;
        Ok(())
    }

    unsafe fn draw(&self, pc: PushConst, bg: [f32; 4]) -> Result<()> {
        self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        let result = self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available,
            vk::Fence::null(),
        );
        let idx = match result {
            Ok((idx, _)) => idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(()),
            Err(e) => bail!("Vulkan error: {:?}", e),
        };
        self.device.reset_fences(&[self.fence])?;
        let cmd = self.cmd_bufs[idx as usize];
        self.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        self.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        self.device.cmd_begin_render_pass(
            cmd,
            &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[idx as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: self.extent,
                })
                .clear_values(&[vk::ClearValue {
                    color: vk::ClearColorValue { float32: bg },
                }]),
            vk::SubpassContents::INLINE,
        );
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.descriptor_sets[idx as usize]],
            &[],
        );
        self.device.cmd_push_constants(
            cmd,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytemuck::bytes_of(&pc),
        );
        self.device.cmd_draw(cmd, 4, 1, 0, 0);
        self.device.cmd_end_render_pass(cmd);
        self.device.end_command_buffer(cmd)?;
        self.device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::default()
                .wait_semaphores(&[self.image_available])
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&[cmd])
                .signal_semaphores(&[self.render_done])],
            self.fence,
        )?;
        let _ = self.swapchain_loader.queue_present(
            self.queue,
            &vk::PresentInfoKHR::default()
                .wait_semaphores(&[self.render_done])
                .swapchains(&[self.swapchain])
                .image_indices(&[idx]),
        );
        Ok(())
    }
}

impl Drop for VkCtx {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.queue_wait_idle(self.queue);
            for (_, tex) in self.cache.drain() {
                self.device.destroy_image_view(tex.view, None);
                self.device.destroy_image(tex.image, None);
                self.device.free_memory(tex.mem, None);
            }
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_semaphore(self.render_done, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            for &f in &self.framebuffers {
                self.device.destroy_framebuffer(f, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for &v in &self.sc_views {
                self.device.destroy_image_view(v, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_sampler(self.tex_sampler, None);
            self.device.destroy_buffer(self.staging_buf, None);
            self.device.free_memory(self.staging_mem, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn set_wallpaper_x11(path: &Path) -> Result<()> {
    use x11rb::connection::Connection;
    use x11rb::image::{BitsPerPixel, Image, ImageOrder, ScanlinePad};
    use x11rb::protocol::xproto::{self, ConnectionExt as _};
    use x11rb::wrapper::ConnectionExt;

    let (conn, screen_num) = x11rb::connect(None)?;
    let screen = &conn.setup().roots[screen_num];
    let root = screen.root;

    let img = image::open(path)?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();

    // 1. Create a 32-bit pixmap for picom and other compositors
    let depth32 = screen.allowed_depths.iter().find(|d| d.depth == 32);
    let pixmap32 = if let Some(d) = depth32 {
        let p = conn.generate_id()?;
        conn.create_pixmap(d.depth, p, root, w as u16, h as u16)?;
        let mut data32 = Vec::with_capacity((w * h * 4) as usize);
        for p in rgba.pixels() {
            data32.extend_from_slice(&[p[2], p[1], p[0], p[3]]); // BGRA
        }
        let gc = conn.generate_id()?;
        conn.create_gc(gc, p, &xproto::CreateGCAux::default())?;
        Image::new(
            w as u16,
            h as u16,
            ScanlinePad::Pad32,
            d.depth,
            BitsPerPixel::B32,
            ImageOrder::LsbFirst,
            std::borrow::Cow::Borrowed(&data32),
        )?
        .put(&conn, p, gc, 0, 0)?;
        conn.free_gc(gc)?;
        Some(p)
    } else {
        None
    };

    // 2. Create a pixmap matching root depth for the actual background
    let pixmap_root = conn.generate_id()?;
    conn.create_pixmap(screen.root_depth, pixmap_root, root, w as u16, h as u16)?;
    let mut data_root = Vec::with_capacity((w * h * 4) as usize);
    for p in rgba.pixels() {
        // Simple composition over black for the root background
        let alpha = p[3] as f32 / 255.0;
        let r = (p[0] as f32 * alpha) as u8;
        let g = (p[1] as f32 * alpha) as u8;
        let b = (p[2] as f32 * alpha) as u8;
        data_root.extend_from_slice(&[b, g, r, 0]);
    }
    let gc_root = conn.generate_id()?;
    conn.create_gc(gc_root, pixmap_root, &xproto::CreateGCAux::default())?;
    Image::new(
        w as u16,
        h as u16,
        ScanlinePad::Pad32,
        screen.root_depth,
        BitsPerPixel::B32,
        ImageOrder::LsbFirst,
        std::borrow::Cow::Borrowed(&data_root),
    )?
    .put(&conn, pixmap_root, gc_root, 0, 0)?;
    conn.free_gc(gc_root)?;

    // 3. Set properties and background
    let pmap_for_props = pixmap32.unwrap_or(pixmap_root);
    for name in ["_XROOTPMAP_ID", "ESETROOT_PMAP_ID"] {
        let atom = conn.intern_atom(false, name.as_bytes())?.reply()?.atom;
        conn.change_property32(
            xproto::PropMode::REPLACE,
            root,
            atom,
            xproto::AtomEnum::PIXMAP,
            &[pmap_for_props],
        )?;
    }

    conn.change_window_attributes(
        root,
        &xproto::ChangeWindowAttributesAux::new().background_pixmap(pixmap_root),
    )?;
    conn.clear_area(
        false,
        root,
        0,
        0,
        screen.width_in_pixels,
        screen.height_in_pixels,
    )?;

    conn.set_close_down_mode(xproto::CloseDown::RETAIN_PERMANENT)?;
    conn.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(bg) = cli.bg {
        let abs = std::fs::canonicalize(&bg).unwrap_or(bg);
        if std::env::var("WAYLAND_DISPLAY").is_err() {
            set_wallpaper_x11(&abs)?;
        } else {
            println!("Wallpaper support is currently X11-only.");
        }
        return Ok(());
    }
    let mut images = Vec::new();
    let mut current_idx = 0;
    let input_paths = if cli.images.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        cli.images.clone()
    };
    for p in &input_paths {
        if p.is_dir() {
            let w = if cli.recursive {
                walkdir::WalkDir::new(p)
            } else {
                walkdir::WalkDir::new(p).max_depth(1)
            };
            for e in w.into_iter().filter_map(|e| e.ok()) {
                if e.file_type().is_file() {
                    let path =
                        std::fs::canonicalize(e.path()).unwrap_or_else(|_| e.path().to_path_buf());
                    let ext = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if matches!(
                        ext.as_str(),
                        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
                    ) && !images.contains(&path)
                    {
                        images.push(path);
                    }
                }
            }
        } else {
            let abs = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
            let ext = abs
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            if matches!(
                ext.as_str(),
                "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
            ) && !images.contains(&abs)
            {
                images.push(abs.clone());
            }
            let parent = abs.parent().unwrap_or(Path::new("."));
            for e in walkdir::WalkDir::new(parent)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if e.file_type().is_file() {
                    let path =
                        std::fs::canonicalize(e.path()).unwrap_or_else(|_| e.path().to_path_buf());
                    let ext = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_lowercase();
                    if matches!(
                        ext.as_str(),
                        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
                    ) && !images.contains(&path)
                    {
                        images.push(path);
                    }
                }
            }
        }
    }
    if cli.sort {
        images.sort();
    }
    if let Some(f) = input_paths.first() {
        let abs = std::fs::canonicalize(f).unwrap_or_else(|_| f.clone());
        if let Some(p) = images.iter().position(|x| x == &abs) {
            current_idx = p;
        }
    }
    if images.is_empty() {
        bail!("No images found");
    }
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("rviv")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .build(&event_loop)?,
    );
    if cli.fullscreen {
        window.set_fullscreen(Some(Fullscreen::Borderless(None)));
    }
    let mut vk = unsafe { VkCtx::new(&window)? };
    let (mut zoom, mut offset, mut last_load, mut needs_up, mut dims, mut modifiers) = (
        1.0f32,
        [0.0f32, 0.0f32],
        Instant::now(),
        true,
        (1u32, 1u32),
        winit::event::Modifiers::default(),
    );
    window.request_redraw();
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(if cli.slideshow > 0.0 {
                ControlFlow::WaitUntil(Instant::now() + Duration::from_secs_f32(cli.slideshow))
            } else {
                ControlFlow::Wait
            });
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(s) => unsafe {
                        vk.recreate_swapchain(s.width, s.height).unwrap();
                    },
                    WindowEvent::ModifiersChanged(m) => modifiers = m,
                    WindowEvent::RedrawRequested => {
                        if needs_up {
                            dims = unsafe { vk.prepare_texture(&images[current_idx]).unwrap() };
                            needs_up = false;
                            window.set_title(&format!("rviv - {}", images[current_idx].display()));
                            let (ww, wh) = (vk.extent.width as f32, vk.extent.height as f32);
                            zoom = (ww / dims.0 as f32).min(wh / dims.1 as f32).min(1.0);
                            offset = [0.0, 0.0];
                        }
                        let (ww, wh) = (vk.extent.width as f32, vk.extent.height as f32);
                        let (sx, sy) = ((ww / dims.0 as f32) / zoom, (wh / dims.1 as f32) / zoom);
                        unsafe {
                            vk.draw(
                                PushConst {
                                    offset: [
                                        0.5 * (1.0 - sx) - offset[0] * sx,
                                        0.5 * (1.0 - sy) - offset[1] * sy,
                                    ],
                                    scale: [sx, sy],
                                },
                                [0.05, 0.05, 0.05, 1.0],
                            )
                            .unwrap();
                        }
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key,
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        let is_shift = modifiers.state().shift_key();
                        match logical_key {
                            Key::Named(NamedKey::Escape) => elwt.exit(),
                            Key::Character(ref s) if s == "q" => elwt.exit(),
                            Key::Named(NamedKey::ArrowRight) if is_shift => {
                                current_idx = (current_idx + 1) % images.len();
                                needs_up = true;
                                last_load = Instant::now();
                                window.request_redraw();
                            }
                            Key::Named(NamedKey::ArrowLeft) if is_shift => {
                                current_idx = (current_idx + images.len() - 1) % images.len();
                                needs_up = true;
                                last_load = Instant::now();
                                window.request_redraw();
                            }
                            Key::Character(ref s) if s == "l" => {
                                current_idx = (current_idx + 1) % images.len();
                                needs_up = true;
                                last_load = Instant::now();
                                window.request_redraw();
                            }
                            Key::Character(ref s) if s == "h" => {
                                current_idx = (current_idx + images.len() - 1) % images.len();
                                needs_up = true;
                                last_load = Instant::now();
                                window.request_redraw();
                            }
                            Key::Named(NamedKey::ArrowRight) if !is_shift => {
                                offset[0] -= 0.1 / zoom;
                                window.request_redraw();
                            }
                            Key::Named(NamedKey::ArrowLeft) if !is_shift => {
                                offset[0] += 0.1 / zoom;
                                window.request_redraw();
                            }
                            Key::Named(NamedKey::ArrowUp) => {
                                offset[1] += 0.1 / zoom;
                                window.request_redraw();
                            }
                            Key::Named(NamedKey::ArrowDown) => {
                                offset[1] -= 0.1 / zoom;
                                window.request_redraw();
                            }
                            Key::Character(ref s) if s == "j" => {
                                offset[1] -= 0.1 / zoom;
                                window.request_redraw();
                            }
                            Key::Character(ref s) if s == "k" => {
                                offset[1] += 0.1 / zoom;
                                window.request_redraw();
                            }
                            Key::Character(ref s) if s == "+" || s == "=" => {
                                zoom *= 1.2;
                                window.request_redraw();
                            }
                            Key::Character(ref s) if s == "-" => {
                                zoom /= 1.2;
                                window.request_redraw();
                            }
                            Key::Character(ref s) if s == "0" => {
                                zoom = 1.0;
                                offset = [0.0, 0.0];
                                window.request_redraw();
                            }
                            Key::Character(ref s) if s == "f" => {
                                let is_full = window.fullscreen().is_some();
                                window.set_fullscreen(if is_full {
                                    None
                                } else {
                                    Some(Fullscreen::Borderless(None))
                                });
                            }
                            _ => (),
                        }
                    }
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, y),
                        ..
                    } => {
                        if y > 0.0 {
                            zoom *= 1.1;
                        } else {
                            zoom /= 1.1;
                        }
                        window.request_redraw();
                    }
                    _ => (),
                },
                Event::AboutToWait => {
                    if cli.slideshow > 0.0
                        && last_load.elapsed() > Duration::from_secs_f32(cli.slideshow)
                    {
                        current_idx = (current_idx + 1) % images.len();
                        needs_up = true;
                        last_load = Instant::now();
                        window.request_redraw();
                    }
                }
                _ => (),
            }
        })
        .unwrap();
    Ok(())
}
