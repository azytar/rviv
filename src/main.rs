use anyhow::{bail, Context, Result};
use ash::{vk, Device, Entry, Instance};
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use image::GenericImageView;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    ffi::CStr,
    path::{Path, PathBuf},
    sync::{mpsc, Arc, atomic::{AtomicUsize, Ordering}},
    thread,
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

const MAX_FRAMES_IN_FLIGHT: usize = 2;

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
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
    image_available: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    render_done: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    fences: [vk::Fence; MAX_FRAMES_IN_FLIGHT],
    current_frame: usize,
    cache: std::collections::HashMap<PathBuf, Texture>,
    cache_order: std::collections::VecDeque<PathBuf>,
    max_cache_size: usize,
    tex_sampler: vk::Sampler,
    staging_buf: vk::Buffer,
    staging_mem: vk::DeviceMemory,
    staging_capacity: u64,
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
    ) -> Result<u32> {
        let mem_props = inst.get_physical_device_memory_properties(pdev);
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(props)
            {
                return Ok(i);
            }
        }
        bail!("Memory type not found")
    }

    unsafe fn create_buffer(
        inst: &Instance,
        device: &Device,
        pdev: vk::PhysicalDevice,
        size: u64,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buf = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let req = device.get_buffer_memory_requirements(buf);
        let mem_result = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(Self::find_memory_type(
                    inst,
                    pdev,
                    req.memory_type_bits,
                    props,
                )?),
            None,
        );

        match mem_result {
            Ok(mem) => {
                device.bind_buffer_memory(buf, mem, 0)?;
                Ok((buf, mem))
            }
            Err(e) => {
                device.destroy_buffer(buf, None);
                Err(e.into())
            }
        }
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

    unsafe fn one_shot_begin(device: &Device, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        Ok(cmd)
    }

    unsafe fn one_shot_end(
        device: &Device,
        pool: vk::CommandPool,
        queue: vk::Queue,
        cmd: vk::CommandBuffer,
    ) -> Result<()> {
        device.end_command_buffer(cmd)?;
        device.queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(&[cmd])],
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(pool, &[cmd]);
        Ok(())
    }

    unsafe fn new(window: &winit::window::Window) -> Result<Self> {
        let entry = Entry::load().context("Vulkan entry fail")?;
        let display = window.display_handle()?.as_raw();
        let exts = ash_window::enumerate_required_extensions(display)?;
        let inst = entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3))
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
        let mut dynamic_rendering_feature =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let device = inst.create_device(
            pdev,
            &vk::DeviceCreateInfo::default()
                .push_next(&mut dynamic_rendering_feature)
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
        let pipeline = Self::create_pipeline(&device, sc_format, pipeline_layout, extent)?;
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
        let mut image_available = [vk::Semaphore::null(); MAX_FRAMES_IN_FLIGHT];
        let mut render_done = [vk::Semaphore::null(); MAX_FRAMES_IN_FLIGHT];
        let mut fences = [vk::Fence::null(); MAX_FRAMES_IN_FLIGHT];
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            image_available[i] = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            render_done[i] = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            fences[i] = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
        }

        let tex_sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR),
            None,
        )?;
        let (staging_buf, staging_mem) = match Self::create_buffer(
            &inst,
            &device,
            pdev,
            4,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(v) => v,
            Err(e) => {
                for i in 0..MAX_FRAMES_IN_FLIGHT {
                    device.destroy_semaphore(image_available[i], None);
                    device.destroy_semaphore(render_done[i], None);
                    device.destroy_fence(fences[i], None);
                }
                device.destroy_sampler(tex_sampler, None);
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_command_pool(cmd_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_layout, None);
                for &v in &sc_views {
                    device.destroy_image_view(v, None);
                }
                swapchain_loader.destroy_swapchain(swapchain, None);
                device.destroy_device(None);
                surface_loader.destroy_surface(surface, None);
                inst.destroy_instance(None);
                return Err(e);
            }
        };
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
            pipeline_layout,
            pipeline,
            cmd_pool,
            cmd_bufs,
            descriptor_pool,
            descriptor_layout,
            descriptor_sets,
            image_available,
            render_done,
            fences,
            current_frame: 0,
            cache: std::collections::HashMap::new(),
            cache_order: std::collections::VecDeque::new(),
            max_cache_size: 10,
            tex_sampler,
            staging_buf,
            staging_mem,
            staging_capacity: 4,
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
            .context("B8G8R8A8_SRGB format not found")
            .or_else(|_| formats.first().context("No surface formats found"))?;
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

    unsafe fn create_pipeline(
        device: &Device,
        format: vk::Format,
        layout: vk::PipelineLayout,
        extent: vk::Extent2D,
    ) -> Result<vk::Pipeline> {
        let v_spv = include_bytes!("shaders/vert.spv");
        let f_spv = include_bytes!("shaders/frag.spv");

        let v_code = ash::util::read_spv(&mut std::io::Cursor::new(v_spv))
            .context("Failed to read vertex shader SPIR-V")?;
        let f_code = ash::util::read_spv(&mut std::io::Cursor::new(f_spv))
            .context("Failed to read fragment shader SPIR-V")?;

        let v_mod = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&v_code),
            None,
        )?;
        let f_mod = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&f_code),
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

        let color_formats = [format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats);

        let pipeline = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .push_next(&mut rendering_info)
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
                    .layout(layout)],
                None,
            )
            .map_err(|e| e.1)?[0];
        device.destroy_shader_module(v_mod, None);
        device.destroy_shader_module(f_mod, None);
        Ok(pipeline)
    }

    unsafe fn check_cache(&mut self, path: &Path) -> Option<(u32, u32)> {
        if let Some(tex) = self.cache.get(path) {
            if let Some(pos) = self.cache_order.iter().position(|p| p == path) {
                self.cache_order.remove(pos);
            }
            self.cache_order.push_back(path.to_path_buf());
            self.update_descriptors(tex.view);
            return Some(tex.dims);
        }
        None
    }

    unsafe fn upload_texture(
        &mut self,
        path: PathBuf,
        rgba: image::RgbaImage,
        dims: (u32, u32),
    ) -> Result<()> {
        let px = rgba.into_raw();
        let size = px.len() as u64;

        if size > self.staging_capacity {
            let (new_buf, new_mem) = Self::create_buffer(
                &self.instance,
                &self.device,
                self.pdev,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            
            self.device.queue_wait_idle(self.queue)?;
            self.device.destroy_buffer(self.staging_buf, None);
            self.device.free_memory(self.staging_mem, None);
            
            self.staging_buf = new_buf;
            self.staging_mem = new_mem;
            self.staging_capacity = size;
        }

        let ptr = self.device.map_memory(
            self.staging_mem,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(px.as_ptr(), ptr as *mut u8, px.len());
        self.device.unmap_memory(self.staging_mem);

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
        let tex_mem_result = self.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(Self::find_memory_type(
                    &self.instance,
                    self.pdev,
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?),
            None,
        );

        let tex_mem = match tex_mem_result {
            Ok(m) => m,
            Err(e) => {
                self.device.destroy_image(tex_image, None);
                return Err(e.into());
            }
        };

        if let Err(e) = self.device.bind_image_memory(tex_image, tex_mem, 0) {
            self.device.free_memory(tex_mem, None);
            self.device.destroy_image(tex_image, None);
            return Err(e.into());
        }

        let cmd = match Self::one_shot_begin(&self.device, self.cmd_pool) {
            Ok(c) => c,
            Err(e) => {
                self.device.free_memory(tex_mem, None);
                self.device.destroy_image(tex_image, None);
                return Err(e);
            }
        };

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
            self.staging_buf,
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

        if let Err(e) = Self::one_shot_end(&self.device, self.cmd_pool, self.queue, cmd) {
            self.device.free_memory(tex_mem, None);
            self.device.destroy_image(tex_image, None);
            return Err(e);
        }

        let tex_view_result = self.device.create_image_view(
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
        );

        let tex_view = match tex_view_result {
            Ok(v) => v,
            Err(e) => {
                self.device.free_memory(tex_mem, None);
                self.device.destroy_image(tex_image, None);
                return Err(e.into());
            }
        };

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
        self.cache_order.push_back(path);
        Ok(())
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

        self.device
            .destroy_descriptor_pool(self.descriptor_pool, None);
        self.descriptor_pool = self.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&[vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: self.sc_images.len() as u32,
                }])
                .max_sets(self.sc_images.len() as u32),
            None,
        )?;
        self.descriptor_sets = self.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&vec![self.descriptor_layout; self.sc_images.len()]),
        )?;

        self.device.free_command_buffers(self.cmd_pool, &self.cmd_bufs);
        self.cmd_bufs = self.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(self.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(self.sc_images.len() as u32),
        )?;

        self.device.destroy_pipeline(self.pipeline, None);
        self.pipeline = Self::create_pipeline(
            &self.device,
            self.sc_format,
            self.pipeline_layout,
            self.extent,
        )?;

        if let Some(path) = self.cache_order.back().cloned() {
            if let Some(tex) = self.cache.get(&path) {
                self.update_descriptors(tex.view);
            }
        }

        Ok(())
    }

    unsafe fn draw(&mut self, pc: PushConst, bg: [f32; 4]) -> Result<()> {
        let frame_idx = self.current_frame;
        self.device
            .wait_for_fences(&[self.fences[frame_idx]], true, u64::MAX)?;

        let result = self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available[frame_idx],
            vk::Fence::null(),
        );
        let idx = match result {
            Ok((idx, _)) => idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(()),
            Err(e) => bail!("Vulkan error: {:?}", e),
        };

        self.device.reset_fences(&[self.fences[frame_idx]])?;
        let cmd = self.cmd_bufs[idx as usize];
        self.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        self.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

        Self::transition_image(
            &self.device,
            cmd,
            self.sc_images[idx as usize],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        );

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.sc_views[idx as usize])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: bg },
            });

        let color_attachments = [color_attachment];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);

        self.device.cmd_begin_rendering(cmd, &rendering_info);
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
        self.device.cmd_end_rendering(cmd);

        Self::transition_image(
            &self.device,
            cmd,
            self.sc_images[idx as usize],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::empty(),
        );

        self.device.end_command_buffer(cmd)?;
        self.device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::default()
                .wait_semaphores(&[self.image_available[frame_idx]])
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&[cmd])
                .signal_semaphores(&[self.render_done[frame_idx]])],
            self.fences[frame_idx],
        )?;
        let _ = self.swapchain_loader.queue_present(
            self.queue,
            &vk::PresentInfoKHR::default()
                .wait_semaphores(&[self.render_done[frame_idx]])
                .swapchains(&[self.swapchain])
                .image_indices(&[idx]),
        );

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
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
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_semaphore(self.image_available[i], None);
                self.device.destroy_semaphore(self.render_done[i], None);
                self.device.destroy_fence(self.fences[i], None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
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

    let load_version = Arc::new(AtomicUsize::new(0));
    let (tx_load, rx_load) = mpsc::channel::<(PathBuf, usize)>();
    let (tx_done, rx_done) = mpsc::channel::<Result<(PathBuf, image::RgbaImage, (u32, u32), usize), (PathBuf, String, usize)>>();
    
    let worker_version = Arc::clone(&load_version);
    let worker_window = Arc::clone(&window);
    thread::spawn(move || {
        while let Ok((path, version)) = rx_load.recv() {
            if version < worker_version.load(Ordering::Relaxed) {
                continue;
            }
            match image::open(&path) {
                Ok(img) => {
                    let dims = img.dimensions();
                    let rgba = img.to_rgba8();
                    let _ = tx_done.send(Ok((path, rgba, dims, version)));
                }
                Err(e) => {
                    let _ = tx_done.send(Err((path, e.to_string(), version)));
                }
            }
            worker_window.request_redraw();
        }
    });

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
            while let Ok(res) = rx_done.try_recv() {
                match res {
                    Ok((path, rgba, d, version)) => {
                        if version == load_version.load(Ordering::Relaxed) {
                            if let Err(e) = unsafe { vk.upload_texture(path, rgba, d) } {
                                eprintln!("Upload error: {:?}", e);
                                window.set_title(&format!("rviv - [Error] {}", images[current_idx].display()));
                            } else {
                                dims = d;
                                window.set_title(&format!("rviv - {}", images[current_idx].display()));
                                let (ww, wh) = (vk.extent.width as f32, vk.extent.height as f32);
                                zoom = (ww / dims.0 as f32).min(wh / dims.1 as f32).min(1.0);
                                offset = [0.0, 0.0];
                            }
                            window.request_redraw();
                        }
                    }
                    Err((path, err, version)) => {
                        if version == load_version.load(Ordering::Relaxed) {
                            eprintln!("Failed to load {}: {}", path.display(), err);
                            window.set_title(&format!("rviv - [Error] {}", path.display()));
                        }
                    }
                }
            }

            elwt.set_control_flow(if cli.slideshow > 0.0 {
                ControlFlow::WaitUntil(Instant::now() + Duration::from_secs_f32(cli.slideshow))
            } else {
                ControlFlow::Wait
            });
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(s) => {
                        if s.width > 0 && s.height > 0 {
                            unsafe {
                                if let Err(e) = vk.recreate_swapchain(s.width, s.height) {
                                    eprintln!("Swapchain error: {:?}", e);
                                }
                            }
                        }
                    }
                    WindowEvent::ModifiersChanged(m) => modifiers = m,
                    WindowEvent::RedrawRequested => {
                        if needs_up {
                            let path = &images[current_idx];
                            if let Some(d) = unsafe { vk.check_cache(path) } {
                                dims = d;
                                needs_up = false;
                                window.set_title(&format!("rviv - {}", path.display()));
                                let (ww, wh) = (vk.extent.width as f32, vk.extent.height as f32);
                                zoom = (ww / dims.0 as f32).min(wh / dims.1 as f32).min(1.0);
                                offset = [0.0, 0.0];
                            } else {
                                let version = load_version.fetch_add(1, Ordering::Relaxed) + 1;
                                let _ = tx_load.send((path.clone(), version));
                                needs_up = false;
                                window.set_title(&format!("rviv - [Loading] {}", path.display()));
                            }
                        }
                        let (ww, wh) = (vk.extent.width as f32, vk.extent.height as f32);
                        let (sx, sy) = ((ww / dims.0 as f32) / zoom, (wh / dims.1 as f32) / zoom);
                        unsafe {
                            if let Err(e) = vk.draw(
                                PushConst {
                                    offset: [
                                        0.5 * (1.0 - sx) - offset[0] * sx,
                                        0.5 * (1.0 - sy) - offset[1] * sy,
                                    ],
                                    scale: [sx, sy],
                                },
                                [0.05, 0.05, 0.05, 1.0],
                            ) {
                                if e.to_string().contains("OUT_OF_DATE") {
                                } else {
                                    eprintln!("Draw error: {:?}", e);
                                }
                            }
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
        })?;
    Ok(())
}
