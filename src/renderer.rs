use std::fs;

use crate::state;

use image::GenericImageView;

#[cfg(debug_assertions)]
fn asset_base_path() -> std::path::PathBuf {
    let exe_path = std::env::current_exe().unwrap();
    let mut current_path = exe_path.parent();

    while let Some(path) = current_path {
        let manifest_path = path.join("Cargo.toml");

        match fs::metadata(manifest_path) {
            Ok(_) => return path.join("resources.out"),

            Err(err) => {
                if err.kind() == std::io::ErrorKind::NotFound {
                    current_path = path.parent();
                } else {
                    panic!("{}", err);
                }
            }
        }
    }

    panic!("could not find resources folder");
}

#[cfg(not(debug_assertions))]
fn asset_base_path() -> std::path::PathBuf {
    env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("resources")
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    size_mask: [f32; 2],
}

impl Vertex {
    fn new(x: f32, y: f32, u: f32, v: f32, w_mask: f32, h_mask: f32) -> Self {
        Self {
            pos: [x, y],
            uv: [u, v],
            size_mask: [w_mask, h_mask],
        }
    }
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

#[derive(Copy, Clone, Debug)]
pub struct TextureId {
    inner_id: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Instance {
    position: [f32; 2],
    size: [f32; 2],
}

impl Instance {
    fn new(x: f32, y: f32, w: f32, h: f32) -> Instance {
        Instance {
            position: [x, y],
            size: [w, h],
        }
    }
}

unsafe impl bytemuck::Zeroable for Instance {}
unsafe impl bytemuck::Pod for Instance {}

fn create_sprite_vertices() -> (Vec<Vertex>, Vec<u16>) {
    (
        vec![
            Vertex::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vertex::new(1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
            Vertex::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            Vertex::new(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        ],
        vec![0, 1, 2, 2, 3, 0],
    )
}

const SPRITEBATCH_INSTANCE_BUFFER_STARTING_SIZE: u64 = 64;

struct SpriteBatch {
    label: String,
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    instance_buffer: wgpu::Buffer,
    instance_buffer_capacity: u64,
    instance_count: u64,
    bind_group: wgpu::BindGroup,
}

struct SpriteBatchDrawer {
    instances: Vec<Instance>,
    batch: Box<SpriteBatch>,
}

impl SpriteBatch {
    fn new(
        texture: &image::DynamicImage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        label: &str,
    ) -> SpriteBatch {
        let bytes = texture.as_rgba8().unwrap();
        let (width, height) = texture.dimensions();
        let extents = wgpu::Extent3d {
            width,
            height,
            depth: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: extents,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: Some(label),
        });

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &bytes,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * width,
                rows_per_image: height,
            },
            extents,
        );

        let view = texture.create_default_view();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&("Instance storage buffer for ".to_owned() + label)),
            size: SPRITEBATCH_INSTANCE_BUFFER_STARTING_SIZE
                * std::mem::size_of::<Instance>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(instance_buffer.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some(label),
        });

        SpriteBatch {
            texture_view: view,
            sampler,
            instance_buffer,
            instance_buffer_capacity: SPRITEBATCH_INSTANCE_BUFFER_STARTING_SIZE,
            instance_count: 0,
            bind_group,
            label: label.to_owned(),
        }
    }

    fn begin(self) -> SpriteBatchDrawer {
        debug_assert!(self.instance_buffer_capacity <= (usize::MAX as u64));

        SpriteBatchDrawer {
            instances: Vec::with_capacity(self.instance_buffer_capacity as usize),
            batch: Box::new(self),
        }
    }
}

impl<'renderer> SpriteBatchDrawer {
    fn draw(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.instances.push(Instance::new(x, y, w, h));
    }

    fn finish(mut self, renderer: &Renderer) -> SpriteBatch {
        let instance_count = self.instances.len() as u64;

        // Need to grow the instance buffer to fit.
        if instance_count > self.batch.instance_buffer_capacity {
            log::debug!(
                "Resizing spritebatch instance buffer from {} elements to {} ({} bytes)",
                self.batch.instance_buffer_capacity,
                instance_count,
                instance_count * std::mem::size_of::<Instance>() as u64
            );

            self.batch.instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&("Instance storage buffer for ".to_owned() + &self.batch.label)),
                size: instance_count * std::mem::size_of::<Instance>() as u64,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });

            // After recreating the instance buffer we need to remake the bind group
            self.batch.bind_group = renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &renderer.spritebatch_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(
                                self.batch.instance_buffer.slice(..),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.batch.texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.batch.sampler),
                        },
                    ],
                    label: Some(&self.batch.label),
                });
        }

        log::trace!(
            "Writing {} instances to spritebatch instance buffer",
            instance_count
        );

        renderer.queue.write_buffer(
            &self.batch.instance_buffer,
            0,
            bytemuck::cast_slice(&self.instances),
        );

        self.batch.instance_count = instance_count;

        *self.batch
    }
}

fn compute_camera_transform(width: u32, height: u32) -> cgmath::Matrix4<f32> {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    );

    let ortho_matrix = cgmath::ortho(0.0, width as f32, height as f32, 0.0, 0.0, 1.0);
    OPENGL_TO_WGPU_MATRIX * ortho_matrix
}

pub struct Renderer {
    camera_transform: cgmath::Matrix4<f32>,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain: wgpu::SwapChain,
    swap_chain_descriptor: wgpu::SwapChainDescriptor,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    spritebatch_bind_layout: wgpu::BindGroupLayout,
    indices: Vec<u16>,
    spritebatches: Vec<Box<SpriteBatch>>,
}

impl Renderer {
    pub async fn new(window: &winit::window::Window) -> Renderer {
        let asset_path = asset_base_path();

        let size = window.inner_size();
        let camera_transform = compute_camera_transform(size.width, size.height);
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await
            .unwrap();

        let (vertices, indices) = create_sprite_vertices();
        let vertex_buffer = device
            .create_buffer_with_data(bytemuck::cast_slice(&vertices), wgpu::BufferUsage::VERTEX);

        let index_buffer = device
            .create_buffer_with_data(bytemuck::cast_slice(&indices), wgpu::BufferUsage::INDEX);

        let mut vs_path = asset_path.clone();
        vs_path.push("shaders");
        vs_path.push("main.vert.spv");
        let vs_bytes = fs::read(vs_path).expect("could not read main.vert.spv");
        let vs_module_src = wgpu::util::make_spirv(&vs_bytes[..]);
        let vs_module = device.create_shader_module(vs_module_src);

        let mut fs_path = asset_path.clone();
        fs_path.push("shaders");
        fs_path.push("main.frag.spv");
        let fs_bytes = fs::read(fs_path).expect("could not read main.frag.spv");
        let fs_module_src = wgpu::util::make_spirv(&fs_bytes[..]);
        let fs_module = device.create_shader_module(fs_module_src);

        let transform_ref: &[f32; 16] = camera_transform.as_ref();
        let uniform_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(transform_ref),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let spritebatch_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry::new(
                        0,
                        wgpu::ShaderStage::VERTEX,
                        wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                            min_binding_size: wgpu::BufferSize::new(
                                SPRITEBATCH_INSTANCE_BUFFER_STARTING_SIZE
                                    * std::mem::size_of::<Instance>() as u64,
                            ),
                        },
                    ),
                    wgpu::BindGroupLayoutEntry::new(
                        1,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Uint,
                        },
                    ),
                    wgpu::BindGroupLayoutEntry::new(
                        2,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::Sampler { comparison: false },
                    ),
                ],
                label: Some("Spritebatch bind group layout"),
            });

        let main_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry::new(
                0,
                wgpu::ShaderStage::VERTEX,
                wgpu::BindingType::UniformBuffer {
                    dynamic: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<cgmath::Matrix4<f32>>() as _,
                    ),
                },
            )],
            label: Some("Main bind group layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &main_bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(uniform_buffer.slice(..)),
            }],
            label: Some("Main bind group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&main_bind_layout, &spritebatch_bind_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,

            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),

            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            shader_location: 1,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_clamp: 0.0,
                depth_bias_slope_scale: 0.0,
            }),
            depth_stencil_state: None,
            color_states: &[wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                    ..Default::default()
                },
                alpha_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                    ..Default::default()
                },
                write_mask: wgpu::ColorWrite::ALL,
            }],

            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let swap_chain_descriptor = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };

        let swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

        Renderer {
            surface,
            swap_chain,
            swap_chain_descriptor,
            queue,
            device,
            camera_transform,
            pipeline: render_pipeline,
            bind_group,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            spritebatch_bind_layout,
            indices,
            spritebatches: vec![],
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.swap_chain_descriptor.width = new_size.width;
        self.swap_chain_descriptor.height = new_size.height;

        log::debug!(
            "Recreating swapchain with new dimensions {}x{}.",
            new_size.width,
            new_size.height
        );

        self.swap_chain = self
            .device
            .create_swap_chain(&self.surface, &self.swap_chain_descriptor);
        self.camera_transform = compute_camera_transform(new_size.width, new_size.height);
        let transform_ref: &[f32; 16] = self.camera_transform.as_ref();
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(transform_ref));
    }

    pub fn prepare_draw(&mut self, game_state: &crate::state::GameState) {
        let mut drawers: Vec<SpriteBatchDrawer> = self
            .spritebatches
            .drain(..)
            .map(|batch| batch.begin())
            .collect();

        for (_id, (transform, texture)) in game_state
            .world
            .query::<(&state::Transform, &state::Texture)>()
            .into_iter()
        {
            debug_assert!(
                texture.0.inner_id < drawers.len(),
                "Texture ID {} is invalid for this Renderer (value out of range)",
                texture.0.inner_id
            );
            let drawer = &mut drawers[texture.0.inner_id];

            log::trace!(
                "Particle at {}, {} with texture ID {}",
                transform.x,
                transform.y,
                texture.0.inner_id,
            );

            drawer.draw(transform.x, transform.y, 16.0, 16.0);
        }

        for drawer in drawers {
            self.spritebatches.push(Box::new(drawer.finish(&self)));
        }
    }

    pub fn draw(&mut self) {
        log::trace!("Rendering");
        let frame = self.swap_chain.get_next_frame().unwrap().output;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..));

            for texture_idx in 0..self.spritebatches.len() {
                let spritebatch = self.spritebatches.get(texture_idx).unwrap();
                pass.set_bind_group(1, &spritebatch.bind_group, &[]);
                pass.draw_indexed(
                    0..self.indices.len() as u32,
                    0,
                    0..spritebatch.instance_count as u32,
                );
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn load_texture(&mut self, path_fragment: &str) -> Result<TextureId, std::io::Error> {
        let base_path = asset_base_path();
        let asset_path = base_path.join(path_fragment);
        let bytes = fs::read(asset_path).unwrap();
        let image = image::load_from_memory(&bytes).expect("Could not load image");

        let spritebatch = SpriteBatch::new(
            &image,
            &self.device,
            &self.queue,
            &self.spritebatch_bind_layout,
            path_fragment,
        );

        let insertion_index = self.spritebatches.len();
        self.spritebatches.push(Box::new(spritebatch));
        log::trace!("Loaded texture for {}", path_fragment);

        Ok(TextureId {
            inner_id: insertion_index,
        })
    }
}
