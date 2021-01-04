use std::fs;

use crate::state;

use image::GenericImageView;

use wgpu::{ProgrammableStageDescriptor, util::DeviceExt};

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
    std::env::current_exe()
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
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Sprite {
    pos: [f32; 2],
    size: [f32; 2],
}

unsafe impl bytemuck::Zeroable for Sprite {}
unsafe impl bytemuck::Pod for Sprite {}

#[derive(Copy, Clone, Debug)]
pub struct TextureId {
    inner_id: usize,
}

const SPRITEBATCH_BUFFER_STARTING_SIZE: u64 = 64;

struct SpriteBatch {
    label: String,
    sprite_buffer: wgpu::Buffer,
    sprite_buffer_capacity: u64,
    sprite_count: u64,
    vertex_buffer: wgpu::Buffer,
    vertex_buffer_capacity: u64,
    render_bind_group: wgpu::BindGroup,
    compute_bind_group: wgpu::BindGroup,
}

struct SpriteBatchDrawer {
    sprites: Vec<Sprite>,
    batch: Box<SpriteBatch>,
}

impl SpriteBatch {
    fn new(
        texture: &image::DynamicImage,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_bind_group_layout: &wgpu::BindGroupLayout,
        compute_bind_group_layout: &wgpu::BindGroupLayout,
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

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&("Vertex buffer for ".to_owned() + label)),
            size: SPRITEBATCH_BUFFER_STARTING_SIZE * std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::STORAGE,
            mapped_at_creation: false,
        });

        let sprite_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&("Sprite buffer for ".to_owned() + label)),
            size: SPRITEBATCH_BUFFER_STARTING_SIZE * std::mem::size_of::<Sprite>() as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some(label),
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(sprite_buffer.slice(..)),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(vertex_buffer.slice(..)),
                }
            ],
            label: Some(label),
        });

        SpriteBatch {
            vertex_buffer,
            vertex_buffer_capacity: SPRITEBATCH_BUFFER_STARTING_SIZE,
            sprite_buffer,
            sprite_buffer_capacity: SPRITEBATCH_BUFFER_STARTING_SIZE,
            sprite_count: 0,
            render_bind_group,
            compute_bind_group,
            label: label.to_owned(),
        }
    }

    fn begin(self) -> SpriteBatchDrawer {
        debug_assert!(self.vertex_buffer_capacity <= (usize::MAX as u64));

        SpriteBatchDrawer {
            sprites: Vec::with_capacity(self.sprite_buffer_capacity as usize),
            batch: Box::new(self),
        }
    }
}

impl<'renderer> SpriteBatchDrawer {
    fn draw(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32, g: f32, b: f32) {
        self.sprites.push(Sprite {
            pos: [x, y],
            size: [w, h],
        });
    }

    fn finish(mut self, renderer: &Renderer) -> SpriteBatch {
        let sprite_count = self.sprites.len() as u64;
        let vertex_count = sprite_count * 6;

        let recreate_compute_bind_group = sprite_count > self.batch.sprite_buffer_capacity || vertex_count > self.batch.vertex_buffer_capacity;

        // Need to grow the vertex buffer to fit.
        if vertex_count > self.batch.vertex_buffer_capacity {
            log::debug!(
                "Resizing spritebatch vertex buffer from {} elements to {} ({} bytes)",
                self.batch.vertex_buffer_capacity,
                vertex_count,
                vertex_count * std::mem::size_of::<Vertex>() as u64
            );

            self.batch.vertex_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&("Vertex buffer for ".to_owned() + &self.batch.label)),
                size: vertex_count * std::mem::size_of::<Vertex>() as u64,
                usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::STORAGE,
                mapped_at_creation: false,
            });

            self.batch.vertex_buffer_capacity = vertex_count
        }

        if sprite_count > self.batch.sprite_buffer_capacity {
            log::debug!(
                "Resizing spritebatch sprite buffer from {} elements to {} ({} bytes)",
                self.batch.sprite_buffer_capacity,
                sprite_count,
                sprite_count * std::mem::size_of::<Sprite>() as u64
            );

            self.batch.sprite_buffer =
                renderer
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&("Sprite buffer for ".to_owned() + &self.batch.label)),
                        contents: bytemuck::cast_slice(&self.sprites),
                        usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                    });

            self.batch.sprite_buffer_capacity = sprite_count;
        } else {
            log::trace!("Writing {} sprites to spritebatch buffer", sprite_count);
            renderer.queue.write_buffer(
                &self.batch.sprite_buffer,
                0,
                bytemuck::cast_slice(&self.sprites),
            );
        }

        if recreate_compute_bind_group {
            self.batch.compute_bind_group = renderer.device.create_bind_group( &wgpu::BindGroupDescriptor {
                layout: &renderer.spritebatch_compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(self.batch.sprite_buffer.slice(..)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(self.batch.vertex_buffer.slice(..)),
                    }
                ],
                label: Some(&self.batch.label),
            })
        }

        self.batch.sprite_count = sprite_count;

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
    spritebatch_compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    spritebatch_render_bind_group_layout: wgpu::BindGroupLayout,
    spritebatch_compute_bind_group_layout: wgpu::BindGroupLayout,
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

        let mut cs_path = asset_path.clone();
        cs_path.push("shaders");
        cs_path.push("main.compute.spv");
        let cs_bytes = fs::read(cs_path).expect("could not read main.compute.spv");
        let cs_module_src = wgpu::util::make_spirv(&cs_bytes[..]);
        let cs_module = device.create_shader_module(cs_module_src);

        let transform_ref: &[f32; 16] = camera_transform.as_ref();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(transform_ref),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let spritebatch_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: false },
                        count: None,
                    },
                ],
                label: Some("Spritebatch bind group layout"),
            });

        let main_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer {
                    dynamic: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<cgmath::Matrix4<f32>>() as _,
                    ),
                },
                count: None,
            }],
            label: Some("Main bind group layout"),
        });

        let compute_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            readonly: true,
                            dynamic: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            readonly: false,
                            dynamic: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Compute bind group layout"),
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
            bind_group_layouts: &[
                &main_bind_layout,
                &spritebatch_bind_layout,
            ],
            push_constant_ranges: &[],
            label: None,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: Some(&pipeline_layout),

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
                    ],
                }],
            },
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_clamp: 0.0,
                depth_bias_slope_scale: 0.0,
                ..Default::default()
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
            label: None,
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spritebatch compute pipeline layout"),
            bind_group_layouts: &[ &compute_bind_layout ],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Spritebatch compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            compute_stage: ProgrammableStageDescriptor {
                module: &cs_module,
                entry_point: "main",
            },
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
            uniform_buffer,
            spritebatch_render_bind_group_layout: spritebatch_bind_layout,
            spritebatch_compute_bind_group_layout: compute_bind_layout,
            spritebatch_compute_pipeline: compute_pipeline,
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

        for (id, (transform, texture)) in game_state
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

            let (r, g, b) = if let Ok(tint) = game_state.world.get::<state::Tint>(id) {
                (tint.r, tint.g, tint.b)
            } else {
                (1.0, 1.0, 1.0)
            };

            let (w, h) = if let Ok(size) = game_state.world.get::<state::Size>(id) {
                (size.x, size.y)
            } else {
                (16.0, 16.0)
            };

            drawer.draw(transform.x, transform.y, w, h, r, g, b);
        }

        for drawer in drawers {
            self.spritebatches.push(Box::new(drawer.finish(&self)));
        }
    }

    pub fn draw(&mut self) {
        let frame = self.swap_chain.get_current_frame().unwrap().output;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        
        {
            let mut pass = encoder.begin_compute_pass();
            pass.set_pipeline(&self.spritebatch_compute_pipeline);
            
            for texture_idx in 0..self.spritebatches.len() {
                let spritebatch = self.spritebatches.get(texture_idx).unwrap();
                pass.set_bind_group(0, &spritebatch.compute_bind_group, &[]);
                pass.dispatch(spritebatch.sprite_count as u32, 1, 1);
            }
        }

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

            for texture_idx in 0..self.spritebatches.len() {
                let spritebatch = self.spritebatches.get(texture_idx).unwrap();
                pass.set_bind_group(1, &spritebatch.render_bind_group, &[]);
                pass.set_vertex_buffer(0, spritebatch.vertex_buffer.slice(..));
                pass.draw(0..(spritebatch.sprite_count * 6) as u32, 0..1);
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
            &self.spritebatch_render_bind_group_layout,
            &self.spritebatch_compute_bind_group_layout,
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
