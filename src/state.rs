use std::time::Instant;

const EQUILIBRIUM_PARTICLE_COUNT: u32 = 10000;

use rand::distributions::{Distribution, Uniform};

#[derive(Copy, Clone, Debug)]
pub struct Transform {
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Velocity {
    pub x_velocity: f32,
    pub y_velocity: f32,
    pub rotation: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Texture(pub crate::renderer::TextureId);

pub struct GameState {
    pub world: hecs::World,
    pub current_time: Instant,
    pub window_size: (u32, u32),
    particle_texture: crate::renderer::TextureId,
}

impl GameState {
    pub fn new(window_size: (u32, u32), particle_texture: crate::renderer::TextureId) -> GameState {
        GameState {
            world: hecs::World::new(),
            current_time: Instant::now(),
            window_size,
            particle_texture,
        }
    }

    pub fn step(&mut self, delta: f32) {
        let mut out_of_bounds_entities = vec![];
        let mut entity_count = 0;

        for (id, (transform, velocity)) in &mut self.world.query::<(&mut Transform, &Velocity)>() {
            transform.x += velocity.x_velocity * delta;
            transform.y += velocity.y_velocity * delta;
            transform.rotation += velocity.rotation * delta;

            if transform.x < 0.0
                || transform.y < 0.0
                || transform.x > self.window_size.0 as _
                || transform.y > self.window_size.1 as _
            {
                out_of_bounds_entities.push(id);
            }

            entity_count += 1;
        }

        for out_of_bounds_entity in out_of_bounds_entities {
            self.world.despawn(out_of_bounds_entity).unwrap();
            entity_count -= 1;
        }

        if entity_count < EQUILIBRIUM_PARTICLE_COUNT {
            let pos_x_distribution = Uniform::from(10..self.window_size.0);
            let pos_y_distribution = Uniform::from(10..self.window_size.1);
            let velocity_distribution = Uniform::from(-100.0..100.0);
            let mut rng = rand::thread_rng();

            let particle_texture = self.particle_texture;

            self.world
                .spawn_batch((entity_count..=EQUILIBRIUM_PARTICLE_COUNT).map(|_i| {
                    (
                        Texture(particle_texture),
                        Transform {
                            x: pos_x_distribution.sample(&mut rng) as _,
                            y: pos_y_distribution.sample(&mut rng) as _,
                            rotation: 0.0,
                        },
                        Velocity {
                            x_velocity: velocity_distribution.sample(&mut rng) as _,
                            y_velocity: velocity_distribution.sample(&mut rng) as _,
                            rotation: 0.0,
                        },
                    )
                }));
        }
    }
}
