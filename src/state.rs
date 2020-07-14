use std::time::Instant;

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
}

impl GameState {
    pub fn new() -> GameState {
        GameState {
            world: hecs::World::new(),
            current_time: Instant::now(),
        }
    }

    pub fn step(&mut self, delta: f32) {
        for (_id, (transform, velocity)) in &mut self.world.query::<(&mut Transform, &Velocity)>() {
            transform.x += velocity.x_velocity * delta;
            transform.y += velocity.y_velocity * delta;
            transform.rotation += velocity.rotation * delta;
        }
    }
}
