use std::time::Instant;

pub const EQUILIBRIUM_PARTICLE_COUNT: usize = 100_000;

use rand::distributions::{Distribution, Uniform};
use rand::Rng;

// All slices have length EQUILIBRIUM_PARTICLE_COUNT.
pub struct Particles {
    pub positions: Box<[(f32, f32)]>,
    pub velocities: Box<[(f32, f32)]>,
    pub colors: Box<[(f32, f32, f32)]>,
    pub scales: Box<[f32]>,
}
pub struct GameState {
    pub particles: Particles,
    pub current_time: Instant,
    pub window_size: (u32, u32),
}

fn init_random_particles(particles: &mut Particles, start_index: usize, end_index: usize, window_size: (u32, u32)) {
    let pos_x_distribution = Uniform::from(10..window_size.0);
    let pos_y_distribution = Uniform::from(10..window_size.1);
    let velocity_distribution = Uniform::from(-100.0..100.0);
    let size_distribution = Uniform::from(1.0..2.0);
    let mut rng = rand::thread_rng();

    for i in start_index..end_index {
        particles.positions[i] = (pos_x_distribution.sample(&mut rng) as _, pos_y_distribution.sample(&mut rng) as _);
        particles.velocities[i] = (velocity_distribution.sample(&mut rng), velocity_distribution.sample(&mut rng));
        particles.colors[i] = (rng.gen(), rng.gen(), rng.gen());
        particles.scales[i] = size_distribution.sample(&mut rng);
    }
}

impl GameState {
    pub fn new(window_size: (u32, u32)) -> GameState {
        let mut state = GameState {
            particles: Particles {
                // Hack: Ideally we would not construct an intermediate vector. We need placement new
                // aka `box` syntax to avoid this. Allocating large arrays on the stack will cause an
                // overflow; this is the only way that I can find to allocate on the heap from the
                // start, instead of allocating on the stack and then copying to the heap.
                positions: vec![(0.0, 0.0); EQUILIBRIUM_PARTICLE_COUNT].into_boxed_slice(),
                velocities: vec![(0.0, 0.0); EQUILIBRIUM_PARTICLE_COUNT].into_boxed_slice(),
                colors: vec![(0.0, 0.0, 0.0); EQUILIBRIUM_PARTICLE_COUNT].into_boxed_slice(),
                scales: vec![0.0; EQUILIBRIUM_PARTICLE_COUNT].into_boxed_slice(),
            },
            current_time: Instant::now(),
            window_size,
        };

        init_random_particles(&mut state.particles, 0, EQUILIBRIUM_PARTICLE_COUNT, window_size);
        state
    }

    pub fn step(&mut self, delta: f32) {
        for i in 0..EQUILIBRIUM_PARTICLE_COUNT {
            let (x, y) = self.particles.positions[i];
            let (vx, vy) = self.particles.velocities[i];

            let (x, y) = (x + vx * delta, y + vy * delta);

            if x < 0.0 || x > self.window_size.0 as f32 || y < 0.0 || y > self.window_size.1 as f32 {
                init_random_particles(&mut self.particles, i, i, self.window_size);
            } else {
                self.particles.positions[i] = (x, y);
            }
        }
    }
}
