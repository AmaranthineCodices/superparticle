use std::time::Instant;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod renderer;
mod state;

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut renderer = renderer::Renderer::new(&window).await;
    let test_tex_id = renderer.load_texture("textures/test_particle.png").unwrap();

    let mut game_state = state::GameState::new(
        (window.inner_size().width, window.inner_size().height),
        test_tex_id,
    );

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                renderer.resize(new_size);
            }
            Event::RedrawRequested(_) => {
                let now = Instant::now();
                let dt = (now - game_state.current_time).as_secs_f32();
                game_state.current_time = now;
                game_state.step(dt);
                renderer.prepare_render(&game_state);
                renderer.draw();
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {
    pretty_env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Test Window")
        .build(&event_loop)
        .unwrap();

    futures::executor::block_on(run(event_loop, window));
}
