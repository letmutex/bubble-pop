pub const INPUT_WIDTH: usize = 768;
pub const INPUT_HEIGHT: usize = 1024;
pub const MEAN: f32 = 0.449;
pub const STD: f32 = 0.226;

#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    pub scale: f32,
    pub resized_width: usize,
    pub resized_height: usize,
    pub pad_x: usize,
    pub pad_y: usize,
}

impl Letterbox {
    pub fn new(source_width: usize, source_height: usize) -> Self {
        let scale = (INPUT_WIDTH as f32 / source_width as f32)
            .min(INPUT_HEIGHT as f32 / source_height as f32);
        let resized_width = (source_width as f32 * scale).round().max(1.0) as usize;
        let resized_height = (source_height as f32 * scale).round().max(1.0) as usize;
        let pad_x = (INPUT_WIDTH - resized_width) / 2;
        let pad_y = (INPUT_HEIGHT - resized_height) / 2;

        Self {
            scale,
            resized_width,
            resized_height,
            pad_x,
            pad_y,
        }
    }
}
