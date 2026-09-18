use std::borrow::Cow;
use std::path::{Path, PathBuf};
use crate::error::BubblePopError;

/// Supported pixel memory layout formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 32-bit RGBA (4 bytes per pixel: Red, Green, Blue, Alpha).
    Rgba8,
    /// 24-bit RGB (3 bytes per pixel: Red, Green, Blue).
    Rgb8,
    /// 32-bit BGRA (4 bytes per pixel: Blue, Green, Red, Alpha).
    Bgra8,
    /// 24-bit BGR (3 bytes per pixel: Blue, Green, Red).
    Bgr8,
    /// 32-bit ARGB (4 bytes per pixel: Alpha, Red, Green, Blue).
    Argb8,
    /// 8-bit Grayscale (1 byte per pixel).
    Gray8,
}

impl PixelFormat {
    #[inline]
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::Rgba8 | Self::Bgra8 | Self::Argb8 => 4,
            Self::Rgb8 | Self::Bgr8 => 3,
            Self::Gray8 => 1,
        }
    }
}

/// Decoded image view ready for letterbox preprocessing.
pub struct ImageSource<'a> {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Cow<'a, [u8]>,
    pub stride_bytes: usize,
}

fn decode_memory<'a>(bytes: &[u8]) -> Result<ImageSource<'a>, BubblePopError> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| BubblePopError::ImageDecodeError(e.to_string()))?;
    let rgba = img.into_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ImageSource {
        width,
        height,
        format: PixelFormat::Rgba8,
        stride_bytes: (width * 4) as usize,
        data: Cow::Owned(rgba.into_raw()),
    })
}

fn decode_file<'a, P: AsRef<Path>>(path: P) -> Result<ImageSource<'a>, BubblePopError> {
    let p = path.as_ref();
    let bytes = std::fs::read(p)
        .map_err(|e| BubblePopError::IoError(e, p.display().to_string()))?;
    decode_memory(&bytes)
}

/// Trait for types that can be converted into an image source.
pub trait IntoImageSource<'a> {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError>;
}

impl<'a> IntoImageSource<'a> for &'a str {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_file(self)
    }
}

impl<'a> IntoImageSource<'a> for &'a String {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_file(self.as_str())
    }
}

impl<'a> IntoImageSource<'a> for String {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_file(&self)
    }
}

impl<'a> IntoImageSource<'a> for &'a Path {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_file(self)
    }
}

impl<'a> IntoImageSource<'a> for PathBuf {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_file(&self)
    }
}

impl<'a> IntoImageSource<'a> for &'a PathBuf {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_file(self.as_path())
    }
}

impl<'a> IntoImageSource<'a> for &'a [u8] {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_memory(self)
    }
}

impl<'a> IntoImageSource<'a> for Vec<u8> {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_memory(&self)
    }
}

impl<'a> IntoImageSource<'a> for &'a Vec<u8> {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        decode_memory(self.as_slice())
    }
}

impl<'a> IntoImageSource<'a> for image::DynamicImage {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        let rgba = self.into_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(ImageSource {
            width,
            height,
            format: PixelFormat::Rgba8,
            stride_bytes: (width * 4) as usize,
            data: Cow::Owned(rgba.into_raw()),
        })
    }
}

impl<'a> IntoImageSource<'a> for &'a image::DynamicImage {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        match self {
            image::DynamicImage::ImageRgba8(rgba) => rgba.into_image_source(),
            other => {
                let rgba = other.to_rgba8();
                let (width, height) = rgba.dimensions();
                Ok(ImageSource {
                    width,
                    height,
                    format: PixelFormat::Rgba8,
                    stride_bytes: (width * 4) as usize,
                    data: Cow::Owned(rgba.into_raw()),
                })
            }
        }
    }
}

impl<'a> IntoImageSource<'a> for image::RgbaImage {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        let (width, height) = self.dimensions();
        Ok(ImageSource {
            width,
            height,
            format: PixelFormat::Rgba8,
            stride_bytes: (width * 4) as usize,
            data: Cow::Owned(self.into_raw()),
        })
    }
}

impl<'a> IntoImageSource<'a> for &'a image::RgbaImage {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        let (width, height) = self.dimensions();
        Ok(ImageSource {
            width,
            height,
            format: PixelFormat::Rgba8,
            stride_bytes: (width * 4) as usize,
            data: Cow::Borrowed(self.as_raw()),
        })
    }
}

impl<'a> IntoImageSource<'a> for ImageSource<'a> {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        Ok(self)
    }
}

impl<'a> IntoImageSource<'a> for &'a ImageSource<'a> {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        Ok(ImageSource {
            width: self.width,
            height: self.height,
            format: self.format,
            stride_bytes: self.stride_bytes,
            data: match &self.data {
                Cow::Borrowed(b) => Cow::Borrowed(*b),
                Cow::Owned(v) => Cow::Borrowed(v.as_slice()),
            },
        })
    }
}

/// View of a raw pixel buffer in memory without heap allocation.
pub struct RawPixelView<'a> {
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub stride_bytes: Option<usize>,
}

impl<'a> IntoImageSource<'a> for RawPixelView<'a> {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        let bpp = self.format.bytes_per_pixel();
        let stride = self.stride_bytes.unwrap_or(self.width as usize * bpp);
        Ok(ImageSource {
            width: self.width,
            height: self.height,
            format: self.format,
            stride_bytes: stride,
            data: Cow::Borrowed(self.pixels),
        })
    }
}

impl<'a> IntoImageSource<'a> for &'a RawPixelView<'a> {
    fn into_image_source(self) -> Result<ImageSource<'a>, BubblePopError> {
        let bpp = self.format.bytes_per_pixel();
        let stride = self.stride_bytes.unwrap_or(self.width as usize * bpp);
        Ok(ImageSource {
            width: self.width,
            height: self.height,
            format: self.format,
            stride_bytes: stride,
            data: Cow::Borrowed(self.pixels),
        })
    }
}
