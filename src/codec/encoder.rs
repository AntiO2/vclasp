use std::ptr::NonNull;

/// One IDR plus at most sixteen directly Anchor-referenced P frames. Stock
/// x264 caps the decoded-picture/reference buffer at `X264_REF_MAX == 16`.
pub const MAX_ANCHOR_P_GROUP_FRAMES: u32 = 17;

#[repr(C)]
pub struct VClaspX264EncoderOpaque([u8; 0]);

extern "C" {
    fn vclasp_encoder_open(
        width: i32,
        height: i32,
        crf: i32,
        gop_size: i32,
        anchor_p: i32,
    ) -> *mut VClaspX264EncoderOpaque;

    fn vclasp_encoder_open_two_level(
        width: i32,
        height: i32,
        crf: i32,
        gop_size: i32,
        page_size: i32,
    ) -> *mut VClaspX264EncoderOpaque;

    fn vclasp_encoder_open_page_session(
        width: i32,
        height: i32,
        crf: i32,
        keyint: i32,
    ) -> *mut VClaspX264EncoderOpaque;

    fn vclasp_encoder_encode_frame(
        enc: *mut VClaspX264EncoderOpaque,
        yuv: *const u8,
        is_idr: i32,
        out_buf: *mut u8,
        out_buf_size: i32,
    ) -> i32;

    fn vclasp_encoder_flush(
        enc: *mut VClaspX264EncoderOpaque,
        out_buf: *mut u8,
        out_buf_size: i32,
    ) -> i32;

    fn vclasp_encoder_close(enc: *mut VClaspX264EncoderOpaque);
}

pub struct X264Encoder {
    inner: NonNull<VClaspX264EncoderOpaque>,
    width: i32,
    height: i32,
    yuv_frame_size: usize,
    out_buf_size: i32,
    anchor_p: bool,
}

impl X264Encoder {
    fn max_nal_size(width: i32, height: i32) -> i32 {
        let pixels = (width as u32) * (height as u32);
        (pixels * 4) as i32 + 4096
    }

    pub fn new(width: u32, height: u32, crf: u32, gop_size: u32, anchor_p: bool) -> Self {
        let w = width as i32;
        let h = height as i32;
        let yuv_sz = (w as usize) * (h as usize) * 3 / 2;

        let inner =
            unsafe { vclasp_encoder_open(w, h, crf as i32, gop_size as i32, anchor_p as i32) };
        let inner = NonNull::new(inner).expect("vclasp_encoder_open failed");

        Self {
            inner,
            width: w,
            height: h,
            yuv_frame_size: yuv_sz,
            out_buf_size: Self::max_nal_size(w, h),
            anchor_p,
        }
    }

    /// Build a root/checkpoint/target encoder. Each page checkpoint directly
    /// references the GOP root; targets may reference only that root and the
    /// current checkpoint. The page size must divide the GOP size.
    pub fn new_two_level(width: u32, height: u32, crf: u32, gop_size: u32, page_size: u32) -> Self {
        let w = width as i32;
        let h = height as i32;
        let yuv_sz = (w as usize) * (h as usize) * 3 / 2;
        let inner = unsafe {
            vclasp_encoder_open_two_level(w, h, crf as i32, gop_size as i32, page_size as i32)
        };
        let inner = NonNull::new(inner).expect("vclasp_encoder_open_two_level failed");
        Self {
            inner,
            width: w,
            height: h,
            yuv_frame_size: yuv_sz,
            out_buf_size: Self::max_nal_size(w, h),
            anchor_p: true,
        }
    }

    /// Build one independent object-page stream: root IDR, checkpoint P, then
    /// targets that may reference only those two retained pictures.
    pub fn new_page_session(width: u32, height: u32, crf: u32, keyint: u32) -> Self {
        let w = width as i32;
        let h = height as i32;
        let yuv_sz = (w as usize) * (h as usize) * 3 / 2;
        let inner = unsafe { vclasp_encoder_open_page_session(w, h, crf as i32, keyint as i32) };
        let inner = NonNull::new(inner).expect("vclasp_encoder_open_page_session failed");
        Self {
            inner,
            width: w,
            height: h,
            yuv_frame_size: yuv_sz,
            out_buf_size: Self::max_nal_size(w, h),
            anchor_p: true,
        }
    }

    pub fn encode_frame(&mut self, yuv: &[u8], is_idr: bool) -> Vec<u8> {
        self.try_encode_frame(yuv, is_idr)
            .expect("x264 frame encode failed")
    }

    pub fn try_encode_frame(&mut self, yuv: &[u8], is_idr: bool) -> Result<Vec<u8>, String> {
        assert_eq!(
            yuv.len(),
            self.yuv_frame_size,
            "YUV buffer must be exactly {} bytes ({}×{}×3/2)",
            self.yuv_frame_size,
            self.width,
            self.height,
        );

        let mut out = vec![0u8; self.out_buf_size as usize];
        let written = unsafe {
            vclasp_encoder_encode_frame(
                self.inner.as_ptr(),
                yuv.as_ptr(),
                is_idr as i32,
                out.as_mut_ptr(),
                self.out_buf_size,
            )
        };

        if written < 0 {
            return Err("x264 frame encode returned an error".to_string());
        }
        if written == 0 {
            return Ok(Vec::new());
        }

        out.truncate(written as usize);
        Ok(out)
    }

    pub fn flush(&mut self) -> Vec<u8> {
        self.try_flush().expect("x264 flush failed")
    }

    pub fn try_flush(&mut self) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; self.out_buf_size as usize];
        let written = unsafe {
            vclasp_encoder_flush(self.inner.as_ptr(), out.as_mut_ptr(), self.out_buf_size)
        };

        if written < 0 {
            return Err("x264 flush returned an error".to_string());
        }
        if written == 0 {
            return Ok(Vec::new());
        }

        out.truncate(written as usize);
        Ok(out)
    }

    pub fn width(&self) -> u32 {
        self.width as u32
    }

    pub fn height(&self) -> u32 {
        self.height as u32
    }

    pub fn is_anchor_p(&self) -> bool {
        self.anchor_p
    }
}

impl Drop for X264Encoder {
    fn drop(&mut self) {
        unsafe {
            vclasp_encoder_close(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for X264Encoder {}
