#include <stdint.h>
#include <x264.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "x264_encoder.h"

struct x264_encoder_s {
    x264_t        *h;
    int            width;
    int            height;
    int            anchor_p;
    int            frame_count;
    int            gop_size;
    int            page_size;
    int64_t        checkpoint_pts;
};

static void setup_picture(x264_picture_t *pic, const uint8_t *yuv,
                           int width, int height, int is_idr, int non_reference_p,
                           int64_t pts)
{
    x264_picture_init(pic);
    pic->img.i_csp = X264_CSP_I420;
    pic->img.i_plane = 3;
    pic->img.i_stride[0] = width;
    pic->img.i_stride[1] = width / 2;
    pic->img.i_stride[2] = width / 2;
    pic->img.plane[0] = (uint8_t*)yuv;
    pic->img.plane[1] = (uint8_t*)yuv + (size_t)width * height;
    pic->img.plane[2] = (uint8_t*)yuv + (size_t)width * height * 5 / 4;
    pic->i_type = is_idr ? X264_TYPE_IDR : X264_TYPE_P;
    pic->i_pts  = pts;
#ifdef VCLASP_PATCHED_X264
    /* Private input marker consumed by the pinned x264 patch. The emitted
     * slice remains an ordinary H.264 P slice with nal_ref_idc=0. */
    pic->opaque = non_reference_p ? (void*)(intptr_t)0x5654574c : NULL;
#else
    (void)non_reference_p;
#endif
}

/* ------------------------------------------------------------------ */
/*  Public API                                                          */
/* ------------------------------------------------------------------ */

static VClaspX264Encoder* vclasp_encoder_open_common(int width, int height, int crf,
                                               int gop_size, int reference_mode,
                                               int page_size)
{
    VClaspX264Encoder *enc = calloc(1, sizeof(VClaspX264Encoder));
    if (!enc) return NULL;

    enc->width       = width;
    enc->height      = height;
    enc->anchor_p    = reference_mode;
    enc->frame_count = 0;
    enc->gop_size    = gop_size;
    enc->page_size   = page_size;
    enc->checkpoint_pts = -1;

    x264_param_t param;
    x264_param_default_preset(&param, "veryfast", NULL);

    param.i_width        = width;
    param.i_height       = height;
    param.i_csp          = X264_CSP_I420;
    param.i_bitdepth     = 8;

    /* invalidate_reference is an interactive-error-resilience API. x264's
     * contract recommends a very large keyframe interval so that invalidated
     * short-term references do not trigger unrequested recovery IDRs. Callers
     * still delimit logical GOPs explicitly with X264_TYPE_IDR. */
    int keyint = reference_mode ? 1000000 : gop_size;
    param.i_keyint_max   = keyint;
    param.i_keyint_min   = keyint;
    param.i_scenecut_threshold = 0;

    param.i_bframe       = 0;
    param.i_bframe_adaptive = X264_B_ADAPT_NONE;
    param.i_bframe_pyramid = X264_B_PYRAMID_NONE;

    /* x264 updates its short-term reference list before the caller can
     * invalidate the just-encoded target. A two-entry list can therefore
     * evict the old root while briefly holding root/checkpoint/target. Keep
     * the maximum legal short-term capacity; invalidation still restricts the
     * references visible to the next encode to root + current checkpoint. */
    param.i_frame_reference = reference_mode == 2 ? 16 : 2;
    /* P frames are invalidated after encoding so later targets fall back to
     * the GOP anchor. Keep that older anchor in the decoder DPB across the
     * resulting frame_num gaps. x264 explicitly recommends a large DPB when
     * invalidate_reference is used for this purpose. */
    param.i_dpb_size = gop_size < 16 ? gop_size : 16;
    param.analyse.b_mixed_references = 0;

    param.rc.i_rc_method = X264_RC_CRF;
    param.rc.f_rf_constant = crf;
    param.rc.f_rf_constant_max = crf;
    param.rc.i_aq_mode    = X264_AQ_VARIANCE;
    param.rc.i_lookahead  = 0;

    param.b_annexb        = 1;
    param.b_repeat_headers = 1;

    param.i_threads       = 1;
    param.b_sliced_threads = 0;
    param.i_sync_lookahead = 0;

    param.i_log_level = X264_LOG_ERROR;

    param.b_intra_refresh = 0;
    param.b_interlaced = 0;

    param.b_vfr_input = 0;
    param.i_fps_num  = 30;
    param.i_fps_den  = 1;
    param.i_timebase_num = 1;
    param.i_timebase_den = 30;

    x264_param_apply_profile(&param, "baseline");

    enc->h = x264_encoder_open(&param);
    if (!enc->h) {
        free(enc);
        return NULL;
    }

    return enc;
}

VClaspX264Encoder* vclasp_encoder_open(int width, int height, int crf,
                                  int gop_size, int anchor_p)
{
    return vclasp_encoder_open_common(width, height, crf, gop_size,
                                   anchor_p ? 1 : 0, 0);
}

VClaspX264Encoder* vclasp_encoder_open_two_level(int width, int height, int crf,
                                            int gop_size, int page_size)
{
    if (page_size < 2 || gop_size < page_size || gop_size % page_size != 0)
        return NULL;
    return vclasp_encoder_open_common(width, height, crf, gop_size, 2, page_size);
}

VClaspX264Encoder* vclasp_encoder_open_page_session(int width, int height, int crf,
                                               int keyint)
{
    if (keyint < 2)
        return NULL;
    return vclasp_encoder_open_common(width, height, crf, keyint, 3, 0);
}

int vclasp_encoder_encode_frame(VClaspX264Encoder* enc, const uint8_t* yuv,
                              int is_idr,
                              uint8_t* out_buf, int out_buf_size)
{
    if (!enc || !enc->h || !yuv || !out_buf || out_buf_size <= 0)
        return -1;

    x264_picture_t pic_in, pic_out;
    /* x264 invalidates the frame at `pts` and all later frames that depend on
     * it, while retaining older references. Monotonic PTS is therefore
     * required: invalidating P_k leaves the preceding IDR available. */
    int64_t pts = (int64_t)enc->frame_count;
    int frame_in_gop = enc->frame_count % enc->gop_size;
    int is_page_checkpoint = enc->anchor_p == 2 && !is_idr &&
                             frame_in_gop % enc->page_size == 0;
    int is_two_level_target = enc->anchor_p == 2 && !is_idr &&
                              !is_page_checkpoint;

    if (is_idr) {
        enc->checkpoint_pts = -1;
    } else if (is_page_checkpoint && enc->checkpoint_pts >= 0) {
        /* Remove the previous page checkpoint and its already-invalidated
         * children while retaining the older GOP root. The new checkpoint is
         * then forced to reference the root that remains in the DPB. */
        if (x264_encoder_invalidate_reference(enc->h, enc->checkpoint_pts) < 0)
            return -1;
        enc->checkpoint_pts = -1;
    }
    setup_picture(&pic_in, yuv, enc->width, enc->height, is_idr,
                  is_two_level_target, pts);

    x264_nal_t *nals = NULL;
    int i_nals = 0;
    int size = x264_encoder_encode(enc->h, &nals, &i_nals, &pic_in, &pic_out);

    if (size < 0)
        return -1;

    if (enc->anchor_p == 1) {
        if (!is_idr) {
            /* Drop just-encoded P-frame from DPB so the next frame cannot
             * use it as a reference. The base IDR stays because its PTS is
             * lower than the argument passed here. A new IDR resets x264's
             * reference state without a separate invalidation call. */
            if (x264_encoder_invalidate_reference(enc->h, pts) < 0)
                return -1;
        }
    } else if (enc->anchor_p == 2 && !is_idr) {
        if (is_page_checkpoint) {
            enc->checkpoint_pts = pts;
        } else {
            /* A target may use the root and current page checkpoint, but no
             * previous target. This bounds every target closure to at most
             * root + checkpoint + target. */
#ifndef VCLASP_PATCHED_X264
            if (x264_encoder_invalidate_reference(enc->h, pts) < 0)
                return -1;
#endif
        }
    } else if (enc->anchor_p == 3 && !is_idr) {
        if (enc->frame_count == 1) {
            /* A page session is root, checkpoint, then independently
             * selectable targets. Retain the first P as the checkpoint. */
            enc->checkpoint_pts = pts;
        } else {
            if (x264_encoder_invalidate_reference(enc->h, pts) < 0)
                return -1;
        }
    }

    enc->frame_count++;

    if (size == 0)
        return 0;

    if (size > out_buf_size)
        return -1;

    memcpy(out_buf, nals[0].p_payload, (size_t)size);
    return size;
}

int vclasp_encoder_flush(VClaspX264Encoder* enc,
                       uint8_t* out_buf, int out_buf_size)
{
    if (!enc || !enc->h || !out_buf || out_buf_size <= 0)
        return -1;

    x264_picture_t pic_out;
    x264_nal_t *nals = NULL;
    int i_nals = 0;
    int size = x264_encoder_encode(enc->h, &nals, &i_nals, NULL, &pic_out);

    if (size < 0)
        return -1;

    if (size == 0)
        return 0;

    if (size > out_buf_size)
        return -1;

    memcpy(out_buf, nals[0].p_payload, (size_t)size);
    return size;
}

void vclasp_encoder_close(VClaspX264Encoder* enc)
{
    if (!enc)
        return;
    if (enc->h)
        x264_encoder_close(enc->h);
    free(enc);
}
