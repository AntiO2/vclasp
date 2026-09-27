#ifndef X264_ENCODER_H
#define X264_ENCODER_H

#include <stdint.h>

typedef struct x264_encoder_s VClaspX264Encoder;

VClaspX264Encoder* vclasp_encoder_open(int width, int height, int crf,
                                  int gop_size, int anchor_p);
VClaspX264Encoder* vclasp_encoder_open_configured(
    int width, int height, int crf, int gop_size, int anchor_p,
    int fps, const char *preset);

VClaspX264Encoder* vclasp_encoder_open_two_level(int width, int height, int crf,
                                            int gop_size, int page_size);

VClaspX264Encoder* vclasp_encoder_open_page_session(int width, int height, int crf,
                                               int keyint);

int vclasp_encoder_encode_frame(VClaspX264Encoder* enc, const uint8_t* yuv,
                              int is_idr,
                              uint8_t* out_buf, int out_buf_size);

int vclasp_encoder_flush(VClaspX264Encoder* enc,
                       uint8_t* out_buf, int out_buf_size);

void vclasp_encoder_close(VClaspX264Encoder* enc);

#endif /* X264_ENCODER_H */
