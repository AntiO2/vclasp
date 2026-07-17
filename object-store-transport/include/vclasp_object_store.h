#ifndef VCLASP_OBJECT_STORE_H
#define VCLASP_OBJECT_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VClaspS3Client VClaspS3Client;
typedef struct VClaspObjectRange {
  const char *object_key;
  uint64_t offset;
  uint64_t length;
} VClaspObjectRange;
typedef struct VClaspBuffer {
  uint8_t *data;
  size_t length;
} VClaspBuffer;

VClaspS3Client *vclasp_s3_client_new(const char *, const char *, const char *,
                                const char *, const char *, size_t, char **);
void vclasp_s3_client_free(VClaspS3Client *);
int32_t vclasp_s3_fetch_ranges(VClaspS3Client *, const VClaspObjectRange *, size_t,
                            VClaspBuffer **, uint64_t *, char **);
void vclasp_buffers_free(VClaspBuffer *, size_t);
void vclasp_error_free(char *);

#ifdef __cplusplus
}
#endif
#endif
