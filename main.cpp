#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

// This was written on PC using VC++, so I'm using SSE2 for the 16-byte copies
// here; on ARM targets you would use NEON instead.
#include <emmintrin.h>

typedef __m128i pixel4_t;

static inline pixel4_t loadpixel4u(const uint32_t *src)
{
    // unaligned load!
    return _mm_loadu_si128((const __m128i *) src);
}

static void storepixel4(uint32_t *dest, pixel4_t vals)
{
    // aligned store
    _mm_store_si128((__m128i *) dest, vals);
}

// Pixel order with supertiling:
//   https://github.com/laanwj/etna_viv/blob/master/doc/hardware.md#texture-tiling
//
//        (bit index)    c  b a 9 8 7 6 5 4 3 2 1 0
//   index      = tile_idx:y5y4x5x4x3y3y2x2y1y0x1x0 
//              = index_x_tile | index_y
//
// with
//
//        (bit index)    c  b a 9 8 7 6 5 4 3 2 1 0
// index_x_tile = tile_idx:____x5x4x3____x2____x1x0
// index_y_tile = _________y5y4______y3y2__y1y0____

// width of the "outermost" tile (i.e. supertile in this case)
static uintptr_t swizzle_outermost_tile_w()
{
    return 64;
}

// height of the "outermost" tile
static uintptr_t swizzle_outermost_tile_h()
{
    return 64;
}

static uintptr_t swizzle_x_tile(uintptr_t x)
{
    return ((x & 0x03) << 0) | ((x & 0x04) << 2) | ((x & 0x38) << 4) | ((x & ~0x3f) << 6);
}

static uintptr_t swizzle_y(uintptr_t y)
{
    return ((y & 0x03) << 2) | ((y & 0x0c) << 3) | ((y & 0x30) << 6);
}

// "align" must be a power of 2.
static uintptr_t align_down(uintptr_t x, uintptr_t align)
{
    return x & ~(align - 1);
}

static uintptr_t align_up(uintptr_t x, uintptr_t align)
{
    return (x + align-1) & ~(align - 1);
}

// Swizzle sw*sh pixels worth of 32bpp texel data from "src" to "dest" at position (dx,dy).
// "dest" is a 2D texture of dw*dh pixels. Finally, "spitch" is the distance between rows
// in the source image (in units of 32bpp texels).
static void swizzle_32bpp_small(uint32_t *dest, uint32_t dx, uint32_t dy, uint32_t dw, uint32_t dh,
                                const uint32_t *src, uint32_t sw, uint32_t sh, uint32_t spitch)
{
    // Note that this function doesn't presume anything about the details of the tiling scheme,
    // other than assuming that the outermost tiles are stored in row-major order.
    //
    // In other words, you can make this produce a different tiling scheme just by changing
    // the swizzle_* functions above.

    uintptr_t x_mask = swizzle_x_tile(~0u);
    uintptr_t y_mask = swizzle_y(~0u);
    uintptr_t incr_y = swizzle_x_tile(align_up(dw, swizzle_outermost_tile_w()));
    uintptr_t offs_x0_tile = swizzle_x_tile(dx) + incr_y * (dy / swizzle_outermost_tile_h());
    uintptr_t offs_y = swizzle_y(dy);
    ptrdiff_t x_mask_bytes = x_mask * sizeof(uint32_t);

    for (uint32_t y = 0; y < sh; y++) {
        char *dest_line = (char *) (dest + offs_y);
        uintptr_t offs_x = offs_x0_tile * sizeof(uint32_t); // convert to bytes

        for (uint32_t x = 0; x < sw; x++) {
            *((uint32_t *) (dest_line + offs_x)) = src[x];
            offs_x = (offs_x - x_mask_bytes) & x_mask_bytes;
        }

        // advance pointers
        src += spitch;
        offs_y = (offs_y - y_mask) & y_mask;
        if (!offs_y) // wrapped into next tile row
            offs_x0_tile += incr_y;
    }
}

void swizzle_32bpp(uint32_t *dest, uint32_t dx, uint32_t dy, uint32_t dw, uint32_t dh,
                   const uint32_t *src, uint32_t sw, uint32_t sh, uint32_t spitch)
{
    // Innermost loop processes 4x4 pixel tiles at a time.
    static const uint32_t inner_tile_w = 4;
    static const uint32_t inner_tile_h = 4;

    // At 32 bits/pixel, this works out to 64 bytes - a full cache line.
    // Depending on the final implementation, you might want to process slightly larger
    // blocks (e.g. 8x4 or 8x8) at a time to amortize overhead better.
    //
    // The innermost loop here knows about the tile layout; the rest of the code is
    // fully generic and only uses the swizzle_* functions to determine the layout!

    uint32_t dx0 = dx;
    uint32_t dx1 = align_up(dx, inner_tile_w);
    uint32_t dx2 = align_down(dx + sw, inner_tile_w);
    uint32_t dx3 = dx + sw;

    uint32_t dy0 = dy;
    uint32_t dy1 = align_up(dy, inner_tile_h);
    uint32_t dy2 = align_down(dy + sh, inner_tile_h);
    uint32_t dy3 = dy + sh;

    // if we don't cover any full tiles, use the small loop
    if (dx2 <= dx1 || dy2 <= dy1) {
        swizzle_32bpp_small(dest, dx, dy, dw, dh, src, sw, sh, spitch);
        return;
    }

    // vertical slice [dy0,dy1)
    if (dy0 < dy1) {
        swizzle_32bpp_small(dest, dx0, dy0, dw, dh, src, sw, dy1 - dy0, spitch);
        src += (dy1 - dy0) * spitch;
    }

    // vertical slice [dy1,dy2) - tile aligned in y
    if (dx0 < dx1) // leftovers at left side?
        swizzle_32bpp_small(dest, dx0, dy1, dw, dh, src, dx1 - dx0, dy2 - dy1, spitch);

    // center part (fully tile aligned in dest)
    {
        ptrdiff_t x_step4_bytes = swizzle_x_tile(~0u * inner_tile_w) * sizeof(uint32_t);
        uintptr_t y_step4 = swizzle_y(~0u * inner_tile_h);
        uintptr_t incr_y = swizzle_x_tile(align_up(dw, swizzle_outermost_tile_w()));
        uintptr_t offs_x0_tile = swizzle_x_tile(dx1) + incr_y * (dy1 / swizzle_outermost_tile_h());
        uintptr_t offs_y = swizzle_y(dy1);
        const uint32_t *src_line = src + (dx1 - dx0);

        for (uint32_t y = dy1; y < dy2; y += inner_tile_h) {
            char *dest_line = (char *) (dest + offs_y);
            uintptr_t offs_x = offs_x0_tile * sizeof(uint32_t); // convert to bytes

            for (uint32_t x = dx1; x < dx2; x += inner_tile_w) {
                // NB for this loop you really want to make sure the generated code is
                // good (or possibly hand-write assembly code); this is just for illustration.
                pixel4_t row0 = loadpixel4u(src_line);
                pixel4_t row1 = loadpixel4u(src_line + spitch);
                pixel4_t row2 = loadpixel4u(src_line + 2*spitch);
                pixel4_t row3 = loadpixel4u(src_line + 3*spitch);
                uint32_t *dest_tile = (uint32_t *) (dest_line + offs_x);
                storepixel4(dest_tile +  0, row0);
                storepixel4(dest_tile +  4, row1);
                storepixel4(dest_tile +  8, row2);
                storepixel4(dest_tile + 12, row3);

                offs_x = (offs_x - x_step4_bytes) & x_step4_bytes;
                src_line += inner_tile_w;
            }

            // advance pointers
            src_line += inner_tile_h * spitch - (dx2 - dx1);
            offs_y = (offs_y - y_step4) & y_step4;
            if (!offs_y) // wrapped into next tile row
                offs_x0_tile += incr_y;
        }
    }

    if (dx2 < dx3) // leftovers at right side?
        swizzle_32bpp_small(dest, dx2, dy1, dw, dh, src + (dx2 - dx0), dx3 - dx2, dy2 - dy1, spitch);
    src += (dy2 - dy1) * spitch;

    // vertical slice [dy2,dy3)
    if (dy2 < dy3)
        swizzle_32bpp_small(dest, dx0, dy2, dw, dh, src, sw, dy3 - dy2, spitch);
}

// ---- Reference impl and test driver

static void swizzle_32bpp_ref(uint32_t *dest, uint32_t dx, uint32_t dy, uint32_t dw, uint32_t dh,
                              const uint32_t *src, uint32_t sw, uint32_t sh, uint32_t spitch)
{
    uintptr_t tile_y_stride = swizzle_x_tile(align_up(dw, swizzle_outermost_tile_w()));

    for (uint32_t y = 0; y < sh; y++) {
        uintptr_t dest_y_offs = ((dy + y) / swizzle_outermost_tile_h()) * tile_y_stride + swizzle_y(dy + y);
        uint32_t *dest_y = dest + dest_y_offs;

        for (uint32_t x = 0; x < sw; x++)
            dest_y[swizzle_x_tile(dx + x)] = src[x];

        src += spitch;
    }
}

int main(int argc, char **argv)
{
    static const uint32_t max_w = 128;
    static const uint32_t max_h = 128;
    static const uint32_t n_runs = 100000;

    uint32_t *linear = new uint32_t[max_w * max_h];
    uint32_t *tiled_ref = (uint32_t *) _aligned_malloc(max_w * max_h * sizeof(uint32_t), 64);
    uint32_t *tiled_fast = (uint32_t *) _aligned_malloc(max_w * max_h * sizeof(uint32_t), 64);

    // random test texture
    for (uint32_t i = 0; i < max_w * max_h; i++)
        linear[i] = rand();

    // do some test runs
    for (uint32_t run = 0; run < n_runs; run++) {
        static const size_t tiled_size = max_w * max_h * sizeof(uint32_t);
        memset(tiled_ref, 0xcc, tiled_size);
        memset(tiled_fast, 0xcc, tiled_size);

        // determine random target rectangle
        uint32_t x0 = rand() % max_w;
        uint32_t y0 = rand() % max_h;
        uint32_t x1 = rand() % max_w;
        uint32_t y1 = rand() % max_h;

        if (x0 > x1) std::swap(x0, x1);
        if (y0 > y1) std::swap(y0, y1);
        
        // determine random destination texture size that fits target rect
        uint32_t dw = x1;
        uint32_t dh = y1;
        if (dw < max_w) dw += rand() % (max_w - dw);
        if (dh < max_h) dh += rand() % (max_h - dh);

        // swizzle two ways
        swizzle_32bpp_ref(tiled_ref, x0, y0, dw, dh, linear, x1 - x0, y1 - y0, x1 - x0);
        swizzle_32bpp(tiled_fast, x0, y0, dw, dh, linear, x1 - x0, y1 - y0, x1 - x0);

        if (memcmp(tiled_ref, tiled_fast, tiled_size) != 0) {
            printf("mismatch!\n");
            printf("rect=(%d,%d)-(%d,%d), dw=%d dh=%d\n", x0, y0, x1, y1, dw, dh);
            return 1;
        }
    }

    delete[] linear;
    _aligned_free(tiled_ref);
    _aligned_free(tiled_fast);

    printf("all ok!\n");
    return 0;
}
