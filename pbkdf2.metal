#include <metal_stdlib>
using namespace metal;

// SHA256 constants
constant uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA256 helper functions
inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

inline uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

inline uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Core SHA256 compression: takes 16 uint32_t words directly (no byte conversion).
// Uses 16-word circular buffer instead of w[64] to reduce per-thread register pressure
// and improve GPU occupancy (64 bytes vs 256 bytes of thread-local storage).
void sha256_compress_words(thread uint32_t* state, thread const uint32_t* data) {
    uint32_t w[16];
    for (int i = 0; i < 16; i++) w[i] = data[i];

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    // Rounds 0-15: use initial message words directly
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    // Rounds 16-63: expand message schedule on-the-fly with circular buffer
    for (int i = 16; i < 64; i++) {
        w[i & 15] = gamma1(w[(i - 2) & 15]) + w[(i - 7) & 15] +
                     gamma0(w[(i - 15) & 15]) + w[i & 15];
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + k[i] + w[i & 15];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// PBKDF2-SHA256 kernel — minimal per-thread storage for maximum GPU occupancy.
// Previous version used ~600 bytes/thread (pwd[128], salt_local[64], key_block[64],
// pad_block[64], etc.). This version uses ~224 bytes (ipad[8], opad[8], u[8], t[8],
// block[16], st[8]) by: reading password/salt directly from device memory,
// building ipad/opad in word form (no byte intermediaries), and computing U1 in
// word form (no u_bytes conversion).
kernel void pbkdf2_sha256_kernel(
    device const uint8_t* passwords [[buffer(0)]],
    device const uint8_t* salt_block [[buffer(1)]],
    device uint8_t* output [[buffer(2)]],
    constant uint& password_len [[buffer(3)]],
    constant uint& salt_block_len [[buffer(4)]],
    constant uint& iterations [[buffer(5)]],
    constant uint& num_passwords [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_passwords) return;

    uint pwd_len = min(password_len, 64u);  // passwords > 64 bytes not supported (would need hash)
    device const uint8_t* pwd = passwords + gid * password_len;

    // === Phase 1: HMAC midstates (ipad/opad) directly in word form ===
    // Build key as 16 big-endian words from password bytes, zero-padded
    uint32_t ipad_state[8], opad_state[8];
    {
        uint32_t kw[16];
        for (int i = 0; i < 16; i++) kw[i] = 0;
        for (uint i = 0; i < pwd_len; i++)
            kw[i >> 2] |= uint32_t(pwd[i]) << (24 - (i & 3) * 8);

        // ipad = key XOR 0x36363636 (per word)
        uint32_t pad[16];
        for (int i = 0; i < 16; i++) pad[i] = kw[i] ^ 0x36363636;
        ipad_state[0] = 0x6a09e667; ipad_state[1] = 0xbb67ae85;
        ipad_state[2] = 0x3c6ef372; ipad_state[3] = 0xa54ff53a;
        ipad_state[4] = 0x510e527f; ipad_state[5] = 0x9b05688c;
        ipad_state[6] = 0x1f83d9ab; ipad_state[7] = 0x5be0cd19;
        sha256_compress_words(ipad_state, pad);

        // opad = key XOR 0x5c5c5c5c (per word)
        for (int i = 0; i < 16; i++) pad[i] = kw[i] ^ 0x5c5c5c5c;
        opad_state[0] = 0x6a09e667; opad_state[1] = 0xbb67ae85;
        opad_state[2] = 0x3c6ef372; opad_state[3] = 0xa54ff53a;
        opad_state[4] = 0x510e527f; opad_state[5] = 0x9b05688c;
        opad_state[6] = 0x1f83d9ab; opad_state[7] = 0x5be0cd19;
        sha256_compress_words(opad_state, pad);
    }
    // kw[16] and pad[16] are dead — compiler can reuse stack space

    // === Phase 2: U1 = HMAC(password, salt_block) in word form ===
    uint32_t u_words[8], t_words[8];
    {
        // Build inner hash block: salt_block bytes as big-endian words + SHA256 padding
        uint32_t blk[16];
        for (int i = 0; i < 16; i++) blk[i] = 0;
        uint slen = min(salt_block_len, 60u);  // salt_block = salt + INT(1), fits one block
        for (uint i = 0; i < slen; i++)
            blk[i >> 2] |= uint32_t(salt_block[i]) << (24 - (i & 3) * 8);
        // SHA256 padding: 0x80 bit after data
        blk[slen >> 2] |= 0x80u << (24 - (slen & 3) * 8);
        // Bit length of (64 + slen) in big-endian at block[14..15]
        uint64_t bit_len = uint64_t(64 + slen) * 8;
        blk[14] = uint32_t(bit_len >> 32);
        blk[15] = uint32_t(bit_len);

        // Inner hash of U1
        uint32_t st[8];
        for (int i = 0; i < 8; i++) st[i] = ipad_state[i];
        sha256_compress_words(st, blk);

        // Outer hash of U1
        for (int i = 0; i < 8; i++) blk[i] = st[i];
        blk[8] = 0x80000000; for (int i = 9; i < 15; i++) blk[i] = 0; blk[15] = 0x300;
        for (int i = 0; i < 8; i++) st[i] = opad_state[i];
        sha256_compress_words(st, blk);

        for (int i = 0; i < 8; i++) { u_words[i] = st[i]; t_words[i] = st[i]; }
    }
    // blk[16] dead — stack reusable

    // === Phase 3: U2..Uc — the 100k-iteration hot loop ===
    // Only essential state: ipad_state[8], opad_state[8], u_words[8], t_words[8],
    // block[16], st[8] = 56 words = 224 bytes of thread-local storage.
    uint32_t block[16];
    block[8] = 0x80000000; block[9] = 0; block[10] = 0; block[11] = 0;
    block[12] = 0; block[13] = 0; block[14] = 0; block[15] = 0x300;

    for (uint j = 2; j <= iterations; j++) {
        for (int i = 0; i < 8; i++) block[i] = u_words[i];
        uint32_t st[8];
        for (int i = 0; i < 8; i++) st[i] = ipad_state[i];
        sha256_compress_words(st, block);

        for (int i = 0; i < 8; i++) block[i] = st[i];
        for (int i = 0; i < 8; i++) st[i] = opad_state[i];
        sha256_compress_words(st, block);

        for (int i = 0; i < 8; i++) {
            u_words[i] = st[i];
            t_words[i] ^= st[i];
        }
    }

    // Write output (big-endian)
    device uint8_t* out = output + (gid * 32);
    for (int i = 0; i < 8; i++) {
        out[i*4]     = (t_words[i] >> 24) & 0xff;
        out[i*4+1] = (t_words[i] >> 16) & 0xff;
        out[i*4+2] = (t_words[i] >> 8) & 0xff;
        out[i*4+3] = t_words[i] & 0xff;
    }
}
