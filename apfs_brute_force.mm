/*
 * APFS Password Brute Forcer
 * ==============================
 *
 * Full-featured brute force tool for encrypted APFS volumes.
 *
 * Features:
 *   - Native keybag extraction (no Python dependency)
 *   - GPU acceleration via Metal + CPU workers (shared work queue)
 *   - Brute force, dictionary, and rule-based mutation attacks
 *   - Resume/checkpoint support
 *   - Multi-process mode
 *   - Real-time progress reporting
 *
 * Build: make brute-force
 * Usage: ./apfs_brute_force <dmg_file> [options]
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CommonCrypto/CommonDigest.h>
#import <CommonCrypto/CommonHMAC.h>
#import <CommonCrypto/CommonCryptor.h>
#import <pthread.h>
#import <atomic>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <csignal>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

// =============================================================================
// Constants
// =============================================================================

#define SHA256_DIGEST_SIZE 32
#define MAX_PASSWORD_LEN 256
#define DEFAULT_GPU_BATCH_SIZE 32768
#define CPU_BATCH_SIZE 64
#define NXSB_MAGIC 0x4253584E   // "NXSB" little-endian
#define KEYS_MAGIC 0x6B657973   // "keys" little-endian
#define RECS_MAGIC 0x72656373   // "recs" little-endian
#define CHECKPOINT_INTERVAL 60  // seconds
#define PROGRESS_INTERVAL 2     // seconds

static const char *CHARSET_LOWER = "abcdefghijklmnopqrstuvwxyz";
static const char *CHARSET_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const char *CHARSET_DIGITS = "0123456789";
static const char *CHARSET_ALPHANUMERIC = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static const char *CHARSET_LOWER_DIGITS = "abcdefghijklmnopqrstuvwxyz0123456789";
static const char *CHARSET_UPPER_DIGITS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
static const char *CHARSET_ALL = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?";

// Global signal flag
static volatile sig_atomic_t g_interrupted = 0;
static void sigint_handler(int) { g_interrupted = 1; }

// =============================================================================
// Crypto primitives — ARM SHA2 hardware-accelerated SHA256
// =============================================================================

static const uint32_t SHA256_K[64] = {
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

static const uint32_t SHA256_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

#if USE_NEON && defined(__ARM_FEATURE_SHA2)

// Hardware SHA256 compress using ARM Crypto Extensions.
// Processes one 64-byte block (as 16 uint32_t words) against state[8].
// Uses vsha256h/vsha256h2/vsha256su0/vsha256su1 intrinsics — each processes
// 4 rounds, so 16 groups cover all 64 rounds.
static inline void sha256_compress_block(uint32_t state[8], const uint32_t data[16]) {
    uint32x4_t abcd = vld1q_u32(state);
    uint32x4_t efgh = vld1q_u32(state + 4);
    uint32x4_t abcd_save = abcd;
    uint32x4_t efgh_save = efgh;

    uint32x4_t tmp, save;
    uint32x4_t msg0 = vld1q_u32(data);
    uint32x4_t msg1 = vld1q_u32(data + 4);
    uint32x4_t msg2 = vld1q_u32(data + 8);
    uint32x4_t msg3 = vld1q_u32(data + 12);

    // Rounds 0-3
    tmp = vaddq_u32(msg0, vld1q_u32(&SHA256_K[0]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg0 = vsha256su1q_u32(vsha256su0q_u32(msg0, msg1), msg2, msg3);

    // Rounds 4-7
    tmp = vaddq_u32(msg1, vld1q_u32(&SHA256_K[4]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg1 = vsha256su1q_u32(vsha256su0q_u32(msg1, msg2), msg3, msg0);

    // Rounds 8-11
    tmp = vaddq_u32(msg2, vld1q_u32(&SHA256_K[8]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg2 = vsha256su1q_u32(vsha256su0q_u32(msg2, msg3), msg0, msg1);

    // Rounds 12-15
    tmp = vaddq_u32(msg3, vld1q_u32(&SHA256_K[12]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg3 = vsha256su1q_u32(vsha256su0q_u32(msg3, msg0), msg1, msg2);

    // Rounds 16-19
    tmp = vaddq_u32(msg0, vld1q_u32(&SHA256_K[16]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg0 = vsha256su1q_u32(vsha256su0q_u32(msg0, msg1), msg2, msg3);

    // Rounds 20-23
    tmp = vaddq_u32(msg1, vld1q_u32(&SHA256_K[20]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg1 = vsha256su1q_u32(vsha256su0q_u32(msg1, msg2), msg3, msg0);

    // Rounds 24-27
    tmp = vaddq_u32(msg2, vld1q_u32(&SHA256_K[24]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg2 = vsha256su1q_u32(vsha256su0q_u32(msg2, msg3), msg0, msg1);

    // Rounds 28-31
    tmp = vaddq_u32(msg3, vld1q_u32(&SHA256_K[28]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg3 = vsha256su1q_u32(vsha256su0q_u32(msg3, msg0), msg1, msg2);

    // Rounds 32-35
    tmp = vaddq_u32(msg0, vld1q_u32(&SHA256_K[32]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg0 = vsha256su1q_u32(vsha256su0q_u32(msg0, msg1), msg2, msg3);

    // Rounds 36-39
    tmp = vaddq_u32(msg1, vld1q_u32(&SHA256_K[36]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg1 = vsha256su1q_u32(vsha256su0q_u32(msg1, msg2), msg3, msg0);

    // Rounds 40-43
    tmp = vaddq_u32(msg2, vld1q_u32(&SHA256_K[40]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg2 = vsha256su1q_u32(vsha256su0q_u32(msg2, msg3), msg0, msg1);

    // Rounds 44-47
    tmp = vaddq_u32(msg3, vld1q_u32(&SHA256_K[44]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);
    msg3 = vsha256su1q_u32(vsha256su0q_u32(msg3, msg0), msg1, msg2);

    // Rounds 48-51 (no more schedule updates needed)
    tmp = vaddq_u32(msg0, vld1q_u32(&SHA256_K[48]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);

    // Rounds 52-55
    tmp = vaddq_u32(msg1, vld1q_u32(&SHA256_K[52]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);

    // Rounds 56-59
    tmp = vaddq_u32(msg2, vld1q_u32(&SHA256_K[56]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);

    // Rounds 60-63
    tmp = vaddq_u32(msg3, vld1q_u32(&SHA256_K[60]));
    save = abcd; abcd = vsha256hq_u32(abcd, efgh, tmp); efgh = vsha256h2q_u32(efgh, save, tmp);

    // Final addition
    vst1q_u32(state, vaddq_u32(abcd, abcd_save));
    vst1q_u32(state + 4, vaddq_u32(efgh, efgh_save));
}

// 2-way interleaved SHA256 compression: processes two independent hashes simultaneously.
// Hides the 3-cycle latency of vsha256hq/vsha256h2q by interleaving instructions from
// two independent SHA256 computations. This achieves ~2x throughput vs sequential.
// Uses ~20 NEON registers (ARM64 has 32).
static inline void sha256_compress_block_x2(
    uint32_t stA[8], const uint32_t datA[16],
    uint32_t stB[8], const uint32_t datB[16])
{
    uint32x4_t aA = vld1q_u32(stA), eA = vld1q_u32(stA + 4);
    uint32x4_t aB = vld1q_u32(stB), eB = vld1q_u32(stB + 4);
    uint32x4_t aAs = aA, eAs = eA, aBs = aB, eBs = eB;

    uint32x4_t m0A = vld1q_u32(datA), m1A = vld1q_u32(datA+4), m2A = vld1q_u32(datA+8), m3A = vld1q_u32(datA+12);
    uint32x4_t m0B = vld1q_u32(datB), m1B = vld1q_u32(datB+4), m2B = vld1q_u32(datB+8), m3B = vld1q_u32(datB+12);
    uint32x4_t t, s;

    // Macro: 4 rounds for both A and B, with message schedule update
#define DX2(msg, nxt1, nxt2, nxt3, ki) \
    t = vaddq_u32(msg##A, vld1q_u32(&SHA256_K[ki])); \
    s = aA; aA = vsha256hq_u32(aA, eA, t); eA = vsha256h2q_u32(eA, s, t); \
    t = vaddq_u32(msg##B, vld1q_u32(&SHA256_K[ki])); \
    s = aB; aB = vsha256hq_u32(aB, eB, t); eB = vsha256h2q_u32(eB, s, t); \
    msg##A = vsha256su1q_u32(vsha256su0q_u32(msg##A, nxt1##A), nxt2##A, nxt3##A); \
    msg##B = vsha256su1q_u32(vsha256su0q_u32(msg##B, nxt1##B), nxt2##B, nxt3##B);

    // Macro: 4 rounds for both A and B, NO schedule update (last 4 groups)
#define DX2_NOSCHED(msg, ki) \
    t = vaddq_u32(msg##A, vld1q_u32(&SHA256_K[ki])); \
    s = aA; aA = vsha256hq_u32(aA, eA, t); eA = vsha256h2q_u32(eA, s, t); \
    t = vaddq_u32(msg##B, vld1q_u32(&SHA256_K[ki])); \
    s = aB; aB = vsha256hq_u32(aB, eB, t); eB = vsha256h2q_u32(eB, s, t);

    DX2(m0, m1, m2, m3, 0)    // Rounds 0-3
    DX2(m1, m2, m3, m0, 4)    // Rounds 4-7
    DX2(m2, m3, m0, m1, 8)    // Rounds 8-11
    DX2(m3, m0, m1, m2, 12)   // Rounds 12-15
    DX2(m0, m1, m2, m3, 16)   // Rounds 16-19
    DX2(m1, m2, m3, m0, 20)   // Rounds 20-23
    DX2(m2, m3, m0, m1, 24)   // Rounds 24-27
    DX2(m3, m0, m1, m2, 28)   // Rounds 28-31
    DX2(m0, m1, m2, m3, 32)   // Rounds 32-35
    DX2(m1, m2, m3, m0, 36)   // Rounds 36-39
    DX2(m2, m3, m0, m1, 40)   // Rounds 40-43
    DX2(m3, m0, m1, m2, 44)   // Rounds 44-47
    DX2_NOSCHED(m0, 48)        // Rounds 48-51
    DX2_NOSCHED(m1, 52)        // Rounds 52-55
    DX2_NOSCHED(m2, 56)        // Rounds 56-59
    DX2_NOSCHED(m3, 60)        // Rounds 60-63

#undef DX2
#undef DX2_NOSCHED

    vst1q_u32(stA, vaddq_u32(aA, aAs));
    vst1q_u32(stA + 4, vaddq_u32(eA, eAs));
    vst1q_u32(stB, vaddq_u32(aB, aBs));
    vst1q_u32(stB + 4, vaddq_u32(eB, eBs));
}

#else

// Software SHA256 compress fallback (non-ARM or no SHA2 extensions)
static inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static inline void sha256_compress_block(uint32_t state[8], const uint32_t data[16]) {
    uint32_t w[64];
    memcpy(w, data, 64);
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
    uint32_t e=state[4], f=state[5], g=state[6], h=state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t cv = (e & f) ^ (~e & g);
        uint32_t T1 = h + S1 + cv + SHA256_K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t T2 = S0 + mj;
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }

    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

#endif

// Pre-compute HMAC ipad/opad midstates as uint32_t word arrays.
// Builds ipad/opad blocks directly in word form to avoid byte overhead.
static inline void hmac_precompute(const uint8_t *key, size_t key_len,
                                   uint32_t ipad_state[8], uint32_t opad_state[8]) {
    uint8_t key_block[64] = {0};
    if (key_len > 64) {
        CC_SHA256(key, (CC_LONG)key_len, key_block);
    } else {
        memcpy(key_block, key, key_len);
    }

    // Build ipad/opad as uint32_t words directly (key_byte ^ 0x36/0x5c per byte,
    // which equals key_word ^ 0x36363636/0x5c5c5c5c per word)
    uint32_t ipad_words[16], opad_words[16];
    for (int i = 0; i < 16; i++) {
        uint32_t kb = ((uint32_t)key_block[i*4] << 24) | ((uint32_t)key_block[i*4+1] << 16) |
                      ((uint32_t)key_block[i*4+2] << 8) | (uint32_t)key_block[i*4+3];
        ipad_words[i] = kb ^ 0x36363636;
        opad_words[i] = kb ^ 0x5c5c5c5c;
    }

    memcpy(ipad_state, SHA256_INIT, 32);
    sha256_compress_block(ipad_state, ipad_words);

    memcpy(opad_state, SHA256_INIT, 32);
    sha256_compress_block(opad_state, opad_words);
}

// Constant padding block tail for 32-byte data, total_len=96 (bit_len=0x300).
// Used by both inner and outer HMAC hashes in the U2..Uc loop.
static const uint32_t PADDING_96[8] = {0x80000000, 0, 0, 0, 0, 0, 0, 0x300};

static void pbkdf2_sha256_cpu(uint8_t *output, const uint8_t *password, size_t pwd_len,
                              const uint8_t *salt, size_t salt_len, uint32_t iterations) {
    // Pre-compute HMAC midstates (as uint32_t word arrays)
    uint32_t ipad_state[8], opad_state[8];
    hmac_precompute(password, pwd_len, ipad_state, opad_state);

    // U1 = HMAC(password, salt || INT(1))
    // Build the finalization block for the inner hash (salt is ~20 bytes, fits one block)
    uint8_t sb_bytes[64] = {0};
    memcpy(sb_bytes, salt, salt_len);
    sb_bytes[salt_len] = 0x00; sb_bytes[salt_len+1] = 0x00;
    sb_bytes[salt_len+2] = 0x00; sb_bytes[salt_len+3] = 0x01;
    size_t sb_len = salt_len + 4;

    // Inner hash block: salt_block + 0x80 + padding + bit_length
    uint8_t u1_block_bytes[64] = {0};
    memcpy(u1_block_bytes, sb_bytes, sb_len);
    u1_block_bytes[sb_len] = 0x80;
    uint64_t bit_len = (uint64_t)(64 + sb_len) * 8;
    u1_block_bytes[56] = (bit_len >> 56); u1_block_bytes[57] = (bit_len >> 48);
    u1_block_bytes[58] = (bit_len >> 40); u1_block_bytes[59] = (bit_len >> 32);
    u1_block_bytes[60] = (bit_len >> 24); u1_block_bytes[61] = (bit_len >> 16);
    u1_block_bytes[62] = (bit_len >> 8);  u1_block_bytes[63] = (uint8_t)(bit_len);

    uint32_t block[16];
    for (int i = 0; i < 16; i++)
        block[i] = ((uint32_t)u1_block_bytes[i*4] << 24) | ((uint32_t)u1_block_bytes[i*4+1] << 16) |
                   ((uint32_t)u1_block_bytes[i*4+2] << 8) | (uint32_t)u1_block_bytes[i*4+3];

    uint32_t state[8];
    memcpy(state, ipad_state, 32);
    sha256_compress_block(state, block);

    // Outer hash of U1
    memcpy(block, state, 32);
    memcpy(block + 8, PADDING_96, 32);
    memcpy(state, opad_state, 32);
    sha256_compress_block(state, block);

    // state = U1 in word form
    uint32_t u_words[8], t_words[8];
    memcpy(u_words, state, 32);
    memcpy(t_words, state, 32);

    // Set constant padding once (reused for all iterations)
    memcpy(block + 8, PADDING_96, 32);

    // U2..Uc: entirely in uint32_t word space
    for (uint32_t j = 2; j <= iterations; j++) {
        // Inner: SHA256(ipad_state, u_words || padding)
        memcpy(block, u_words, 32);
        memcpy(state, ipad_state, 32);
        sha256_compress_block(state, block);

        // Outer: SHA256(opad_state, inner_hash || padding)
        memcpy(block, state, 32);
        memcpy(state, opad_state, 32);
        sha256_compress_block(state, block);

        // Update u and accumulate t (NEON XOR for t)
        memcpy(u_words, state, 32);
#if USE_NEON
        uint32x4_t tv0 = vld1q_u32(t_words), sv0 = vld1q_u32(state);
        vst1q_u32(t_words, veorq_u32(tv0, sv0));
        uint32x4_t tv1 = vld1q_u32(t_words+4), sv1 = vld1q_u32(state+4);
        vst1q_u32(t_words+4, veorq_u32(tv1, sv1));
#else
        for (int i = 0; i < 8; i++) t_words[i] ^= state[i];
#endif
    }

    // Convert t_words to output bytes (big-endian)
    for (int i = 0; i < 8; i++) {
        output[i*4]   = (t_words[i] >> 24) & 0xff;
        output[i*4+1] = (t_words[i] >> 16) & 0xff;
        output[i*4+2] = (t_words[i] >> 8) & 0xff;
        output[i*4+3] = t_words[i] & 0xff;
    }
}

// Process two passwords simultaneously with interleaved SHA256 compressions.
// The ARM SHA2 instructions have ~3-cycle latency but 1-cycle throughput, so
// interleaving two independent computations hides the latency for ~2x throughput.
#if USE_NEON && defined(__ARM_FEATURE_SHA2)
static void pbkdf2_sha256_cpu_pair(uint8_t *outA, uint8_t *outB,
                                    const uint8_t *pwdA, size_t lenA,
                                    const uint8_t *pwdB, size_t lenB,
                                    const uint32_t u1_block_words[16],
                                    uint32_t iterations) {
    // Pre-compute HMAC midstates for both passwords
    uint32_t ipadA[8], opadA[8], ipadB[8], opadB[8];
    hmac_precompute(pwdA, lenA, ipadA, opadA);
    hmac_precompute(pwdB, lenB, ipadB, opadB);

    // U1 for both: inner hash (can't interleave — need results before outer hash)
    uint32_t blkA[16], blkB[16], stA[8], stB[8];
    memcpy(blkA, u1_block_words, 64);
    memcpy(blkB, u1_block_words, 64);
    memcpy(stA, ipadA, 32);
    memcpy(stB, ipadB, 32);
    sha256_compress_block_x2(stA, blkA, stB, blkB);

    // U1: outer hash
    memcpy(blkA, stA, 32); memcpy(blkA + 8, PADDING_96, 32);
    memcpy(blkB, stB, 32); memcpy(blkB + 8, PADDING_96, 32);
    memcpy(stA, opadA, 32);
    memcpy(stB, opadB, 32);
    sha256_compress_block_x2(stA, blkA, stB, blkB);

    uint32_t uA[8], tA[8], uB[8], tB[8];
    memcpy(uA, stA, 32); memcpy(tA, stA, 32);
    memcpy(uB, stB, 32); memcpy(tB, stB, 32);

    // Set constant padding once
    memcpy(blkA + 8, PADDING_96, 32);
    memcpy(blkB + 8, PADDING_96, 32);

    // U2..Uc: interleaved inner loop
    for (uint32_t j = 2; j <= iterations; j++) {
        // Inner hash: SHA256(ipad, u || padding)
        memcpy(blkA, uA, 32); memcpy(stA, ipadA, 32);
        memcpy(blkB, uB, 32); memcpy(stB, ipadB, 32);
        sha256_compress_block_x2(stA, blkA, stB, blkB);

        // Outer hash: SHA256(opad, inner || padding)
        memcpy(blkA, stA, 32); memcpy(stA, opadA, 32);
        memcpy(blkB, stB, 32); memcpy(stB, opadB, 32);
        sha256_compress_block_x2(stA, blkA, stB, blkB);

        // Update u and XOR into t
        memcpy(uA, stA, 32); memcpy(uB, stB, 32);
        uint32x4_t v0, v1;
        v0 = veorq_u32(vld1q_u32(tA), vld1q_u32(stA)); vst1q_u32(tA, v0);
        v1 = veorq_u32(vld1q_u32(tA+4), vld1q_u32(stA+4)); vst1q_u32(tA+4, v1);
        v0 = veorq_u32(vld1q_u32(tB), vld1q_u32(stB)); vst1q_u32(tB, v0);
        v1 = veorq_u32(vld1q_u32(tB+4), vld1q_u32(stB+4)); vst1q_u32(tB+4, v1);
    }

    for (int i = 0; i < 8; i++) {
        outA[i*4]=(tA[i]>>24)&0xff; outA[i*4+1]=(tA[i]>>16)&0xff;
        outA[i*4+2]=(tA[i]>>8)&0xff; outA[i*4+3]=tA[i]&0xff;
        outB[i*4]=(tB[i]>>24)&0xff; outB[i*4+1]=(tB[i]>>16)&0xff;
        outB[i*4+2]=(tB[i]>>8)&0xff; outB[i*4+3]=tB[i]&0xff;
    }
}
#endif

static void pbkdf2_sha256_cpu_batch(uint8_t *outputs, const char **passwords,
                                    const size_t *pwd_lens, size_t batch_size,
                                    const uint8_t *salt, size_t salt_len, uint32_t iterations) {
    // Pre-build U1 inner hash block template (same for all passwords)
    uint8_t sb_bytes[64] = {0};
    memcpy(sb_bytes, salt, salt_len);
    sb_bytes[salt_len] = 0x00; sb_bytes[salt_len+1] = 0x00;
    sb_bytes[salt_len+2] = 0x00; sb_bytes[salt_len+3] = 0x01;
    size_t sb_len = salt_len + 4;

    uint8_t u1_block_bytes[64] = {0};
    memcpy(u1_block_bytes, sb_bytes, sb_len);
    u1_block_bytes[sb_len] = 0x80;
    uint64_t bit_len = (uint64_t)(64 + sb_len) * 8;
    u1_block_bytes[56] = (bit_len >> 56); u1_block_bytes[57] = (bit_len >> 48);
    u1_block_bytes[58] = (bit_len >> 40); u1_block_bytes[59] = (bit_len >> 32);
    u1_block_bytes[60] = (bit_len >> 24); u1_block_bytes[61] = (bit_len >> 16);
    u1_block_bytes[62] = (bit_len >> 8);  u1_block_bytes[63] = (uint8_t)(bit_len);

    uint32_t u1_block_words[16];
    for (int i = 0; i < 16; i++)
        u1_block_words[i] = ((uint32_t)u1_block_bytes[i*4] << 24) | ((uint32_t)u1_block_bytes[i*4+1] << 16) |
                             ((uint32_t)u1_block_bytes[i*4+2] << 8) | (uint32_t)u1_block_bytes[i*4+3];

#if USE_NEON && defined(__ARM_FEATURE_SHA2)
    // Process in pairs using 2-way interleaved SHA256
    size_t p = 0;
    for (; p + 1 < batch_size; p += 2) {
        pbkdf2_sha256_cpu_pair(
            outputs + p * SHA256_DIGEST_SIZE,
            outputs + (p + 1) * SHA256_DIGEST_SIZE,
            (const uint8_t *)passwords[p], pwd_lens[p],
            (const uint8_t *)passwords[p + 1], pwd_lens[p + 1],
            u1_block_words, iterations);
    }
    // Handle odd remainder
    if (p < batch_size) {
        pbkdf2_sha256_cpu(outputs + p * SHA256_DIGEST_SIZE,
                          (const uint8_t *)passwords[p], pwd_lens[p],
                          salt, salt_len, iterations);
    }
#else
    for (size_t p = 0; p < batch_size; p++) {
        pbkdf2_sha256_cpu(outputs + p * SHA256_DIGEST_SIZE,
                          (const uint8_t *)passwords[p], pwd_lens[p],
                          salt, salt_len, iterations);
    }
#endif
}

// AES key unwrapping (RFC 3394)
static bool aes_unwrap_key(uint8_t *unwrapped, const uint8_t *wrapped, size_t wrapped_len,
                           const uint8_t *kek, size_t kek_len) {
    if (wrapped_len < 24 || wrapped_len % 8 != 0) return false;
    size_t n = (wrapped_len / 8) - 1;
    uint8_t a[8];
    uint8_t *r = (uint8_t *)malloc(n * 8);
    if (!r) return false;

    memcpy(a, wrapped, 8);
    memcpy(r, wrapped + 8, n * 8);

    for (int j = 5; j >= 0; j--) {
        for (int i = (int)n - 1; i >= 0; i--) {
            uint64_t t_val = (uint64_t)(n * j + i + 1);
            uint64_t a64 = ((uint64_t)a[0]<<56)|((uint64_t)a[1]<<48)|
                           ((uint64_t)a[2]<<40)|((uint64_t)a[3]<<32)|
                           ((uint64_t)a[4]<<24)|((uint64_t)a[5]<<16)|
                           ((uint64_t)a[6]<<8)|(uint64_t)a[7];
            a64 ^= t_val;
            for (int k = 0; k < 8; k++) a[k] = (uint8_t)(a64 >> (56 - k*8));

            uint8_t b[16] __attribute__((aligned(16)));
            memcpy(b, a, 8);
            memcpy(b + 8, r + i*8, 8);
            size_t out_len = 16;
            if (CCCrypt(kCCDecrypt, kCCAlgorithmAES, 0, kek, kek_len,
                        NULL, b, 16, b, 16, &out_len) != kCCSuccess) {
                free(r); return false;
            }
            memcpy(a, b, 8);
            memcpy(r + i*8, b + 8, 8);
        }
    }

    uint64_t a64 = ((uint64_t)a[0]<<56)|((uint64_t)a[1]<<48)|
                   ((uint64_t)a[2]<<40)|((uint64_t)a[3]<<32)|
                   ((uint64_t)a[4]<<24)|((uint64_t)a[5]<<16)|
                   ((uint64_t)a[6]<<8)|(uint64_t)a[7];
    bool valid = (a64 == 0xA6A6A6A6A6A6A6A6ULL);
    if (valid) memcpy(unwrapped, r, n * 8);
    free(r);
    return valid;
}

// =============================================================================
// AES-XTS for keybag decryption (ported from encrypted_recovery.py)
// =============================================================================

static void gf128_multiply(uint8_t tweak[16]) {
    int carry = 0;
    for (int i = 0; i < 16; i++) {
        int new_carry = (tweak[i] >> 7) & 1;
        tweak[i] = ((tweak[i] << 1) | carry) & 0xFF;
        carry = new_carry;
    }
    if (carry) tweak[0] ^= 0x87;
}

static bool aes_ecb_encrypt(uint8_t *output, const uint8_t *input, size_t len,
                            const uint8_t *key, size_t key_len) {
    size_t out_len = len;
    return CCCrypt(kCCEncrypt, kCCAlgorithmAES, kCCOptionECBMode,
                   key, key_len, NULL, input, len, output, len, &out_len) == kCCSuccess;
}

static bool aes_ecb_decrypt(uint8_t *output, const uint8_t *input, size_t len,
                            const uint8_t *key, size_t key_len) {
    size_t out_len = len;
    return CCCrypt(kCCDecrypt, kCCAlgorithmAES, kCCOptionECBMode,
                   key, key_len, NULL, input, len, output, len, &out_len) == kCCSuccess;
}

static bool aes_xts_decrypt(uint8_t *output, const uint8_t *ciphertext, size_t len,
                            const uint8_t *key, uint64_t block_no) {
    // APFS uses 512-byte sectors, block_size=4096, so cs_factor=8
    const int sector_size = 512;
    const int cs_factor = 4096 / sector_size;
    uint64_t sector_no = block_no * cs_factor;

    // key is 16 bytes, used as both key1 (data) and key2 (tweak)
    const uint8_t *key1 = key;
    const uint8_t *key2 = key;

    for (size_t sector_start = 0; sector_start < len; sector_start += sector_size) {
        size_t sector_len = (sector_start + sector_size <= len) ? sector_size : (len - sector_start);

        // Encrypt tweak: AES-ECB(key2, sector_no || 0)
        uint8_t tweak_input[16] = {0};
        memcpy(tweak_input, &sector_no, 8); // little-endian sector_no
        uint8_t tweak[16];
        if (!aes_ecb_encrypt(tweak, tweak_input, 16, key2, 16)) return false;

        for (size_t i = 0; i < sector_len; i += 16) {
            uint8_t block[16];
            // XOR with tweak
            for (int j = 0; j < 16; j++)
                block[j] = ciphertext[sector_start + i + j] ^ tweak[j];
            // AES-ECB decrypt
            uint8_t dec[16];
            if (!aes_ecb_decrypt(dec, block, 16, key1, 16)) return false;
            // XOR with tweak
            for (int j = 0; j < 16; j++)
                output[sector_start + i + j] = dec[j] ^ tweak[j];
            gf128_multiply(tweak);
        }
        sector_no++;
    }
    return true;
}

// =============================================================================
// Native keybag extraction (ported from apfs_brute_force_gpu.py)
// =============================================================================

struct KeybagData {
    uint8_t salt[16];
    uint32_t iterations;
    uint8_t *wrapped_kek;
    size_t wrapped_kek_len;
    uint8_t *wrapped_vek;
    size_t wrapped_vek_len;
};

static bool extract_keybag_native(const char *image_path, KeybagData *kb) {
    memset(kb, 0, sizeof(KeybagData));

    // mmap the image
    int fd = open(image_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "  ERROR: Cannot open %s\n", image_path); return false; }

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return false; }
    size_t file_size = st.st_size;

    uint8_t *data = (uint8_t *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { fprintf(stderr, "  ERROR: Cannot mmap file\n"); return false; }

    // Find NXSB superblock - check common offsets first
    size_t partition_offset = 0;
    bool found_nxsb = false;
    // Common offsets: 0, 20480 (0x5000 = 5*4096)
    size_t try_offsets[] = {0, 20480, 4096, 8192, 12288, 16384};
    for (size_t off : try_offsets) {
        if (off + 1312 < file_size) {
            uint32_t magic;
            memcpy(&magic, data + off + 32, 4);
            if (magic == NXSB_MAGIC) {
                partition_offset = off;
                found_nxsb = true;
                break;
            }
        }
    }
    if (!found_nxsb) {
        // Scan first 100 blocks
        for (size_t off = 0; off < file_size && off < 100*4096; off += 4096) {
            uint32_t magic;
            memcpy(&magic, data + off + 32, 4);
            if (magic == NXSB_MAGIC) {
                partition_offset = off;
                found_nxsb = true;
                break;
            }
        }
    }
    if (!found_nxsb) {
        fprintf(stderr, "  ERROR: NXSB superblock not found\n");
        munmap(data, file_size); return false;
    }

    uint8_t *sb = data + partition_offset;
    uint32_t block_size;
    memcpy(&block_size, sb + 36, 4);
    if (block_size == 0 || block_size > 65536) block_size = 4096;

    uint8_t container_uuid[16];
    memcpy(container_uuid, sb + 72, 16);
    printf("  Container UUID: ");
    for (int i = 0; i < 8; i++) printf("%02x", container_uuid[i]);
    printf("...\n");

    // Read keylocker location
    uint64_t keylocker_start, keylocker_count;
    memcpy(&keylocker_start, sb + 1296, 8);
    memcpy(&keylocker_count, sb + 1304, 8);

    if (keylocker_start == 0 || keylocker_count == 0) {
        printf("  WARNING: No keylocker in superblock, scanning...\n");
        size_t max_blocks = file_size / block_size;
        if (max_blocks > 1000) max_blocks = 1000;
        for (size_t blk = 0; blk < max_blocks; blk++) {
            size_t off = partition_offset + blk * block_size;
            if (off + 28 >= file_size) break;
            uint32_t obj_type;
            memcpy(&obj_type, data + off + 24, 4);
            if (obj_type == KEYS_MAGIC) {
                printf("  Found keybag at block %zu\n", blk);
                keylocker_start = blk;
                keylocker_count = 1;
                break;
            }
        }
        if (keylocker_start == 0) {
            fprintf(stderr, "  ERROR: No keybag found\n");
            munmap(data, file_size); return false;
        }
    }
    printf("  Container keybag: block %llu\n", keylocker_start);

    // Read and decrypt container keybag
    size_t kb_off = partition_offset + keylocker_start * block_size;
    if (kb_off + block_size > file_size) {
        fprintf(stderr, "  ERROR: Keybag block out of range\n");
        munmap(data, file_size); return false;
    }

    uint8_t *keybag_dec = (uint8_t *)malloc(block_size);
    if (!aes_xts_decrypt(keybag_dec, data + kb_off, block_size, container_uuid, keylocker_start)) {
        fprintf(stderr, "  ERROR: Failed to decrypt container keybag\n");
        free(keybag_dec); munmap(data, file_size); return false;
    }

    uint32_t obj_type;
    memcpy(&obj_type, keybag_dec + 24, 4);
    if (obj_type != KEYS_MAGIC) {
        fprintf(stderr, "  ERROR: Invalid keybag (got 0x%08x, expected 0x%08x)\n", obj_type, KEYS_MAGIC);
        free(keybag_dec); munmap(data, file_size); return false;
    }
    printf("  \u2713 Container keybag decrypted\n");

    // Parse container keybag entries
    uint16_t nkeys;
    memcpy(&nkeys, keybag_dec + 34, 2);
    size_t entry_off = 48;
    uint64_t vol_keybag_block = 0;
    uint8_t entry_uuid[16] = {0};
    bool have_vek = false;

    for (int i = 0; i < nkeys; i++) {
        if (entry_off + 24 > block_size) break;
        uint8_t this_uuid[16];
        memcpy(this_uuid, keybag_dec + entry_off, 16);
        uint16_t tag, keylen;
        memcpy(&tag, keybag_dec + entry_off + 16, 2);
        memcpy(&keylen, keybag_dec + entry_off + 18, 2);
        if (entry_off + 24 + keylen > block_size) break;
        uint8_t *edata = keybag_dec + entry_off + 24;

        if (tag == 2) { // Wrapped VEK
            kb->wrapped_vek = (uint8_t *)malloc(keylen);
            memcpy(kb->wrapped_vek, edata, keylen);
            kb->wrapped_vek_len = keylen;
            printf("  \u2713 Found wrapped VEK (%u bytes)\n", keylen);
            have_vek = true;
        } else if (tag == 3 && keylen >= 8) { // Volume keybag reference
            memcpy(&vol_keybag_block, edata, 8);
            memcpy(entry_uuid, this_uuid, 16);
            printf("  \u2713 Found volume keybag at block %llu\n", vol_keybag_block);
        }
        entry_off += (24 + keylen + 15) & ~15;
    }
    free(keybag_dec);

    if (vol_keybag_block == 0) {
        fprintf(stderr, "  ERROR: Volume keybag not found\n");
        munmap(data, file_size); return false;
    }

    // Read and decrypt volume keybag
    size_t vkb_off = partition_offset + vol_keybag_block * block_size;
    if (vkb_off + block_size > file_size) {
        fprintf(stderr, "  ERROR: Volume keybag block out of range\n");
        munmap(data, file_size); return false;
    }

    uint8_t *vol_kb_dec = (uint8_t *)malloc(block_size);
    if (!aes_xts_decrypt(vol_kb_dec, data + vkb_off, block_size, entry_uuid, vol_keybag_block)) {
        fprintf(stderr, "  ERROR: Failed to decrypt volume keybag\n");
        free(vol_kb_dec); munmap(data, file_size); return false;
    }

    memcpy(&obj_type, vol_kb_dec + 24, 4);
    if (obj_type != KEYS_MAGIC && obj_type != RECS_MAGIC) {
        printf("  WARNING: Unexpected volume keybag type 0x%08x, parsing anyway...\n", obj_type);
    }
    printf("  \u2713 Volume keybag decrypted\n");

    // Parse volume keybag for KEK parameters
    uint16_t vol_nkeys;
    memcpy(&vol_nkeys, vol_kb_dec + 34, 2);
    size_t vol_off = 48;
    bool have_salt = false, have_iter = false, have_kek = false;

    for (int j = 0; j < vol_nkeys; j++) {
        if (vol_off + 24 > block_size) break;
        uint16_t v_tag, v_keylen;
        memcpy(&v_tag, vol_kb_dec + vol_off + 16, 2);
        memcpy(&v_keylen, vol_kb_dec + vol_off + 18, 2);
        if (vol_off + 24 + v_keylen > block_size) break;
        uint8_t *v_data = vol_kb_dec + vol_off + 24;

        if (v_tag == 3) { // KEK info (DER-encoded)
            for (size_t idx = 0; idx + 2 < v_keylen; idx++) {
                if (v_data[idx] == 0x85 && v_data[idx+1] == 0x10 && idx + 2 + 16 <= v_keylen) {
                    memcpy(kb->salt, v_data + idx + 2, 16);
                    have_salt = true;
                }
                if (v_data[idx] == 0x84 && idx + 2 <= v_keylen) {
                    uint8_t length = v_data[idx+1];
                    if (length <= 8 && idx + 2 + length <= v_keylen) {
                        kb->iterations = 0;
                        for (uint8_t b = 0; b < length; b++)
                            kb->iterations = (kb->iterations << 8) | v_data[idx + 2 + b];
                    }
                    if (kb->iterations > 0) have_iter = true;
                }
                if (v_data[idx] == 0x83 && v_data[idx+1] == 0x28 && idx + 2 + 40 <= v_keylen) {
                    kb->wrapped_kek = (uint8_t *)malloc(40);
                    memcpy(kb->wrapped_kek, v_data + idx + 2, 40);
                    kb->wrapped_kek_len = 40;
                    have_kek = true;
                }
            }
        }
        vol_off += (24 + v_keylen + 15) & ~15;
    }
    free(vol_kb_dec);
    munmap(data, file_size);

    if (!have_salt || !have_iter || !have_kek || !have_vek) {
        fprintf(stderr, "  ERROR: Missing keybag parameters (salt=%d iter=%d kek=%d vek=%d)\n",
                have_salt, have_iter, have_kek, have_vek);
        return false;
    }

    printf("  \u2713 Salt: ");
    for (int i = 0; i < 16; i++) printf("%02x", kb->salt[i]);
    printf("\n");
    printf("  \u2713 Iterations: %u\n", kb->iterations);
    printf("  \u2713 Wrapped KEK: %zu bytes\n", kb->wrapped_kek_len);
    printf("  \u2713 Wrapped VEK: %zu bytes\n", kb->wrapped_vek_len);
    return true;
}

static void free_keybag(KeybagData *kb) {
    if (kb->wrapped_kek) { free(kb->wrapped_kek); kb->wrapped_kek = NULL; }
    if (kb->wrapped_vek) { free(kb->wrapped_vek); kb->wrapped_vek = NULL; }
}

// =============================================================================
// Password generators
// =============================================================================

class PasswordGenerator {
public:
    virtual ~PasswordGenerator() {}
    virtual bool nextBatch(std::vector<std::string> &batch, size_t count) = 0;
    virtual uint64_t totalCount() const = 0;
    virtual uint64_t currentIndex() const = 0;
    virtual void seekTo(uint64_t index) = 0;

    // Checkpoint
    virtual void saveState(FILE *f) const = 0;
    virtual bool loadState(FILE *f) = 0;
    virtual const char *modeName() const = 0;
};

// --- Brute Force Generator ---
class BruteForceGenerator : public PasswordGenerator {
    std::string charset;
    int min_len, max_len;
    uint64_t total;
    uint64_t index;

    // Compute total combinations
    uint64_t computeTotal() const {
        uint64_t n = 0;
        uint64_t base = charset.size();
        for (int len = min_len; len <= max_len; len++) {
            uint64_t count = 1;
            for (int i = 0; i < len; i++) count *= base;
            n += count;
        }
        return n;
    }

    // Convert absolute index to password string
    std::string indexToPassword(uint64_t idx) const {
        uint64_t base = charset.size();
        uint64_t offset = 0;
        for (int len = min_len; len <= max_len; len++) {
            uint64_t count = 1;
            for (int i = 0; i < len; i++) count *= base;
            if (idx < offset + count) {
                uint64_t rel = idx - offset;
                std::string pwd(len, ' ');
                for (int i = len - 1; i >= 0; i--) {
                    pwd[i] = charset[rel % base];
                    rel /= base;
                }
                return pwd;
            }
            offset += count;
        }
        return "";
    }

public:
    BruteForceGenerator(const std::string &cs, int minl, int maxl)
        : charset(cs), min_len(minl), max_len(maxl), index(0) {
        total = computeTotal();
    }

    bool nextBatch(std::vector<std::string> &batch, size_t count) override {
        batch.clear();
        for (size_t i = 0; i < count && index < total; i++, index++) {
            batch.push_back(indexToPassword(index));
        }
        return !batch.empty();
    }

    uint64_t totalCount() const override { return total; }
    uint64_t currentIndex() const override { return index; }
    void seekTo(uint64_t idx) override { index = idx; }
    const char *modeName() const override { return "brute"; }

    void saveState(FILE *f) const override {
        fprintf(f, "charset=%s\n", charset.c_str());
        fprintf(f, "min_length=%d\n", min_len);
        fprintf(f, "max_length=%d\n", max_len);
        fprintf(f, "current_index=%llu\n", index);
    }

    bool loadState(FILE *) override { return true; } // index set via seekTo
};

// --- Dictionary Generator ---
class DictionaryGenerator : public PasswordGenerator {
    std::vector<std::string> words;
    uint64_t index;

public:
    DictionaryGenerator(const char *wordlist_path) : index(0) {
        FILE *f = fopen(wordlist_path, "r");
        if (!f) { fprintf(stderr, "ERROR: Cannot open wordlist: %s\n", wordlist_path); return; }
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
            if (len > 0) words.push_back(std::string(line));
        }
        fclose(f);
        printf("  Loaded %zu words from %s\n", words.size(), wordlist_path);
    }

    bool nextBatch(std::vector<std::string> &batch, size_t count) override {
        batch.clear();
        for (size_t i = 0; i < count && index < words.size(); i++, index++)
            batch.push_back(words[index]);
        return !batch.empty();
    }

    uint64_t totalCount() const override { return words.size(); }
    uint64_t currentIndex() const override { return index; }
    void seekTo(uint64_t idx) override { index = idx; }
    const char *modeName() const override { return "dict"; }

    void saveState(FILE *f) const override { fprintf(f, "current_index=%llu\n", index); }
    bool loadState(FILE *) override { return true; }

    const std::vector<std::string> &getWords() const { return words; }
};

// --- Rule Mutation Generator ---
class RuleMutationGenerator : public PasswordGenerator {
    std::vector<std::string> words;
    std::vector<std::string> mutations_cache; // flattened: all mutations of all words
    uint64_t index;

    static void generateMutations(const std::string &word, std::vector<std::string> &out) {
        out.push_back(word); // original

        // Capitalize first letter
        if (!word.empty()) {
            std::string cap = word;
            cap[0] = toupper(cap[0]);
            if (cap != word) out.push_back(cap);
        }

        // UPPERCASE
        { std::string u = word; for (auto &c : u) c = toupper(c); if (u != word) out.push_back(u); }

        // lowercase
        { std::string l = word; for (auto &c : l) c = tolower(c); if (l != word) out.push_back(l); }

        // Toggle case
        {
            std::string t = word;
            for (auto &c : t) c = islower(c) ? toupper(c) : tolower(c);
            if (t != word) out.push_back(t);
        }

        // Leet speak
        {
            std::string leet = word;
            bool changed = false;
            for (auto &c : leet) {
                char orig = c;
                switch (tolower(c)) {
                    case 'a': c = '@'; break;
                    case 'e': c = '3'; break;
                    case 'o': c = '0'; break;
                    case 's': c = '$'; break;
                    case 'i': c = '1'; break;
                    case 't': c = '7'; break;
                    default: break;
                }
                if (c != orig) changed = true;
            }
            if (changed) out.push_back(leet);
        }

        // Append single digits 0-9
        for (char d = '0'; d <= '9'; d++)
            out.push_back(word + d);

        // Append double digits 00-99
        for (int d = 0; d < 100; d++) {
            char buf[4]; snprintf(buf, sizeof(buf), "%02d", d);
            out.push_back(word + buf);
        }

        // Append common years
        const int years[] = {1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,
                             2000,2001,2002,2020,2021,2022,2023,2024,2025,2026};
        for (int y : years) {
            char buf[8]; snprintf(buf, sizeof(buf), "%d", y);
            out.push_back(word + buf);
        }

        // Append common symbols
        const char *syms = "!@#$%^&*";
        for (int s = 0; syms[s]; s++)
            out.push_back(word + syms[s]);

        // Prepend digits 0-9
        for (char d = '0'; d <= '9'; d++)
            out.push_back(std::string(1, d) + word);

        // Reverse
        { std::string r(word.rbegin(), word.rend()); if (r != word) out.push_back(r); }

        // Duplicate
        out.push_back(word + word);
    }

public:
    RuleMutationGenerator(const char *wordlist_path) : index(0) {
        FILE *f = fopen(wordlist_path, "r");
        if (!f) { fprintf(stderr, "ERROR: Cannot open wordlist: %s\n", wordlist_path); return; }
        char line[4096];
        while (fgets(line, sizeof(line), f)) {
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
            if (len > 0) words.push_back(std::string(line));
        }
        fclose(f);

        // Pre-generate all mutations
        for (const auto &w : words)
            generateMutations(w, mutations_cache);

        printf("  Loaded %zu words, generated %zu mutations\n", words.size(), mutations_cache.size());
    }

    bool nextBatch(std::vector<std::string> &batch, size_t count) override {
        batch.clear();
        for (size_t i = 0; i < count && index < mutations_cache.size(); i++, index++)
            batch.push_back(mutations_cache[index]);
        return !batch.empty();
    }

    uint64_t totalCount() const override { return mutations_cache.size(); }
    uint64_t currentIndex() const override { return index; }
    void seekTo(uint64_t idx) override { index = idx; }
    const char *modeName() const override { return "rules"; }

    void saveState(FILE *f) const override { fprintf(f, "current_index=%llu\n", index); }
    bool loadState(FILE *) override { return true; }
};

// =============================================================================
// GPU PBKDF2 (Metal)
// =============================================================================

class MetalPBKDF2 {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> computePipeline;
    id<MTLBuffer> cachedSaltBlockBuffer;
    NSUInteger optimalThreadGroupSize;
    size_t saltBlockLength;
    uint32_t iterations;
    bool gpuAvailable;

    // Double-buffered: two buffer sets for pipelining
    static const int NUM_SLOTS = 2;
    struct BufferSlot {
        id<MTLBuffer> passwordBuffer;
        id<MTLBuffer> outputBuffer;
        id<MTLCommandBuffer> cmdBuf;
        size_t pwdBufSize;
        size_t outBufSize;
        size_t batchSize;
        bool inflight;
    };
    BufferSlot slots[NUM_SLOTS];
    int currentSlot;

    void ensureBuffers(int slot, size_t pwdBufSize, size_t outBufSize) {
        auto &s = slots[slot];
        if (!s.passwordBuffer || s.pwdBufSize < pwdBufSize) {
            s.pwdBufSize = pwdBufSize;
            s.passwordBuffer = [device newBufferWithLength:pwdBufSize
                                                  options:MTLResourceStorageModeShared];
        }
        if (!s.outputBuffer || s.outBufSize < outBufSize) {
            s.outBufSize = outBufSize;
            s.outputBuffer = [device newBufferWithLength:outBufSize
                                                options:MTLResourceStorageModeShared];
        }
    }

public:
    MetalPBKDF2(const uint8_t *salt, size_t salt_len, uint32_t iters)
        : cachedSaltBlockBuffer(nil), iterations(iters), gpuAvailable(false), currentSlot(0) {

        for (int i = 0; i < NUM_SLOTS; i++) {
            slots[i] = {nil, nil, nil, 0, 0, 0, false};
        }

        device = MTLCreateSystemDefaultDevice();
        if (!device) return;
        commandQueue = [device newCommandQueue];
        if (!commandQueue) return;

        NSError *error = nil;
        id<MTLLibrary> library = nil;

        NSString *currentDir = [[NSFileManager defaultManager] currentDirectoryPath];
        NSString *metallib_path = [currentDir stringByAppendingPathComponent:@"pbkdf2.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:metallib_path]) {
            NSURL *url = [NSURL fileURLWithPath:metallib_path];
            library = [device newLibraryWithURL:url error:&error];
        }
        if (!library) {
            NSString *metal_path = [currentDir stringByAppendingPathComponent:@"pbkdf2.metal"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:metal_path]) {
                NSString *src = [NSString stringWithContentsOfFile:metal_path
                                                         encoding:NSUTF8StringEncoding error:&error];
                if (src) library = [device newLibraryWithSource:src options:nil error:&error];
            }
        }
        if (!library) { NSLog(@"Metal shader not found"); return; }

        id<MTLFunction> function = [library newFunctionWithName:@"pbkdf2_sha256_kernel"];
        if (!function) return;

        computePipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!computePipeline) return;

        optimalThreadGroupSize = [computePipeline threadExecutionWidth];

        uint8_t salt_block[64] = {0};
        memcpy(salt_block, salt, salt_len);
        salt_block[salt_len]   = 0x00;
        salt_block[salt_len+1] = 0x00;
        salt_block[salt_len+2] = 0x00;
        salt_block[salt_len+3] = 0x01;
        saltBlockLength = salt_len + 4;

        cachedSaltBlockBuffer = [device newBufferWithBytes:salt_block length:64
                                                  options:MTLResourceStorageModeShared];
        gpuAvailable = true;
    }

    bool isAvailable() const { return gpuAvailable; }

    // Submit a batch to GPU asynchronously (fills buffer and dispatches).
    // Call waitBatch() to get results. Double-buffered: can overlap dispatch with fill.
    void submitBatch(const std::vector<std::string> &passwords) {
        if (!gpuAvailable || passwords.empty()) return;

        auto &s = slots[currentSlot];

        // Wait for previous use of this slot if still inflight
        if (s.inflight && s.cmdBuf) {
            [s.cmdBuf waitUntilCompleted];
            s.inflight = false;
        }

        size_t max_pwd_len = 0;
        for (const auto &p : passwords)
            max_pwd_len = std::max(max_pwd_len, p.length());
        if (max_pwd_len > 128) max_pwd_len = 128;
        if (max_pwd_len == 0) max_pwd_len = 1;

        size_t pwdBufSize = passwords.size() * max_pwd_len;
        size_t outBufSize = passwords.size() * SHA256_DIGEST_SIZE;
        ensureBuffers(currentSlot, pwdBufSize, outBufSize);

        uint8_t *ptr = (uint8_t *)[s.passwordBuffer contents];
        for (size_t i = 0; i < passwords.size(); i++) {
            size_t len = passwords[i].length();
            memcpy(ptr + i * max_pwd_len, passwords[i].c_str(), len);
            if (len < max_pwd_len)
                memset(ptr + i * max_pwd_len + len, 0, max_pwd_len - len);
        }

        uint32_t pwd_len_val = (uint32_t)max_pwd_len;
        uint32_t salt_block_len_val = (uint32_t)saltBlockLength;
        uint32_t iters_val = iterations;
        uint32_t num_val = (uint32_t)passwords.size();

        s.cmdBuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [s.cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:computePipeline];
        [enc setBuffer:s.passwordBuffer offset:0 atIndex:0];
        [enc setBuffer:cachedSaltBlockBuffer offset:0 atIndex:1];
        [enc setBuffer:s.outputBuffer offset:0 atIndex:2];
        [enc setBytes:&pwd_len_val length:4 atIndex:3];
        [enc setBytes:&salt_block_len_val length:4 atIndex:4];
        [enc setBytes:&iters_val length:4 atIndex:5];
        [enc setBytes:&num_val length:4 atIndex:6];

        NSUInteger numTG = (passwords.size() + optimalThreadGroupSize - 1) / optimalThreadGroupSize;
        [enc dispatchThreadgroups:MTLSizeMake(numTG, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(optimalThreadGroupSize, 1, 1)];
        [enc endEncoding];
        [s.cmdBuf commit];

        s.batchSize = passwords.size();
        s.inflight = true;
        currentSlot = (currentSlot + 1) % NUM_SLOTS;
    }

    // Wait for submitted batch to complete. Returns pointer to raw output buffer
    // and batch size. Avoids heap allocation of per-result vectors.
    // Caller reads results directly: outPtr + i * SHA256_DIGEST_SIZE.
    bool waitBatch(int slot, uint8_t *&outPtr, size_t &batchSize) {
        auto &s = slots[slot];
        if (!s.inflight || !s.cmdBuf) { outPtr = nullptr; batchSize = 0; return false; }

        [s.cmdBuf waitUntilCompleted];
        s.inflight = false;

        outPtr = (uint8_t *)[s.outputBuffer contents];
        batchSize = s.batchSize;
        return true;
    }

    // Convenience: synchronous deriveBatch (for backward compatibility)
    std::vector<std::vector<uint8_t>> deriveBatch(const std::vector<std::string> &passwords) {
        if (!gpuAvailable || passwords.empty()) return {};
        int slot = currentSlot;
        submitBatch(passwords);
        uint8_t *outPtr = nullptr;
        size_t batchSize = 0;
        waitBatch(slot, outPtr, batchSize);
        std::vector<std::vector<uint8_t>> results;
        if (outPtr) {
            for (size_t i = 0; i < batchSize; i++)
                results.emplace_back(outPtr + i * SHA256_DIGEST_SIZE,
                                     outPtr + (i + 1) * SHA256_DIGEST_SIZE);
        }
        return results;
    }

    int getCurrentSlot() const { return currentSlot; }
};

// =============================================================================
// Password tester (shared between GPU and CPU workers)
// =============================================================================

static bool try_unwrap(const uint8_t *derived_key,
                       const uint8_t *wrapped_kek, size_t wrapped_kek_len,
                       const uint8_t *wrapped_vek_blob, size_t wrapped_vek_blob_len) {
    uint8_t kek[32];
    if (!aes_unwrap_key(kek, wrapped_kek, wrapped_kek_len, derived_key, 32))
        return false;

    // Parse VEK blob for 0x83 0x28 tag
    uint8_t wrapped_vek[64];
    size_t wrapped_vek_len = 0;
    if (wrapped_vek_blob_len >= 42) {
        for (size_t i = 0; i + 2 < wrapped_vek_blob_len; i++) {
            if (wrapped_vek_blob[i] == 0x83 && wrapped_vek_blob[i+1] == 0x28 &&
                i + 2 + 40 <= wrapped_vek_blob_len) {
                memcpy(wrapped_vek, wrapped_vek_blob + i + 2, 40);
                wrapped_vek_len = 40;
                break;
            }
        }
        if (wrapped_vek_len == 0 && wrapped_vek_blob_len >= 40) {
            memcpy(wrapped_vek, wrapped_vek_blob, 40);
            wrapped_vek_len = 40;
        }
    }
    if (wrapped_vek_len == 0) return false;

    uint8_t vek[32];
    return aes_unwrap_key(vek, wrapped_vek, wrapped_vek_len, kek, 32);
}

// =============================================================================
// Worker threading
// =============================================================================

struct SharedState {
    PasswordGenerator *generator;
    MetalPBKDF2 *gpu;
    KeybagData *kb;
    std::atomic<bool> found;
    std::atomic<uint64_t> tested;
    char found_password[MAX_PASSWORD_LEN];
    pthread_mutex_t gen_mutex;  // protects generator
    pthread_mutex_t found_mutex;
    int gpu_batch_size;
    volatile sig_atomic_t *interrupted;
};

static void *gpu_worker(void *arg) {
    SharedState *st = (SharedState *)arg;

    // Double-buffered GPU pipeline:
    // 1. Submit batch A to GPU
    // 2. While GPU computes A: fetch next batch B, fill buffer B
    // 3. Wait for A results, submit B, check A results
    // This overlaps GPU computation with batch generation + result checking.

    // Prime the pipeline: submit first batch
    std::vector<std::string> inflight_batch;
    int inflight_slot = -1;
    {
        pthread_mutex_lock(&st->gen_mutex);
        st->generator->nextBatch(inflight_batch, st->gpu_batch_size);
        pthread_mutex_unlock(&st->gen_mutex);
        if (inflight_batch.empty()) return NULL;
        inflight_slot = st->gpu->getCurrentSlot();
        st->gpu->submitBatch(inflight_batch);
    }

    while (!st->found.load() && !*st->interrupted) {
        // Fetch next batch while GPU processes current one
        std::vector<std::string> next_batch;
        pthread_mutex_lock(&st->gen_mutex);
        st->generator->nextBatch(next_batch, st->gpu_batch_size);
        pthread_mutex_unlock(&st->gen_mutex);

        // Submit next batch (if any) before waiting for current results
        int next_slot = -1;
        if (!next_batch.empty() && !st->found.load() && !*st->interrupted) {
            next_slot = st->gpu->getCurrentSlot();
            st->gpu->submitBatch(next_batch);
        }

        // Now wait for inflight results and check them (zero-copy from GPU buffer)
        uint8_t *outPtr = nullptr;
        size_t batchSize = 0;
        st->gpu->waitBatch(inflight_slot, outPtr, batchSize);
        if (outPtr) {
            for (size_t i = 0; i < batchSize; i++) {
                if (st->found.load()) break;
                if (try_unwrap(outPtr + i * SHA256_DIGEST_SIZE,
                               st->kb->wrapped_kek, st->kb->wrapped_kek_len,
                               st->kb->wrapped_vek, st->kb->wrapped_vek_len)) {
                    pthread_mutex_lock(&st->found_mutex);
                    if (!st->found.load()) {
                        st->found.store(true);
                        strncpy(st->found_password, inflight_batch[i].c_str(), MAX_PASSWORD_LEN - 1);
                    }
                    pthread_mutex_unlock(&st->found_mutex);
                    break;
                }
            }
        }
        st->tested.fetch_add(inflight_batch.size());

        // Rotate: next becomes inflight
        if (next_batch.empty() || st->found.load() || *st->interrupted) break;
        inflight_batch = std::move(next_batch);
        inflight_slot = next_slot;
    }
    return NULL;
}

static void *cpu_worker(void *arg) {
    SharedState *st = (SharedState *)arg;
    uint8_t batch_out[CPU_BATCH_SIZE * SHA256_DIGEST_SIZE];
    const char *batch_pwds[CPU_BATCH_SIZE];
    size_t batch_lens[CPU_BATCH_SIZE];

    while (!st->found.load() && !*st->interrupted) {
        std::vector<std::string> batch;
        pthread_mutex_lock(&st->gen_mutex);
        st->generator->nextBatch(batch, CPU_BATCH_SIZE);
        pthread_mutex_unlock(&st->gen_mutex);
        if (batch.empty()) break;

        for (size_t i = 0; i < batch.size(); i++) {
            batch_pwds[i] = batch[i].c_str();
            batch_lens[i] = batch[i].length();
        }

        pbkdf2_sha256_cpu_batch(batch_out, batch_pwds, batch_lens, batch.size(),
                                st->kb->salt, 16, st->kb->iterations);

        for (size_t i = 0; i < batch.size(); i++) {
            if (st->found.load()) break;
            if (try_unwrap(batch_out + i * SHA256_DIGEST_SIZE,
                           st->kb->wrapped_kek, st->kb->wrapped_kek_len,
                           st->kb->wrapped_vek, st->kb->wrapped_vek_len)) {
                pthread_mutex_lock(&st->found_mutex);
                if (!st->found.load()) {
                    st->found.store(true);
                    strncpy(st->found_password, batch[i].c_str(), MAX_PASSWORD_LEN - 1);
                }
                pthread_mutex_unlock(&st->found_mutex);
                break;
            }
        }
        st->tested.fetch_add(batch.size());
    }
    return NULL;
}

// =============================================================================
// Progress reporter
// =============================================================================

struct ProgressData {
    SharedState *state;
    uint64_t total;
    struct timeval start_time;
};

static void *progress_thread(void *arg) {
    ProgressData *pd = (ProgressData *)arg;
    while (!pd->state->found.load() && !*pd->state->interrupted) {
        sleep(PROGRESS_INTERVAL);
        if (pd->state->found.load() || *pd->state->interrupted) break;

        uint64_t tested = pd->state->tested.load();
        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - pd->start_time.tv_sec) +
                         (now.tv_usec - pd->start_time.tv_usec) / 1e6;
        double rate = (elapsed > 0) ? tested / elapsed : 0;

        int hrs = (int)elapsed / 3600;
        int mins = ((int)elapsed % 3600) / 60;
        int secs = (int)elapsed % 60;

        if (pd->total > 0) {
            double pct = 100.0 * tested / pd->total;
            double remaining = (rate > 0) ? (pd->total - tested) / rate : 0;
            int eta_h = (int)remaining / 3600;
            int eta_m = ((int)remaining % 3600) / 60;
            int eta_s = (int)remaining % 60;
            fprintf(stderr, "\r[%02d:%02d:%02d] %llu / %llu (%.1f%%) | %.1f pwd/s | ETA: %02d:%02d:%02d   ",
                    hrs, mins, secs, tested, pd->total, pct, rate, eta_h, eta_m, eta_s);
        } else {
            fprintf(stderr, "\r[%02d:%02d:%02d] %llu tested | %.1f pwd/s   ",
                    hrs, mins, secs, tested, rate);
        }
        fflush(stderr);
    }
    return NULL;
}

// =============================================================================
// Checkpoint
// =============================================================================

static void save_checkpoint(const char *path, const char *image_path,
                            PasswordGenerator *gen, uint64_t tested, double elapsed) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "APFS_BF_STATE_V1\n");
    fprintf(f, "image=%s\n", image_path);
    fprintf(f, "mode=%s\n", gen->modeName());
    fprintf(f, "passwords_tested=%llu\n", tested);
    fprintf(f, "elapsed_seconds=%.2f\n", elapsed);
    gen->saveState(f);
    fclose(f);
}

static bool load_checkpoint(const char *path, uint64_t *out_index, uint64_t *out_tested,
                            double *out_elapsed, const char *expected_image) {
    FILE *f = fopen(path, "r");
    if (!f) return false;

    char line[4096];
    *out_index = 0; *out_tested = 0; *out_elapsed = 0;
    bool valid = false;

    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n")] = 0;
        if (strncmp(line, "image=", 6) == 0) {
            if (strcmp(line + 6, expected_image) != 0) {
                fprintf(stderr, "ERROR: Checkpoint image mismatch\n");
                fclose(f); return false;
            }
            valid = true;
        }
        if (strncmp(line, "current_index=", 14) == 0)
            *out_index = strtoull(line + 14, NULL, 10);
        if (strncmp(line, "passwords_tested=", 17) == 0)
            *out_tested = strtoull(line + 17, NULL, 10);
        if (strncmp(line, "elapsed_seconds=", 16) == 0)
            *out_elapsed = atof(line + 16);
    }
    fclose(f);
    return valid;
}

// =============================================================================
// Main run logic
// =============================================================================

static const char *run_attack(PasswordGenerator *gen, KeybagData *kb,
                              int num_workers, int gpu_batch_size, bool use_gpu,
                              const char *checkpoint_path, const char *image_path,
                              uint64_t resume_tested, double resume_elapsed) {
    SharedState state;
    state.generator = gen;
    state.kb = kb;
    state.found.store(false);
    state.tested.store(resume_tested);
    state.found_password[0] = 0;
    state.gpu_batch_size = gpu_batch_size;
    state.interrupted = &g_interrupted;
    pthread_mutex_init(&state.gen_mutex, NULL);
    pthread_mutex_init(&state.found_mutex, NULL);

    // Initialize GPU
    MetalPBKDF2 *gpu = nullptr;
    if (use_gpu) {
        gpu = new MetalPBKDF2(kb->salt, 16, kb->iterations);
        if (!gpu->isAvailable()) {
            printf("  GPU not available, using CPU only\n");
            delete gpu; gpu = nullptr;
        } else {
            printf("  \u2713 GPU acceleration enabled\n");
        }
    }
    state.gpu = gpu;

    printf("\nStarting attack (%s)...\n", gen->modeName());
    printf("  Total passwords: %llu\n", gen->totalCount());
    printf("  Workers: %d CPU", num_workers);
    if (gpu) printf(" + GPU (batch %d)", gpu_batch_size);
    printf("\n\n");

    // Progress thread
    ProgressData pd;
    pd.state = &state;
    pd.total = gen->totalCount();
    gettimeofday(&pd.start_time, NULL);
    // Adjust start time for resumed elapsed
    pd.start_time.tv_sec -= (long)resume_elapsed;

    pthread_t prog_tid;
    pthread_create(&prog_tid, NULL, progress_thread, &pd);

    // GPU thread
    pthread_t gpu_tid = 0;
    if (gpu) pthread_create(&gpu_tid, NULL, gpu_worker, &state);

    // CPU threads
    std::vector<pthread_t> cpu_tids(num_workers);
    for (int i = 0; i < num_workers; i++)
        pthread_create(&cpu_tids[i], NULL, cpu_worker, &state);

    // Wait for completion
    if (gpu_tid) pthread_join(gpu_tid, NULL);
    for (auto &t : cpu_tids) pthread_join(t, NULL);

    // Cancel progress thread
    pthread_cancel(prog_tid);
    pthread_join(prog_tid, NULL);
    fprintf(stderr, "\r%80s\r", ""); // clear progress line

    // Final stats
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    double elapsed = (end_time.tv_sec - pd.start_time.tv_sec) +
                     (end_time.tv_usec - pd.start_time.tv_usec) / 1e6;
    uint64_t tested = state.tested.load();
    double rate = (elapsed > 0) ? tested / elapsed : 0;

    printf("\nResults:\n");
    printf("  Time: %.2f seconds\n", elapsed);
    printf("  Passwords tested: %llu\n", tested);
    printf("  Rate: %.1f passwords/second\n", rate);

    // Save checkpoint if interrupted
    if (g_interrupted && checkpoint_path) {
        save_checkpoint(checkpoint_path, image_path, gen, tested, elapsed);
        printf("  Checkpoint saved: %s\n", checkpoint_path);
    }

    pthread_mutex_destroy(&state.gen_mutex);
    pthread_mutex_destroy(&state.found_mutex);
    if (gpu) delete gpu;

    if (state.found.load()) {
        static char result[MAX_PASSWORD_LEN];
        strncpy(result, state.found_password, MAX_PASSWORD_LEN);
        return result;
    }
    return NULL;
}

// =============================================================================
// CLI
// =============================================================================

static void print_usage(const char *prog) {
    fprintf(stderr,
        "APFS Password Brute Forcer\n"
        "==============================\n\n"
        "Usage: %s <dmg_file> [options]\n\n"
        "Attack Modes:\n"
        "  --brute               Exhaustive brute force (default)\n"
        "  --dict <wordlist>     Dictionary attack from wordlist file\n"
        "  --rules               Apply mutation rules to dictionary words\n"
        "                        (requires --dict)\n\n"
        "Brute Force Options:\n"
        "  --min-length N        Minimum password length (default: 1)\n"
        "  --max-length N        Maximum password length (default: 4)\n"
        "  --charset TYPE        Character set: lower, upper, digits,\n"
        "                        alphanumeric, lower-digits, upper-digits,\n"
        "                        all, or a custom string\n\n"
        "Performance:\n"
        "  --workers N           CPU worker threads (default: %d)\n"
        "  --gpu-batch N         GPU batch size (default: %d)\n"
        "  --no-gpu              Disable GPU acceleration\n"
        "  --processes N         Spawn N child processes (default: 1)\n\n"
        "Resume:\n"
        "  --resume <file>       Resume from checkpoint file\n"
        "  --checkpoint <file>   Checkpoint path (default: .apfs_bf_checkpoint)\n\n"
        "Other:\n"
        "  --test-password PWD   Test a single password\n"
        "  --help                Show this help\n\n"
        "Examples:\n"
        "  %s encrypted.dmg --charset lower --max-length 6\n"
        "  %s encrypted.dmg --dict wordlist.txt --rules\n"
        "  %s encrypted.dmg --resume .apfs_bf_checkpoint\n"
        "  %s encrypted.dmg --processes 4 --charset alphanumeric\n",
        prog, (int)sysconf(_SC_NPROCESSORS_ONLN), DEFAULT_GPU_BATCH_SIZE,
        prog, prog, prog, prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    // Parse arguments
    const char *image_path = NULL;
    const char *dict_path = NULL;
    const char *test_password = NULL;
    const char *resume_path = NULL;
    const char *checkpoint_path = ".apfs_bf_checkpoint";
    const char *charset_str = CHARSET_ALPHANUMERIC;
    int min_length = 1, max_length = 4;
    int num_workers = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int gpu_batch_size = DEFAULT_GPU_BATCH_SIZE;
    int num_processes = 1;
    bool use_gpu = true;
    bool mode_rules = false;
    bool show_help = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else if (strcmp(argv[i], "--brute") == 0) {
            // default mode, no-op
        } else if (strcmp(argv[i], "--dict") == 0 && i+1 < argc) {
            dict_path = argv[++i];
        } else if (strcmp(argv[i], "--rules") == 0) {
            mode_rules = true;
        } else if (strcmp(argv[i], "--min-length") == 0 && i+1 < argc) {
            min_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-length") == 0 && i+1 < argc) {
            max_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--charset") == 0 && i+1 < argc) {
            const char *t = argv[++i];
            if (strcmp(t, "lower") == 0) charset_str = CHARSET_LOWER;
            else if (strcmp(t, "upper") == 0) charset_str = CHARSET_UPPER;
            else if (strcmp(t, "digits") == 0) charset_str = CHARSET_DIGITS;
            else if (strcmp(t, "alphanumeric") == 0) charset_str = CHARSET_ALPHANUMERIC;
            else if (strcmp(t, "lower-digits") == 0) charset_str = CHARSET_LOWER_DIGITS;
            else if (strcmp(t, "upper-digits") == 0) charset_str = CHARSET_UPPER_DIGITS;
            else if (strcmp(t, "all") == 0) charset_str = CHARSET_ALL;
            else charset_str = t; // custom
        } else if (strcmp(argv[i], "--workers") == 0 && i+1 < argc) {
            num_workers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-batch") == 0 && i+1 < argc) {
            gpu_batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-gpu") == 0) {
            use_gpu = false;
        } else if (strcmp(argv[i], "--processes") == 0 && i+1 < argc) {
            num_processes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--test-password") == 0 && i+1 < argc) {
            test_password = argv[++i];
        } else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) {
            resume_path = argv[++i];
        } else if (strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) {
            checkpoint_path = argv[++i];
        } else if (argv[i][0] != '-' && !image_path) {
            image_path = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n\n", argv[i]);
            print_usage(argv[0]); return 1;
        }
    }

    if (show_help) { print_usage(argv[0]); return 0; }
    if (!image_path) { fprintf(stderr, "ERROR: No image file specified\n\n"); print_usage(argv[0]); return 1; }

    if (mode_rules && !dict_path) {
        fprintf(stderr, "ERROR: --rules requires --dict <wordlist>\n");
        return 1;
    }

    // Install signal handler
    struct sigaction sa;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    // Print banner
    printf("APFS Password Brute Forcer\n");
    printf("==============================\n");
    printf("Image: %s\n", image_path);

    // Extract keybag
    printf("\nExtracting keybag data...\n");
    KeybagData kb;
    if (!extract_keybag_native(image_path, &kb)) {
        fprintf(stderr, "ERROR: Failed to extract keybag data\n");
        return 1;
    }

    // Test single password
    if (test_password) {
        printf("\nTesting password '%s'...\n", test_password);
        uint8_t derived[32];
        pbkdf2_sha256_cpu(derived, (const uint8_t *)test_password, strlen(test_password),
                          kb.salt, 16, kb.iterations);
        if (try_unwrap(derived, kb.wrapped_kek, kb.wrapped_kek_len,
                       kb.wrapped_vek, kb.wrapped_vek_len)) {
            printf("\n\u2705 PASSWORD FOUND: '%s'\n", test_password);
            free_keybag(&kb); return 0;
        } else {
            printf("\n\u274C Password '%s' is NOT correct\n", test_password);
            free_keybag(&kb); return 1;
        }
    }

    // Create generator
    PasswordGenerator *gen = nullptr;
    if (dict_path && mode_rules) {
        gen = new RuleMutationGenerator(dict_path);
    } else if (dict_path) {
        gen = new DictionaryGenerator(dict_path);
    } else {
        gen = new BruteForceGenerator(charset_str, min_length, max_length);
        printf("Password length: %d-%d characters\n", min_length, max_length);
        printf("Character set: %s (%zu chars)\n", charset_str, strlen(charset_str));
    }

    if (gen->totalCount() == 0) {
        fprintf(stderr, "ERROR: No passwords to test\n");
        delete gen; free_keybag(&kb); return 1;
    }

    // Handle resume
    uint64_t resume_tested = 0;
    double resume_elapsed = 0;
    if (resume_path) {
        uint64_t resume_index = 0;
        if (load_checkpoint(resume_path, &resume_index, &resume_tested, &resume_elapsed, image_path)) {
            gen->seekTo(resume_index);
            printf("  Resumed from checkpoint: index %llu, %llu tested, %.1fs elapsed\n",
                   resume_index, resume_tested, resume_elapsed);
        } else {
            fprintf(stderr, "WARNING: Could not load checkpoint, starting from beginning\n");
        }
    }

    // Multi-process mode
    if (num_processes > 1 && !dict_path) {
        // Divide CPU threads evenly among processes
        int workers_per_process = std::max(1, num_workers / num_processes);
        printf("  Multi-process: %d processes × %d threads = %d total threads\n",
               num_processes, workers_per_process, num_processes * workers_per_process);
        uint64_t total = gen->totalCount();
        uint64_t per_process = total / num_processes;

        // Shared result file
        char result_file[] = "/tmp/apfs_bf_result_XXXXXX";
        int rfd = mkstemp(result_file);
        if (rfd >= 0) close(rfd);

        std::vector<pid_t> children;
        for (int p = 0; p < num_processes; p++) {
            pid_t pid = fork();
            if (pid == 0) {
                // Child process
                uint64_t start = p * per_process;
                uint64_t count __attribute__((unused)) = (p == num_processes - 1) ? (total - start) : per_process;

                PasswordGenerator *child_gen = new BruteForceGenerator(charset_str, min_length, max_length);
                child_gen->seekTo(start);

                const char *result = run_attack(child_gen, &kb, workers_per_process, gpu_batch_size,
                                                use_gpu, NULL, image_path, 0, 0);
                if (result) {
                    FILE *rf = fopen(result_file, "w");
                    if (rf) { fprintf(rf, "%s", result); fclose(rf); }
                    delete child_gen; free_keybag(&kb);
                    _exit(0);
                }
                delete child_gen; free_keybag(&kb);
                _exit(1);
            } else if (pid > 0) {
                children.push_back(pid);
            }
        }

        // Parent: wait for any child to succeed
        int found = 0;
        while (!children.empty()) {
            int status;
            pid_t done = waitpid(-1, &status, 0);
            if (done > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                found = 1;
                // Kill remaining children
                for (pid_t c : children) {
                    if (c != done) kill(c, SIGTERM);
                }
                break;
            }
            children.erase(std::remove(children.begin(), children.end(), done), children.end());
        }

        if (found) {
            char pwd[MAX_PASSWORD_LEN] = {0};
            FILE *rf = fopen(result_file, "r");
            if (rf) { fgets(pwd, sizeof(pwd), rf); fclose(rf); }
            unlink(result_file);
            printf("\n\u2705 PASSWORD FOUND: '%s'\n", pwd);
            delete gen; free_keybag(&kb); return 0;
        }
        unlink(result_file);
        printf("\n\u274C Password not found\n");
        delete gen; free_keybag(&kb); return 1;
    }

    // Single process mode
    const char *result = run_attack(gen, &kb, num_workers, gpu_batch_size,
                                    use_gpu, checkpoint_path, image_path,
                                    resume_tested, resume_elapsed);

    if (result) {
        printf("\n\u2705 PASSWORD FOUND: '%s'\n", result);
        // Remove checkpoint on success
        if (checkpoint_path) unlink(checkpoint_path);
        delete gen; free_keybag(&kb); return 0;
    }

    if (g_interrupted) {
        printf("\nInterrupted. Use --resume %s to continue.\n", checkpoint_path);
    } else {
        printf("\n\u274C Password not found\n");
    }

    delete gen;
    free_keybag(&kb);
    return 1;
}
