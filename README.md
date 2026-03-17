# APFS Password Brute Forcer

Brute force tool for encrypted APFS volumes (disk images). Uses Metal GPU compute shaders and ARM SHA2 hardware intrinsics for maximum throughput on Apple Silicon.

## Performance

Benchmarked on M4 Max (40 GPU cores, 16 CPU cores, 128GB RAM) against APFS volumes with 100,000 PBKDF2-SHA256 iterations:

| Mode | Rate |
|------|------|
| GPU + CPU (16 workers) | ~13,100 pwd/s |
| CPU-only (16 workers) | ~4,190 pwd/s |
| GPU-only | ~9,500 pwd/s |

### What that means in practice

| Password space | Candidates | Time (GPU+CPU) |
|---------------|-----------|----------------|
| 4 chars, lowercase | 460k | ~35 seconds |
| 5 chars, lowercase | 12M | ~15 minutes |
| 6 chars, lowercase | 310M | ~6.5 hours |
| 4 chars, alphanumeric | 15M | ~19 minutes |
| 14M word dictionary | 14M | ~18 minutes |

## Requirements

- macOS 13+ with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- No external dependencies (uses Metal, CommonCrypto, Security frameworks)

## Build

```bash
make brute-force-pro
```

This produces `apfs_brute_force_pro`. The Metal shader (`pbkdf2.metal`) is compiled at runtime.

## Usage

### Test a known password

```bash
./apfs_brute_force_pro encrypted.dmg --test-password mypassword
```

### Brute force attack

```bash
# Lowercase, up to 5 characters
./apfs_brute_force_pro encrypted.dmg --charset lower --max-length 5

# Alphanumeric, 4-6 characters
./apfs_brute_force_pro encrypted.dmg --charset alphanumeric --min-length 4 --max-length 6

# Custom character set
./apfs_brute_force_pro encrypted.dmg --charset "abc123!@#" --max-length 8
```

### Dictionary attack

```bash
# Basic dictionary
./apfs_brute_force_pro encrypted.dmg --dict wordlist.txt

# Dictionary with mutation rules (capitalize, append digits, leet speak, etc.)
./apfs_brute_force_pro encrypted.dmg --dict wordlist.txt --rules
```

### Options

```
Attack Modes:
  --brute               Exhaustive brute force (default)
  --dict <wordlist>     Dictionary attack from wordlist file
  --rules               Apply mutation rules to dictionary words

Brute Force Options:
  --min-length N        Minimum password length (default: 1)
  --max-length N        Maximum password length (default: 4)
  --charset TYPE        lower, upper, digits, alphanumeric,
                        lower-digits, upper-digits, all, or custom string

Performance:
  --workers N           CPU worker threads (default: auto-detected)
  --gpu-batch N         GPU batch size (default: 32768)
  --no-gpu              Disable GPU acceleration
  --processes N         Multi-process mode for multi-machine scaling

Resume:
  --resume <file>       Resume from checkpoint file
  --checkpoint <file>   Checkpoint path (default: .apfs_bf_checkpoint)
```

## How it works

### Key extraction (no password needed)

APFS keybags are encrypted with volume/container UUIDs, which are stored in plaintext:

1. Read container superblock, get container UUID
2. Decrypt container keybag with UUID (AES-XTS)
3. Extract wrapped VEK and volume keybag reference
4. Decrypt volume keybag with entry UUID
5. Extract PBKDF2 parameters: salt (16 bytes), iteration count, wrapped KEK (40 bytes)

### Password testing

For each candidate password:

1. PBKDF2-HMAC-SHA256(password, salt, 100k iterations) -> 32-byte derived key
2. AES Key Unwrap (RFC 3394) the wrapped KEK with derived key
3. If IV check passes (0xA6A6A6A6A6A6A6A6) -> unwrap VEK with KEK -> password found

Step 1 is the bottleneck: 200,000 sequential SHA256 compressions per password.

### Optimizations

**GPU (Metal compute shader — `pbkdf2.metal`):**
- HMAC midstate caching: pre-computes ipad/opad SHA256 states once per password, eliminating 200k redundant compressions
- 16-word circular buffer for SHA256 message schedule (64 bytes vs 256 bytes per thread)
- Word-form computation throughout: no byte-to-word endian conversions in the hot loop
- Minimal per-thread storage (~224 bytes) for maximum GPU occupancy
- Double-buffered dispatch: overlaps GPU compute with batch prep and result checking

**CPU (ARM SHA2 intrinsics — `apfs_brute_force_pro.mm`):**
- Hardware SHA256 using `vsha256hq_u32` / `vsha256h2q_u32` / `vsha256su0q_u32` / `vsha256su1q_u32`
- 2-way interleaved SHA256: processes two passwords simultaneously to hide ARM SHA2 instruction latency (~3 cycle latency, 1 cycle throughput)
- HMAC midstate caching (same principle as GPU)
- Shared work queue with GPU for load balancing

### Architecture

```
                    +------------------+
                    | Password         |
                    | Generator        |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Shared Work Queue |
                    | (mutex-protected) |
                    +--+----------+----+
                       |          |
              +--------v--+  +---v-----------+
              | GPU Worker |  | CPU Workers   |
              | (Metal)    |  | (16 threads)  |
              |            |  |               |
              | 32k batch  |  | 64 pwd batch  |
              | double-buf |  | 2-way ARM SHA2|
              +-----+------+  +-------+-------+
                    |                  |
                    +--------+---------+
                             |
                    +--------v---------+
                    | AES Key Unwrap   |
                    | (RFC 3394)       |
                    +------------------+
```

## Creating test images

```bash
python create_simple_encrypted_image.py test.dmg mypassword 50
```

Creates a 50MB encrypted APFS disk image with the given password.

## Files

- `apfs_brute_force_pro.mm` — Main brute force tool (C++/Objective-C++, ~1800 lines)
- `pbkdf2.metal` — Metal GPU compute shader for PBKDF2-SHA256
- `Makefile` — Build system
- `create_simple_encrypted_image.py` — Test image creation script
