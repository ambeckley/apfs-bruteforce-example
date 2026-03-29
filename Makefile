# Makefile for APFS Brute Forcer
# Uses Metal framework for GPU acceleration and ARM SHA2 intrinsics

CXX = clang++
CXXFLAGS = -Wall -Wextra -O3 -mcpu=native -flto -std=c++17

BRUTE_FORCE = apfs_brute_force
BRUTE_FORCE_SRC = apfs_brute_force.mm

.PHONY: all clean brute-force

all: $(BRUTE_FORCE)

brute-force: $(BRUTE_FORCE)

$(BRUTE_FORCE): $(BRUTE_FORCE_SRC) pbkdf2.metal
	$(CXX) $(CXXFLAGS) -o $(BRUTE_FORCE) $(BRUTE_FORCE_SRC) \
		-framework Metal -framework Foundation -framework MetalKit \
		-framework Security -pthread -I.

clean:
	rm -f $(BRUTE_FORCE)
