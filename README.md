# WebGPU n-body sim

Currently naive n^2 implementation. Testing some simple compute shader optimizations like using work group shared memory on different hardware. (Spoiler alert, doesn't add anything on Apple GPU's).

50 000 particles at real-ish time on M1 Pro.

My dream is to implement Barnes-Hut or some other acceleration structure to hit something like 1 million particles, but that may or may not ever happen.
