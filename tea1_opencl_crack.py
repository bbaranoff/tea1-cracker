import argparse
import pyopencl as cl
import numpy as np
import time
import sys
import os
import warnings
# This suppresses the specific CompilerWarning from pyopencl
warnings.filterwarnings("ignore", category=UserWarning, module="pyopencl")
# Affichage des logs de compilation pour débugger le hardware si besoin
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

KERNEL_CODE = """
__constant ushort LUT_A[8] = { 0xDA86, 0x85E9, 0x29B5, 0x2BC6, 0x8C6B, 0x974C, 0xC671, 0x93E2 };
__constant ushort LUT_B[8] = { 0x85D6, 0x791A, 0xE985, 0xC671, 0x2B9C, 0xEC92, 0xC62B, 0x9C47 };
__constant uchar SBOX[256] = {
    0x9B, 0xF8, 0x3B, 0x72, 0x75, 0x62, 0x88, 0x22, 0xFF, 0xA6, 0x10, 0x4D, 0xA9, 0x97, 0xC3, 0x7B,
    0x9F, 0x78, 0xF3, 0xB6, 0xA0, 0xCC, 0x17, 0xAB, 0x4A, 0x41, 0x8D, 0x89, 0x25, 0x87, 0xD3, 0xE3,
    0xCE, 0x47, 0x35, 0x2C, 0x6D, 0xFC, 0xE7, 0x6A, 0xB8, 0xB7, 0xFA, 0x8B, 0xCD, 0x74, 0xEE, 0x11,
    0x23, 0xDE, 0x39, 0x6C, 0x1E, 0x8E, 0xED, 0x30, 0x73, 0xBE, 0xBB, 0x91, 0xCA, 0x69, 0x60, 0x49,
    0x5F, 0xB9, 0xC0, 0x06, 0x34, 0x2A, 0x63, 0x4B, 0x90, 0x28, 0xAC, 0x50, 0xE4, 0x6F, 0x36, 0xB0,
    0xA4, 0xD2, 0xD4, 0x96, 0xD5, 0xC9, 0x66, 0x45, 0xC5, 0x55, 0xDD, 0xB2, 0xA1, 0xA8, 0xBF, 0x37,
    0x32, 0x2B, 0x3E, 0xB5, 0x5C, 0x54, 0x67, 0x92, 0x56, 0x4C, 0x20, 0x6B, 0x42, 0x9D, 0xA7, 0x58,
    0x0E, 0x52, 0x68, 0x95, 0x09, 0x7F, 0x59, 0x9C, 0x65, 0xB1, 0x64, 0x5E, 0x4F, 0xBA, 0x81, 0x1C,
    0xC2, 0x0C, 0x02, 0xB4, 0x31, 0x5B, 0xFD, 0x1D, 0x0A, 0xC8, 0x19, 0x8F, 0x83, 0x8A, 0xCF, 0x33,
    0x9E, 0x3A, 0x80, 0xF2, 0xF9, 0x76, 0x26, 0x44, 0xF1, 0xE2, 0xC4, 0xF5, 0xD6, 0x51, 0x46, 0x07,
    0x14, 0x61, 0xF4, 0xC1, 0x24, 0x7A, 0x94, 0x27, 0x00, 0xFB, 0x04, 0xDF, 0x1F, 0x93, 0x71, 0x53,
    0xEA, 0xD8, 0xBD, 0x3D, 0xD0, 0x79, 0xE6, 0x7E, 0x4E, 0x9A, 0xD7, 0x98, 0x1B, 0x05, 0xAE, 0x03,
    0xC7, 0xBC, 0x86, 0xDB, 0x84, 0xE8, 0xD1, 0xF7, 0x16, 0x21, 0x6E, 0xE5, 0xCB, 0xA3, 0x1A, 0xEC,
    0xA2, 0x7D, 0x18, 0x85, 0x48, 0xDA, 0xAA, 0xF0, 0x08, 0xC6, 0x40, 0xAD, 0x57, 0x0D, 0x29, 0x82,
    0x7C, 0xE9, 0x8C, 0xFE, 0xDC, 0x0F, 0x2D, 0x3C, 0x2E, 0xF6, 0x15, 0x2F, 0xAF, 0xE1, 0xEB, 0x3F,
    0x99, 0x43, 0x13, 0x0B, 0xE0, 0xA5, 0x12, 0x77, 0x5D, 0xB3, 0x38, 0xD9, 0xEF, 0x5A, 0x01, 0x70
};

inline uchar tea1_word_to_byte(ushort wSt, __constant ushort* lut) {
    uchar b0 = wSt & 0xFF, b1 = (wSt >> 8) & 0xFF, out = 0;
    for (int i = 0; i < 8; i++) {
        if (lut[i] & (1 << (((b0 >> 7) & 1) | ((b0 << 1) & 2) | ((b1 << 1) & 12)))) out |= (1 << i);
        b0 = (b0 >> 1) | (b0 << 7); b1 = (b1 >> 1) | (b1 << 7);
    }
    return out;
}

inline uchar tea1_reorder(uchar b) {
    return ((b << 6) & 0x40) | ((b << 1) & 0x20) | ((b << 2) & 0x08) | 
           ((b >> 3) & 0x14) | ((b >> 2) & 0x01) | ((b >> 5) & 0x02) | ((b << 4) & 0x80);
}

__kernel void crack_tea1(uint start_counter, ulong match_target_64, ulong qwIv, __global uint* found_key) {
    uint counter = start_counter + get_global_id(0);
    uint dwKeyReg = counter; 
    ulong iv = qwIv;
    
    uint dwNumSkipRounds = 54;
    ulong current_ks_64 = 0;

    // Calcul des 8 premiers octets (64 bits) pour éliminer les collisions
    for (int i = 0; i < 8; i++) { 
        for (int j = 0; j < dwNumSkipRounds; j++) {
            uchar sboxOut = SBOX[((dwKeyReg >> 24) ^ dwKeyReg) & 0xFF];
            dwKeyReg = (dwKeyReg << 8) | sboxOut;
            uchar d12 = tea1_word_to_byte((iv >> 8) & 0xFFFF, LUT_A);
            uchar d56 = tea1_word_to_byte((iv >> 40) & 0xFFFF, LUT_B);
            uchar r4  = tea1_reorder((iv >> 32) & 0xFF);
            iv = ((iv << 8) ^ ((ulong)d12 << 32)) | (uchar)(d56 ^ (iv >> 56) ^ r4 ^ sboxOut);
        }
        current_ks_64 = (current_ks_64 << 8) | (uchar)(iv >> 56);
        dwNumSkipRounds = 19;
    }

    if (current_ks_64 == match_target_64) {
        atomic_xchg(found_key, counter);
    }
}
"""

def build_iv(f):
    iv = ((f['tn']-1)|(f['fn']<<2)|(f['mn']<<7)|((f['hn']&0x7FFF)<<13)|(f['dir']<<28))
    xorred = (((iv ^ 0x96724FA1) << 8) & 0xFFFFFFFF) | ((iv ^ 0x96724FA1) >> 24)
    qw = (iv << 32) | xorred
    return ((qw >> 8) & 0x00FFFFFFFFFFFFFF) | ((qw & 0xFF) << 56)

def crack(args):
    # Cible sur 64 bits (16 caractères hex) pour éviter les collisions 32 bits
    if len(args.ks) < 16:
        print("[-] Erreur : Fournissez au moins 16 caractères hex du keystream.")
        return
    
    match_hex = args.ks[:16]
    match_target_64 = int(match_hex, 16)

    # Initialisation OpenCL
    platforms = cl.get_platforms()
    dev = platforms[0].get_devices(cl.device_type.GPU)[0]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNEL_CODE).build(options=["-cl-mad-enable"])    
    # On réutilise l'objet kernel pour éviter le Warning "RepeatedKernelRetrieval"
    knl = cl.Kernel(prg, "crack_tea1")

    iv = build_iv({'tn': args.tn, 'fn': args.fn, 'mn': args.mn, 'hn': args.hn, 'dir': args.direction})
    found_key_np = np.array([0], dtype=np.uint32)
    found_key_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=found_key_np)
    
    global_size = 1 << 20 # Batch de ~1 million (plus fluide pour le laptop)
    
    print(f"[*] Hardware: {dev.name}")
    print(f"[*] Target KS (64-bit): {match_hex}")
    start_time = time.time()

    for start in range(0, 0xFFFFFFFF, global_size):
        try:
            knl.set_arg(0, np.uint32(start))
            knl.set_arg(1, np.uint64(match_target_64))
            knl.set_arg(2, np.uint64(iv))
            knl.set_arg(3, found_key_buf)
            
            cl.enqueue_nd_range_kernel(queue, knl, (global_size,), None)
            
            cl.enqueue_copy(queue, found_key_np, found_key_buf)
            if found_key_np[0] != 0:
                print(f"\n[+] SUCCESS! KEY FOUND: {found_key_np[0]:08X}")
                print(f"[+] Temps total: {time.time() - start_time:.2f}s")
                return
            
            if (start % (global_size * 10)) == 0:
                prog = (start / 0xFFFFFFFF) * 100
                sys.stdout.write(f"\rProgress: {prog:.2f}% | Keys/s: {start/(time.time()-start_time+0.001):.0f}")
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n[!] Abandon.")
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tn", type=int); parser.add_argument("hn", type=int)
    parser.add_argument("mn", type=int); parser.add_argument("fn", type=int)
    parser.add_argument("sn", type=int); parser.add_argument("direction", type=int)
    parser.add_argument("ks", type=str)
    crack(parser.parse_args())
