#pragma once

//
// CODE QUALITY ISSUE:
// These should be moved into Constants.cs so we can change them without needing to recompile the code.
// Justification: some of these needed at compile time. -MR
//
#if _WIN64 
const int MAX_UNDO_OPERATIONS = 20;
const int FILE_SYSTEM_TILE_CACHE_SIZE = 1024;
const int MAX_DEVICE_TILE_CACHE_SIZE = 512;
const int SPLIT_ADJUST_BUFFER_TILE_HALO = 2;
#else
const int MAX_UNDO_OPERATIONS = 5;
const int FILE_SYSTEM_TILE_CACHE_SIZE = 512;
const int MAX_DEVICE_TILE_CACHE_SIZE = 256;
const int SPLIT_ADJUST_BUFFER_TILE_HALO = 1;
#endif
const int TILE_PIXELS = 512;
const int EXTRA_SEGMENTS_PER_SESSION = 1024;