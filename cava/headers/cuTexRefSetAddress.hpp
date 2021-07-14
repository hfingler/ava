

CUresult CUDAAPI cuTexRefSetAddress(unsigned int *ByteOffset, CUtexref hTexRef, 
    CUdeviceptr dptr, unsigned int bytes);

//add extern "C"

CUresult CUDAAPI cuTexRefSetAddress_v2(unsigned int *ByteOffset, CUtexref hTexRef, 
    CUdeviceptr dptr, unsigned int bytes);
