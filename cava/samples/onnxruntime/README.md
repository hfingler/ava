Install ONNXruntime 1.2.0 from [yuhc/ava-onnxruntime](https://github.com/yuhc/ava-onnxruntime).

Steps to run onnx_opt with a specified dump directory:

0. Set AVA_GUEST_DUMP_DIR and AVA_WORKER_DUMP_DIR on the guestlib application side when running. Two separate envvars are necessary in svless mode due to the guestlib application residing in the VM.

1. Generate / compile the onnx_opt spec with -DAVA_PRELOAD_CUBIN in the ava_cxx flags at the top of onnx_opt.cpp. This is just to make sure onnx_dump doesn't try to call ava_load_cubin_worker in the cmd handler. There's probably something better like a flag for each spec or something.

You should see the dump directory name output from the internal_api_handler when the command is received on the worker's side.


Current issue:
There may be a race condition with API commands being remoted before the worker's loaded the fatbins. I'm a little suspicious of that reason though as I was doing this fine before with no problems, so it's probably worth investigating further.

If it is a race condition, things I've tried so far:
- switch `__handle_command_onnx_opt_init()` and `nw_init_guestlib(ONNX_OPT_API)` in the guestlib's guest context (in the generated code). Don't do this, bad things happen.
- switch the order in guestlib/init.cpp that the DUMP_DIR and INIT_API commands are sent. I don't think this works, need to look into it more.
