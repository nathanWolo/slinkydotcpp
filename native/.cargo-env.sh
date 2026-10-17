# source me: cargo writes its build output to RAM, not the worn SSD
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/dev/shm/native-az-target
