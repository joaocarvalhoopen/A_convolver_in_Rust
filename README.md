# A convolver in Rust
A small port of a Overlap and Add convolution implementation with FFTs.

## Description
This is a small port, of one implementation of a convolver from The Wolf Sound from Python to Rust. It has some specific differences other than the language because the FFT lib used, doesn't deliver scaled result values, but with the work around, the results are the same. <br>
Excellent explanation and the original python code is in <br>
**Fast Convolution FFT-based Overlap-Add, Overlap-Save, Partitioned** <br>
[https://thewolfsound.com/fast-convolution-fft-based-overlap-add-overlap-save-partitioned/](https://thewolfsound.com/fast-convolution-fft-based-overlap-add-overlap-save-partitioned/) <br>


## To run it do
```
cargo test
```


## License
Mit Open Source license.


## Have fun!
Best regards, <br>
João Carvalho