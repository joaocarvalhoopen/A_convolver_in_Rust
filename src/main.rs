// A convolver in Rust
// A small port of a Overlap and Add convolution implementation with FFTs.
// Author of the port: João Carvalho
// Date: 2022.08.13
// Description
// This is a small port of one implementation of a convolver from
// The Wolf Sound from Python to Rust. It has some specific differences
// other than the language because the FFT lib used, doesn't deliver
// scaled result values, but with the work around, the results are the same.
//
// The original python code is in 
// **Fast Convolution FFT-based Overlap-Add, Overlap-Save, Partitioned**
// https://thewolfsound.com/fast-convolution-fft-based-overlap-add-overlap-save-partitioned/
//
// License: Mit Open Source License

use rustfft::FftPlanner;

fn main() {
    println!("Convolver in Rust.");
}

///////////////
// Convolution

fn next_power_of_2(n: usize) -> usize {
    1 << (f32::log2(n as f32 - 1.0) as usize + 1)
}

/// Append zeros to x to make the new_length, x is copied.
fn pad_zeros_to(x: &Vec<f32>, new_length: usize) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.0; new_length];
    output[0..x.len()].copy_from_slice(&x[..]);
    output
}

/// Makes the FFT convolution, y = x*h for block_len.
///   x: Vector A
///   h: Vector B
///   fft_block_len: The k size of the power of two of the block size,
///              if not filled automatically goes to the next power of 2.
///              FFT's of powers of 2 are faster to calculate.
fn fft_convolution(planner: & mut FftPlanner<f32>, x: &Vec<f32>, h: &Vec<f32>, fft_block_len: Option<usize>) -> Vec<f32> {
    let x_len = x.len();
    let h_len = h.len();
    // Output length
    let y_len = x_len + h_len - 1;

    // Reference for IFFT that don't return IFFT(FFT(A)) != A like Python does:
    // https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.4/imlug/imlug_langref_sect197.htm
    
    // Find p so that 2**(p-1) < wn <= 2**p 
    let p = f32::ceil( f32::log(y_len as f32, std::f32::consts::E) / f32::log(2.0, std::f32::consts::E) );
    let nice = f32::powf(2.0, p);

    // Make K smallest optimal
    let fft_block_len = match fft_block_len {
            None => next_power_of_2(y_len),
            Some(val) => val, 
        };
    
    let x_padded = pad_zeros_to(x, fft_block_len);
    let h_padded = pad_zeros_to(h, fft_block_len);
    
    // Perform a forward FFT of size k block_len .
    use rustfft::num_complex::Complex;

    // Creates the FFT planner that will be used by the 2 FFT's because they are of the same size.
    //let mut planner = FftPlanner::<f32>::new();  // aqui
    
    let fft = planner.plan_fft_forward(fft_block_len);

    // Calculate the fast Fourier transforms 
    // of the time-domain signals
    // X = np.fft.fft(pad_zeros_to(x, K))
    // H = np.fft.fft(pad_zeros_to(h, K))

    // Calculate the FFT for x_padded complex array.
    let mut buffer_x = vec![Complex{ re: 0.0_f32, im: 0.0_f32 }; fft_block_len];
    for i in 0..x_padded.len() {
        buffer_x[i].re = x_padded[i];
    }
    fft.process(& mut buffer_x[..]);

    // Calculate the FFT for x_padded complex array.
    let mut buffer_h = vec![Complex{ re: 0.0_f32, im: 0.0_f32 }; fft_block_len];
    for i in 0..h_padded.len() {
        buffer_h[i].re = h_padded[i];
    }
    fft.process(& mut buffer_h[..]);

    // Perform circular convolution in the frequency domain
    // Y = FFT(x_padded) * FFT(y_padded)
    let mut fft_buffer_y = buffer_x.iter()
                                    .zip(&buffer_h)
                                    .map(|(c_x, c_h)| c_x * c_h )
                                    .collect::<Vec<Complex<f32>>>();
  
    // println!(" {:?} ", fft_buffer_y);

    // Go back to time domain

    // Creates the FFT planner that will be used by the IFFT's.
    // let mut planner_i = FftPlanner::<f32>::new();
    let planner_i = planner;
    
    
    // let ifft = planner.plan_fft_inverse(block_len);
    let ifft = planner_i.plan_fft_inverse(fft_buffer_y.len());

    // Calculate the IFFT.
    ifft.process(& mut fft_buffer_y[..]);
    
    let ifft_buffer_y = fft_buffer_y;
    
    // println!(" {:?} ", ifft_buffer_y);

    // Check if it is the real or the norm, in the example it was the real part, but I think it's the norm.
    // y = np.real(np.fft.ifft(Y));
    // let mut res_y = ifft_buffer_y.iter().map(|c| c.norm()).collect::<Vec<f32>>();
    let mut res_y = ifft_buffer_y.iter().map(|c| {
        // NOTE IMPORTANT: Without the division by "nice" the result of IFFT(FFT( A )) != A
        //                  It's different from Pythons NumPy. 
        let res = c / nice;
        res.re
        // c.norm()
        }).collect::<Vec<f32>>();

    // Trim the signal to the expected length
    res_y.resize(y_len, 0.0);
    res_y
}


///  Overlap-Add convolution of x and h with block length B block_len_b
///   x: Vector A
///   h: Vector B
///   block_len_b: Convolution block size, not the fft_block_len.
///   fft_block_len: The k size of the power of two of the block size,
///              if not filled automatically goes to the next power of 2.
///              FFT's of powers of 2 are faster to calculate.
fn overlap_add_convolution(x: & Vec<f32>, h: & Vec<f32>, block_len_b: usize, k:Option<usize>) -> Vec<f32> {

    let m_len = x.len();
    let n_len = h.len();

    // Calculate the number of input blocks
    let num_input_blocks = f32::ceil(m_len as f32 / block_len_b as f32) as usize;

    // Pad x to an integer multiple of num_blocks
    let xp = pad_zeros_to(x, num_input_blocks * block_len_b);

    // Your turn ...
    let output_size = num_input_blocks * block_len_b + n_len - 1;
    let mut res_y = vec![0.0_f32; output_size];

    // Creates the FFT planner that will be used by the 2 FFT's because they are of the same size.
    let mut planner = FftPlanner::<f32>::new();

    // Convolve all blocks
    for n in 0..num_input_blocks {
        // Extract the n-th input block
        let sub = &xp[(n * block_len_b)..((n + 1) * block_len_b)]; 
        let sub_len = xp.len();
        let mut xb: Vec<f32> = Vec::with_capacity(sub_len);      
        xb.extend(sub.iter());

        // Fast convolution
        let u = fft_convolution(& mut planner, &xb, &h, k);

        // Overlap-Add the partial convolution result
        // res_y[ (n * block_len_b)..(n * block_len_b + len(u)) ] += u;

        let range_a = (n * block_len_b)..(n * block_len_b + u.len());
        let range_b = 0..u.len();

        for (res_y_i, u_i) in  range_a.zip(range_b) {
            res_y[res_y_i] += u[u_i];
        }
    }

    // Trim the signal to the expected length
    res_y.resize(m_len + n_len - 1, 0.0);
    res_y
}


////////
// Tests

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(120), 128);
        assert_eq!(next_power_of_2(200), 256);
        assert_eq!(next_power_of_2(500), 512);
        assert_eq!(next_power_of_2(1000), 1024);
        assert_eq!(next_power_of_2(2000), 2048);
        assert_eq!(next_power_of_2(4095), 4096);
    }

    #[test]
    fn test_pad_zeros_to() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0];
        assert_eq!(pad_zeros_to(&x, 4), vec![1.0_f32, 2.0, 3.0, 0.0]);
        assert_eq!(pad_zeros_to(&x, 5), vec![1.0_f32, 2.0, 3.0, 0.0, 0.0]);

    }

    #[test]
    fn test_fft_convolution_a() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let h = vec![1.0_f32];
        let fft_block_len = None;
        // Creates the FFT planner that will be used by the 2 FFT's because they are of the same size.
        let mut planner = FftPlanner::<f32>::new();     
        let res_convolution = fft_convolution(& mut planner, &x, &h, fft_block_len);
        assert_eq!(&x, &res_convolution);
    }

    #[test]
    fn test_fft_convolution_b() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let h = vec![0.0_f32, 1.0, 0.5];
        let fft_block_len = None;
        // Creates the FFT planner that will be used by the 2 FFT's because they are of the same size.
        let mut planner = FftPlanner::<f32>::new();
        let res_convolution = fft_convolution(& mut planner, &x, &h, fft_block_len);
        let correct_res = vec![0.0_f32 , 1.0 , 2.5, 4.0 , 1.5];
        for (res_elem_a, res_correct_b ) in res_convolution.iter().zip(&correct_res) {
            assert!(res_elem_a - res_correct_b < 0.0001);    
        }        
    }

    #[test]
    fn test_overlap_add_convolution() {
        let x = vec![1.0_f32, 2.0, 3.0];
        let h = vec![0.0_f32, 1.0, 0.5];
        let fft_block_len = None;
        let correct_res = vec![0.0_f32 , 1.0 , 2.5, 4.0 , 1.5];
        let block_len_b = 2; 
        let res_convolution = overlap_add_convolution(& x, &h, block_len_b, fft_block_len);
        for (res_elem_a, res_correct_b ) in res_convolution.iter().zip(&correct_res) {
            assert!(res_elem_a - res_correct_b < 0.0001);    
        }
    }

}

