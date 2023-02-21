use image;
use std::format;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;
use ocl::ProQue;


fn get_color(n_iter: u32, max_iter: u32) -> image::Rgb<u8>{

    if n_iter > max_iter{
        return image::Rgb([255, 255, 255]);
    }

    if n_iter == max_iter{
        let index = n_iter as u8;
        return image::Rgb([index, index, index]);
    }

    let index = (((n_iter as f32)/(max_iter as f32)) * 255.0).round() as u8;
    return image::Rgb([index, index, index]);
}

fn create_set(width: u32, height: u32) -> image::RgbImage{

    let mut img = image::RgbImage::new(width, height);

    for i in 0..width{
        let x0 = ((i as f32)/(width as f32))*3.5 - 2.5;
        for j in 0..height{
            let y0 = ((j as f32)/(height as f32))*2.0 - 1.0;
            let mut x = 0.0;
            let mut y = 0.0;
            let mut iteration: u32 = 0;
            while (x*x + y*y <= 4.0) && iteration< 1000{
                let xtemp = x*x - y*y + x0;
                y = 2.0 * x * y + y0;
                x = xtemp;
                iteration = iteration + 1;
            }
        let rgb = get_color(iteration, 1000);
        img.put_pixel(i, j, rgb);
        }
    }
    img
}

fn create_set_opencl(w: u32, h: u32) -> image::RgbImage {
    let mut img = image::RgbImage::new(w as u32, h as u32);
    let src = r#"
        __kernel void mandelbrot(__global uint* buffer, uint width, uint height, uint max_iterations) {
            int c = get_global_id(0);
            int r = get_global_id(1);
            float x0 = ((float)c / width) * 3.5 - 2.5;
            float y0 = ((float)r / height) * 2.0 - 1.0;
            float x = 0.0;
            float y = 0.0;
            float x2 = 0.0;
            float y2 = 0.0;
            uint iteration = 0;
            while (((x2 + y2) <= 4.0) && (iteration < max_iterations)) {
                y = (x + x) * y + y0;
                x = x2 - y2 + x0;
                x2 = x * x;
                y2 = y * y;
                iteration = iteration + 1;
            }
            buffer[width * r + c] = iteration;
        }
    "#;
    let pro_que = ProQue::builder().src(src).dims((w, h)).build().unwrap();
    let buffer = pro_que.create_buffer::<u32>().unwrap();
    let kernel = pro_que
        .kernel_builder("mandelbrot")
        .arg(&buffer)
        .arg(w)
        .arg(h)
        .arg(1000 as u32)
        .build()
        .unwrap();
    unsafe { kernel.enq().unwrap() };
    let mut vec = vec![0u32; buffer.len()];
    buffer.read(&mut vec).enq().unwrap();
    for (idx, iteration) in vec.iter().enumerate() {
        let rgb = get_color(*iteration, 1000);
        let x = idx as u32 % w;
        let y = idx as u32 / w;
        img.put_pixel(x, y, rgb);
    }
    img
}

fn main() {
    let w = 10000;
    let h = 6000;
    let now = Instant::now();
    let img = create_set_opencl(w, h);
    let elapsed = now.elapsed().as_millis() as f32/1000.0;
    println!("Elapsed: {}s", elapsed);
    let fname = format!("Mandelbrot_set_1.png");
    img.save_with_format(fname, image::ImageFormat::Png).unwrap();
}
