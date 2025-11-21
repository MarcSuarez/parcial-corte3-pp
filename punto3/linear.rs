use std::time::Instant;
use std::fs::File;
use std::io::{self, BufRead};

fn get_rss_kb() -> Option<u64> {
    if let Ok(file) = File::open("/proc/self/status") {
        let reader = io::BufReader::new(file);
        for line in reader.lines().flatten() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<u64>() {
                        return Some(kb);
                    }
                }
            }
        }
    }
    None
}

fn main() {
    let x = vec![1.0,2.0,3.0,4.0,5.0];
    let y = vec![2.0,4.0,6.0,8.0,10.0];
    let mut w = 0.0_f64;
    let mut b = 0.0_f64;
    let learning_rate = 0.01_f64;
    let epochs = 1000_usize;
    let m = x.len() as f64;

    let t0 = Instant::now();
    for epoch in 0..epochs {
        let mut dw = 0.0_f64;
        let mut db = 0.0_f64;
        let mut mse = 0.0_f64;
        for i in 0..x.len() {
            let y_pred = w * x[i] + b;
            let error = y_pred - y[i];
            dw += error * x[i];
            db += error;
            mse += error * error;
        }
        dw = (2.0 / m) * dw;
        db = (2.0 / m) * db;
        w -= learning_rate * dw;
        b -= learning_rate * db;
        if (epoch + 1) % 200 == 0 {
            println!("Epoch {}, MSE: {:.6}, w: {:.6}, b: {:.6}", epoch+1, mse / m, w, b);
        }
    }
    let elapsed = t0.elapsed();
    let rss_kb = get_rss_kb().unwrap_or(0);
    println!("\nRust result:");
    println!("w ≈ {:.6}, b ≈ {:.6}", w, b);
    println!("time_s: {:.6}s", elapsed.as_secs_f64());
    println!("rss_kb: {} KB", rss_kb);
}

/**
Rust result:
time_s: 0.002050s
rss_kb: 2228 KB
**/