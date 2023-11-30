#![allow(dead_code, unused)]
use plotters::prelude::*;
use std::cmp::*;

fn main() {
    q1();
    q2();
}

fn q1() {
    let _10x10 = vec![
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
        vec![0., 2., 3., 4., 5., 6., 7., 8., 9.],
        vec![0., 0., 3., 4., 5., 6., 7., 8., 9.],
        vec![0., 0., 0., 4., 5., 6., 7., 8., 9.],
        vec![0., 0., 0., 0., 5., 6., 7., 8., 9.],
        vec![0., 0., 0., 0., 0., 6., 7., 8., 9.],
        vec![0., 0., 0., 0., 0., 0., 7., 8., 9.],
        vec![0., 0., 0., 0., 0., 0., 0., 8., 9.],
        vec![0., 0., 0., 0., 0., 0., 0., 0., 9.],
    ];
    let det10x10 = det(&_10x10);
    let prod_diag10x10 = prod_diag(&_10x10);
    println!("det of upper 10x10:{:?}", det10x10);
    println!("product of diagonal 10x10:{:?}", prod_diag10x10);

    let swapped = swap_cols(&_10x10, 3, 4);
    let detswapped = det(&swapped);
    println!("det of 10x10 with swapped cols:{:?}", detswapped);

    let _10x10_dup = vec![
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
        vec![0., 0., 3., 4., 5., 6., 7., 8., 9.],
        vec![0., 0., 0., 4., 5., 6., 7., 8., 9.],
        vec![0., 0., 0., 0., 5., 6., 7., 8., 9.],
        vec![0., 0., 0., 0., 0., 6., 7., 8., 9.],
        vec![0., 0., 0., 0., 0., 0., 7., 8., 9.],
        vec![0., 0., 0., 0., 0., 0., 0., 8., 9.],
        vec![0., 0., 0., 0., 0., 0., 0., 0., 9.],
    ];
    let det10x10_dup = det(&_10x10_dup);
    println!("det of 10x10 with duplicate rows:{:?}", det10x10_dup);

    let detmatmul = det(&mat_mul(&_10x10, &_10x10_dup));
    println!(
        "\ndet(A):{:?}\ndet(B):{:?}\ndet(A)det(B):{:?}\ndet(AB):{:?}",
        det10x10,
        det10x10_dup,
        det10x10 * det10x10_dup,
        detmatmul
    );

    let dettrans = det(&transpose(&swapped));
    println!("\ndet of 10x10:{:?}", detswapped);
    println!("det of 10x10 transposed:{:?}", dettrans);
}

fn q2() {
    let _ = factorial_plot();
}

fn print_mat(a: &Vec<Vec<f32>>) {
    for i in a {
        println!("{:?}", i)
    }
}

fn det(a: &Vec<Vec<f32>>) -> f32 {
    let n = a.len();
    match n {
        0 => 0.0,
        1 => a[0][0],
        2 => a[0][0] * a[1][1] - a[0][1] * a[1][0],
        _ => {
            let mut d = 0.0;
            for (ind, head) in (&a[0]).iter().enumerate() {
                let mut curr = (-1 as f32).powf(ind as f32);
                curr *= head;
                curr *= det(&a
                    .iter()
                    .enumerate()
                    .filter(|(n1, _)| *n1 != 0) // don't get the first row
                    .map(|(_, i)| {
                        i.iter()
                            .enumerate()
                            .filter(|(n2, _)| *n2 != ind) // don't get the current column
                            .map(|(_, j)| *j)
                            .collect()
                    })
                    .collect());
                d += curr
            }
            d
        }
    }
}

fn prod_diag(a: &Vec<Vec<f32>>) -> f32 {
    let mut ans = 1.0;
    for (n1, i) in a.iter().enumerate() {
        for (n2, j) in i.iter().enumerate() {
            if n1 == n2 {
                ans *= j
            }
        }
    }
    ans
}

fn swap_cols(a: &Vec<Vec<f32>>, i: usize, j: usize) -> Vec<Vec<f32>> {
    let mut b = vec![];
    for (n1, s) in (&a).iter().enumerate() {
        let mut c = vec![];
        for (n2, t) in s.iter().enumerate() {
            if n2 == i {
                c.push((&a)[n1][j]);
            } else if n2 == j {
                c.push((&a)[n1][i]);
            } else {
                c.push(*t);
            }
        }
        b.push(c)
    }
    b
}

fn mat_mul(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    assert_eq!(a[0].len(), b.len());
    let mut c = vec![];
    for (_, i) in (&a).iter().enumerate() {
        let mut d = vec![];
        for f in 0..a[0].len() {
            let mut e = 0.;
            for (n2, j) in i.iter().enumerate() {
                e += j * b[n2][f];
            }
            d.push(e);
        }
        c.push(d);
    }
    c
}

fn transpose(a: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut b = vec![];
    for (n1, i) in (&a).iter().enumerate() {
        let mut c = vec![];
        for (n2, _) in i.iter().enumerate() {
            c.push(a[n2][n1]);
        }
        b.push(c);
    }
    b
}

fn mat_scale(a: &Vec<Vec<f32>>, s: f32) -> Vec<Vec<f32>> {
    a.iter()
        .map(|i| i.iter().map(|j| j * s).collect())
        .collect()
}

fn mat_add(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    assert_eq!(a.len(), b.len());
    let mut c = vec![];
    for (n1, i) in (&a).iter().enumerate() {
        let mut d = vec![];
        for (n2, j) in i.iter().enumerate() {
            d.push(j + b[n1][n2]);
        }
        c.push(d);
    }
    c
}

fn factorial(n: i32) -> i32 {
    match n {
        0 => 1,
        _ => n * factorial(n - 1),
    }
}

fn factorial_plot() -> Result<(), Box<dyn std::error::Error>> {
    let xrange = 0..7;
    let factrange = xrange.clone().map(|x| factorial(x as i32));
    let exprange = xrange.clone().map(|x| (x as f32).exp());
    let root = BitMapBackend::new("factorial_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            xrange.clone(),
            0..max(factrange.last().unwrap(), exprange.last().unwrap() as i32),
        )?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            xrange.clone().map(|x| (x as i32, factorial(x) as i32)),
            &RED,
        ))?
        .label("y=x!")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart
        .draw_series(LineSeries::new(
            xrange.clone().map(|x| (x as i32, (x as f32).exp() as i32)),
            &BLUE,
        ))?
        .label("y=exp(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    root.present()?;

    Ok(())
}
