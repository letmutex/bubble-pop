const DX: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
const DY: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];
const MAX_PATH_POINTS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntPoint {
    pub x: i32,
    pub y: i32,
}

#[inline(always)]
fn direction_index(dx: i32, dy: i32) -> usize {
    for (i, (&kdx, &kdy)) in DX.iter().zip(DY.iter()).enumerate() {
        if kdx == dx && kdy == dy {
            return i;
        }
    }
    6
}

/// Traces the external contour of a connected component using Moore boundary tracing.
pub fn trace_contour(
    component: &[u8],
    queue: &[usize],
    count: usize,
    width: usize,
    height: usize,
) -> Vec<IntPoint> {
    if count == 0 {
        return Vec::new();
    }

    let mut start = queue[0];
    for &candidate in &queue[1..count] {
        let (cy, cx) = (candidate / width, candidate % width);
        let (sy, sx) = (start / width, start % width);
        if cy < sy || (cy == sy && cx < sx) {
            start = candidate;
        }
    }

    let mut x = (start % width) as i32;
    let mut y = (start / width) as i32;
    let start_x = x;
    let start_y = y;
    let mut back_x = x - 1;
    let mut back_y = y;
    let start_back_x = back_x;
    let start_back_y = back_y;

    let mut contour = Vec::with_capacity((count * 2).min(MAX_PATH_POINTS * 4));
    let max_steps = (count * 8).max(8);

    for _ in 0..max_steps {
        contour.push(IntPoint { x, y });
        let back_dir = direction_index(back_x - x, back_y - y);
        let mut found_dir = None;

        for step in 1..=8 {
            let dir = (back_dir + step) & 7;
            let nx = x + DX[dir];
            let ny = y + DY[dir];
            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32
                && component[ny as usize * width + nx as usize] != 0
            {
                found_dir = Some(dir);
                break;
            }
        }

        let Some(dir) = found_dir else {
            break;
        };

        let preceding = (dir + 7) & 7;
        back_x = x + DX[preceding];
        back_y = y + DY[preceding];
        x += DX[dir];
        y += DY[dir];

        if x == start_x && y == start_y && back_x == start_back_x && back_y == start_back_y {
            break;
        }
    }

    contour
}

#[inline(always)]
fn perpendicular_distance_sq(p: IntPoint, a: IntPoint, b: IntPoint) -> f32 {
    let dx = (b.x - a.x) as f32;
    let dy = (b.y - a.y) as f32;
    let d2 = dx * dx + dy * dy;
    if d2 < 1e-6 {
        let px = (p.x - a.x) as f32;
        let py = (p.y - a.y) as f32;
        return px * px + py * py;
    }
    let cross = (p.x - a.x) as f32 * dy - (p.y - a.y) as f32 * dx;
    (cross * cross) / d2
}

fn simplify_rdp_recursive(
    points: &[IntPoint],
    start: usize,
    end: usize,
    epsilon_sq: f32,
    keep: &mut [bool],
) {
    if end <= start + 1 {
        return;
    }
    let mut max_dist_sq = 0.0f32;
    let mut max_idx = start;

    for i in (start + 1)..end {
        let dist_sq = perpendicular_distance_sq(points[i], points[start], points[end]);
        if dist_sq > max_dist_sq {
            max_dist_sq = dist_sq;
            max_idx = i;
        }
    }

    if max_dist_sq > epsilon_sq {
        keep[max_idx] = true;
        simplify_rdp_recursive(points, start, max_idx, epsilon_sq, keep);
        simplify_rdp_recursive(points, max_idx, end, epsilon_sq, keep);
    }
}

/// Simplifies a polygon contour using Ramer-Douglas-Peucker (RDP) algorithm.
pub fn simplify_polygon_rdp(contour: &[IntPoint], epsilon_px: f32) -> Vec<IntPoint> {
    if contour.len() <= 4 {
        return contour.to_vec();
    }
    let epsilon_sq = epsilon_px * epsilon_px;
    let mut keep = vec![false; contour.len()];
    keep[0] = true;
    *keep.last_mut().unwrap() = true;

    let mut max_d2 = 0.0f32;
    let mut split_idx = contour.len() / 2;
    for (i, pt) in contour.iter().enumerate().skip(1) {
        let dx = (pt.x - contour[0].x) as f32;
        let dy = (pt.y - contour[0].y) as f32;
        let d2 = dx * dx + dy * dy;
        if d2 > max_d2 {
            max_d2 = d2;
            split_idx = i;
        }
    }
    keep[split_idx] = true;

    simplify_rdp_recursive(contour, 0, split_idx, epsilon_sq, &mut keep);
    simplify_rdp_recursive(contour, split_idx, contour.len() - 1, epsilon_sq, &mut keep);

    let result: Vec<IntPoint> = contour
        .iter()
        .zip(keep.iter())
        .filter_map(|(&pt, &k)| if k { Some(pt) } else { None })
        .collect();

    if result.len() < 3 {
        return contour.to_vec();
    }

    if result.len() > MAX_PATH_POINTS {
        let step = (result.len() as f64 / MAX_PATH_POINTS as f64).ceil() as usize;
        let mut capped = Vec::with_capacity(MAX_PATH_POINTS);
        for i in (0..result.len()).step_by(step) {
            capped.push(result[i]);
        }
        return capped;
    }

    result
}
