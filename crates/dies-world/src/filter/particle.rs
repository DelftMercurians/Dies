use dies_core::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Configuration for the particle filter
#[derive(Clone, Debug)]
pub struct ParticleFilterConfig {
    /// Number of particles to maintain
    pub num_particles: usize,
    /// Process noise standard deviation for position (meters)
    pub process_noise_pos: f64,
    /// Process noise standard deviation for velocity (meters/second)
    pub process_noise_vel: f64,
    /// Measurement noise standard deviation (meters)
    pub measurement_noise: f64,
    /// Gravity acceleration (meters/second^2)
    pub gravity: f64,
    /// Effective damping/drag coefficient for the ball
    pub drag_coefficient: f64,
}

impl Default for ParticleFilterConfig {
    fn default() -> Self {
        ParticleFilterConfig {
            num_particles: 500,               // Increased for stability
            process_noise_pos: 0.0,           // Reduced: position drift is minimal
            process_noise_vel: 0.0,           // Reduced from 50: was adding massive noise
            measurement_noise: 0.0,          // mm - SSL Vision accuracy
            gravity: 0.0,                     // Disabled: flat surface
            drag_coefficient: 0.0,            // Disabled: no drag needed
        }
    }
}

#[derive(Clone, Debug)]
pub struct Particle {
    pub pos: Vector3,
    pub vel: Vector3,
    pub weight: f64,
}

#[derive(Debug)]
pub struct ParticleFilter {
    particles: Vec<Particle>,
    config: ParticleFilterConfig,
    initialized: bool,
}

impl ParticleFilter {
    /// Create a new particle filter with default configuration
    pub fn new(config: ParticleFilterConfig) -> Self {
        ParticleFilter {
            particles: Vec::with_capacity(config.num_particles),
            config,
            initialized: false,
        }
    }

    /// Initialize particles around a measurement
    pub fn initialize_from_measurement(&mut self, measurement: Vector3, velocity_estimate: Vector3) {
        let mut rng = rand::thread_rng();
        let normal_pos = Normal::new(0.0, self.config.process_noise_pos).unwrap();
        let normal_vel = Normal::new(0.0, self.config.process_noise_vel).unwrap();

        self.particles.clear();
        for _ in 0..self.config.num_particles {
            let pos_noise: Vector3 = Vector3::new(
                normal_pos.sample(&mut rng),
                normal_pos.sample(&mut rng),
                normal_pos.sample(&mut rng),
            );
            let vel_noise: Vector3 = Vector3::new(
                normal_vel.sample(&mut rng),
                normal_vel.sample(&mut rng),
                normal_vel.sample(&mut rng),
            );

            let pos = measurement + pos_noise;
            let vel = velocity_estimate + vel_noise;

            self.particles.push(Particle {
                pos,
                vel,
                weight: 1.0 / self.config.num_particles as f64,
            });
        }
        self.initialized = true;
        log::debug!(
            "ParticleFilter::initialize_from_measurement - measurement: {:?}, first_particle: pos={:?}, vel={:?}",
            measurement,
            self.particles.first().map(|p| p.pos),
            self.particles.first().map(|p| p.vel)
        );
    }

    /// Predict step: apply motion model with gravity and drag to all particles
    pub fn predict(&mut self, delta_time: f64) {
        let mut rng = rand::thread_rng();
        let normal_pos = Normal::new(0.0, self.config.process_noise_pos).unwrap();
        let _normal_vel = Normal::new(0.0, self.config.process_noise_vel).unwrap();

        log::debug!(
            "ParticleFilter::predict - dt: {}, first_particle before: pos={:?}, vel={:?}",
            delta_time,
            self.particles.first().map(|p| p.pos),
            self.particles.first().map(|p| p.vel)
        );

        for particle in &mut self.particles {
            // Apply drag force (exponential decay)
            let drag_factor = (-self.config.drag_coefficient * delta_time).exp();
            particle.vel.x *= drag_factor;
            particle.vel.y *= drag_factor;
            particle.vel.z *= drag_factor;

            // Apply gravity
            particle.vel.z -= self.config.gravity * delta_time;

            // Update position
            particle.pos += particle.vel * delta_time;

            // Add process noise
            let pos_noise: Vector3 = Vector3::new(
                normal_pos.sample(&mut rng),
                normal_pos.sample(&mut rng),
                normal_pos.sample(&mut rng),
            );
            particle.pos += pos_noise;

            // Clamp z position to ground
            if particle.pos.z < 0.0 {
                particle.pos.z = 0.0;
                particle.vel.z = particle.vel.z.abs() * 0.8; // Bounce with some damping
            }
        }

        log::debug!(
            "ParticleFilter::predict - first_particle after: pos={:?}, vel={:?}",
            self.particles.first().map(|p| p.pos),
            self.particles.first().map(|p| p.vel)
        );
    }

    /// Update weights based on measurements (multiple measurements supported)
    pub fn update(&mut self, measurements: &[Vector3]) {
        if measurements.is_empty() {
            return;
        }

        log::debug!(
            "ParticleFilter::update - measurements: count={}, first={:?}",
            measurements.len(),
            measurements.first()
        );

        // Update weights using Gaussian likelihood for each particle
        for particle in &mut self.particles {
            let mut likelihood = 1.0;

            for measurement in measurements {
                let distance = (particle.pos - measurement).norm();
                // Gaussian likelihood: exp(-distance^2 / (2 * sigma^2))
                let sigma = self.config.measurement_noise;
                let weight_contrib =
                    (-distance.powi(2) / (2.0 * sigma.powi(2))).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt());
                likelihood *= weight_contrib;
            }

            particle.weight = likelihood;
        }

        // Normalize weights
        let total_weight: f64 = self.particles.iter().map(|p| p.weight).sum();
        log::debug!(
            "ParticleFilter::update - total_weight before normalization: {}",
            total_weight
        );

        if total_weight > 0.0 {
            for particle in &mut self.particles {
                particle.weight /= total_weight;
            }
        } else {
            // If total weight is zero, use uniform weights
            log::warn!("ParticleFilter::update - total_weight is zero, using uniform weights");
            let uniform_weight = 1.0 / self.particles.len() as f64;
            for particle in &mut self.particles {
                particle.weight = uniform_weight;
            }
        }

        // Resample particles based on degeneracy
        self.resample_if_needed();
    }

    /// Resample particles if effective sample size is too low
    fn resample_if_needed(&mut self) {
        // Calculate effective sample size
        let neff: f64 = self
            .particles
            .iter()
            .map(|p| p.weight * p.weight)
            .sum::<f64>()
            .recip();

        let resample_threshold = self.config.num_particles as f64 * 0.5;
        if neff < resample_threshold {
            self.resample();
        }
    }

    /// Resample particles using cumulative sum roulette wheel selection
    fn resample(&mut self) {
        let mut new_particles = Vec::with_capacity(self.particles.len());
        for _ in 0..self.particles.len() {
            let index = self.resample_index();
            let mut particle = self.particles[index].clone();
            particle.weight = 1.0 / self.particles.len() as f64;
            new_particles.push(particle);
        }
        self.particles = new_particles;
    }

    /// Roulette wheel selection based on particle weights
    fn resample_index(&self) -> usize {
        let rand_val = rand::thread_rng().gen::<f64>();
        let mut cumsum = 0.0;

        for (i, particle) in self.particles.iter().enumerate() {
            cumsum += particle.weight;
            if rand_val < cumsum {
                return i;
            }
        }

        // Fallback: return last particle
        self.particles.len() - 1
    }

    /// Get all particles (read-only)
    pub fn get_particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Get the best estimate (maximum weight particle)
    pub fn get_best_estimate(&self) -> Option<Vector3> {
        self.particles
            .iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
            .map(|p| p.pos)
    }

    /// Get weighted mean estimate of all particles
    pub fn get_mean_estimate(&self) -> Option<Vector3> {
        let total_weight: f64 = self.particles.iter().map(|p| p.weight).sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut weighted_pos = Vector3::zeros();
        for particle in &self.particles {
            weighted_pos += particle.pos * particle.weight;
        }
        Some(weighted_pos / total_weight)
    }

    /// Get weighted mean velocity estimate
    pub fn get_mean_velocity(&self) -> Option<Vector3> {
        let total_weight: f64 = self.particles.iter().map(|p| p.weight).sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut weighted_vel = Vector3::zeros();
        for particle in &self.particles {
            weighted_vel += particle.vel * particle.weight;
        }
        Some(weighted_vel / total_weight)
    }

    /// Check if filter is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ParticleFilterConfig) {
        self.config = config;
    }
}