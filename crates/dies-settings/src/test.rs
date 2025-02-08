struct Tracker {
    controlled_player_active_filter: ActiveFilter,
}

impl Tracker {
    settings! {
        #![settings(name = "Tracker", description = "The tracker settings")]

        #[setting(label = "Max FPS", description = "The maximum FPS to allow", min = 1, max = 120, step = 1, unit = "FPS")]
        max_fps: u32 = 60,

        #[setting(label = "Field Mask", description = "Use this to limit tracking to a specific part of the field")]
        field_mask: FieldMask = FieldMask::default(),

        #[setting(label = "Controlled Player Filter", description = "The type of filter to use for the controlled player")]
        controlled_player_filter: FilterConfig = FilterConfig::Gaussian { alpha: 0.5 },

        #[setting(label = "Non Controlled Player Filter", description = "The type of filter to use for the non controlled player")]
        non_controlled_player_filter: FilterConfig = FilterConfig::Gaussian { alpha: 0.5 },

        #[setting(label = "Ball Filter", description = "The type of filter to use for the ball")]
        ball_filter: FilterConfig = FilterConfig::Gaussian { alpha: 0.5 },
    }

    fn update(&mut self) {
        // Accessing a setting field with the `()` syntax
        let max_fps = self.max_fps();
        println!("Max FPS: {}", max_fps);

        let field_mask = self.field_mask();
        println!("Field Mask: {:?}", field_mask);

        let controlled_player_filter = self.get_controlled_player_filter();
    }

    fn get_controlled_player_filter(&mut self) -> ActiveFilter {
        // If the filter type differs from the active filter, update the active filter
        if !self
            .controlled_player_filter()
            .matches(&self.controlled_player_active_filter)
        {
            self.controlled_player_active_filter =
                ActiveFilter::from_config(self.controlled_player_filter);
        }
        self.controlled_player_active_filter
    }

    // ... other filter getters
}

#[derive(Debug, Clone, Settings, Serialize, Deserialize)]
struct FieldMask {
    #[setting(label = "Top Left", display = "slider")]
    top_left: Vec2,
    #[setting(label = "Bottom Right", display = "slider")]
    bottom_right: Vec2,
}

impl Default for FieldMask {
    fn default() -> Self {
        Self {
            top_left: Vec2::new(0.0, 0.0),
            bottom_right: Vec2::new(1.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Settings, Serialize, Deserialize)]
enum FilterConfig {
    Gaussian {
        #[setting(label = "Alpha")]
        alpha: f32,
    },
    LowPass {
        #[setting(label = "Alpha")]
        alpha: f32,
    },
    Kalman {
        #[setting(label = "Var Noise", description = "The variance of the noise")]
        var_noise: f32,
        #[setting(label = "Var Measure", description = "The variance of the measurement")]
        var_measure: f32,
    },
}

impl FilterConfig {
    fn matches(&self, filter: &ActiveFilter) -> bool {
        match (self, filter) {
            (FilterConfig::Gaussian { .. }, ActiveFilter::Gaussian(_)) => true,
            (FilterConfig::LowPass { .. }, ActiveFilter::LowPass(_)) => true,
            (FilterConfig::Kalman { .. }, ActiveFilter::Kalman(_)) => true,
            _ => false,
        }
    }
}

enum ActiveFilter {
    Gaussian(GaussianFilter),
    LowPass(LowPassFilter),
    Kalman(KalmanFilter),
}

impl ActiveFilter {
    fn from_config(config: FilterConfig) -> Self {
        match config {
            FilterConfig::Gaussian { alpha } => ActiveFilter::Gaussian(GaussianFilter { alpha }),
            FilterConfig::LowPass { alpha } => ActiveFilter::LowPass(LowPassFilter { alpha }),
            FilterConfig::Kalman {
                var_noise,
                var_measure,
            } => ActiveFilter::Kalman(KalmanFilter {
                var_noise,
                var_measure,
            }),
        }
    }

    fn filter_type(&self) -> FilterType {
        match self {
            ActiveFilter::Gaussian(_) => FilterType::Gaussian,
            ActiveFilter::LowPass(_) => FilterType::LowPass,
            ActiveFilter::Kalman(_) => FilterType::Kalman,
        }
    }
}
