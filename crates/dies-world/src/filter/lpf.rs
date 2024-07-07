use std::collections::VecDeque;

/// A struct representing an n-th order low-pass filter.
#[derive(Debug)]
pub struct LowpassFilter {
    /// The order of the filter.
    order: usize,
    /// The coefficients for the input samples (b coefficients).
    b_coeffs: Vec<f64>,
    /// The coefficients for the output samples (a coefficients).
    a_coeffs: Vec<f64>,
    /// A buffer to store previous input samples.
    input_buffer: VecDeque<f64>,
    /// A buffer to store previous output samples.
    output_buffer: VecDeque<f64>,
}

impl LowpassFilter {
    /// Creates a new LowpassFilter with the given order and coefficients.
    pub fn new(order: usize, b_coeffs: Vec<f64>, a_coeffs: Vec<f64>) -> Self {
        assert_eq!(b_coeffs.len(), order + 1);
        assert_eq!(a_coeffs.len(), order + 1);

        LowpassFilter {
            order,
            b_coeffs,
            a_coeffs,
            input_buffer: VecDeque::with_capacity(order + 1),
            output_buffer: VecDeque::with_capacity(order),
        }
    }

    /// Updates the filter with a new input sample and returns the filtered output.
    pub fn update(&mut self, input: f64) -> f64 {
        // Add new input to the buffer
        self.input_buffer.push_front(input);
        if self.input_buffer.len() > self.order + 1 {
            self.input_buffer.pop_back();
        }

        // Calculate the new output
        let mut output = 0.0;
        for (i, &b) in self.b_coeffs.iter().enumerate() {
            if i < self.input_buffer.len() {
                output += b * self.input_buffer[i];
            }
        }
        for (i, &a) in self.a_coeffs.iter().skip(1).enumerate() {
            if i < self.output_buffer.len() {
                output -= a * self.output_buffer[i];
            }
        }
        output /= self.a_coeffs[0] + std::f64::EPSILON;

        // Add new output to the buffer
        self.output_buffer.push_front(output);
        if self.output_buffer.len() > self.order {
            self.output_buffer.pop_back();
        }

        output
    }

    /// Updates the filter settings (order and coefficients) at runtime.
    pub fn update_settings(
        &mut self,
        new_order: usize,
        new_b_coeffs: Vec<f64>,
        new_a_coeffs: Vec<f64>,
    ) {
        assert_eq!(new_b_coeffs.len(), new_order + 1);
        assert_eq!(new_a_coeffs.len(), new_order + 1);
        assert!(
            new_a_coeffs[0] != 0.0,
            "First a coefficient must not be zero"
        );

        self.order = new_order;
        self.b_coeffs = new_b_coeffs;
        self.a_coeffs = new_a_coeffs;

        // Resize buffers
        self.input_buffer.resize(new_order + 1, 0.0);
        self.output_buffer.resize(new_order, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_create_filter() {
        let filter = LowpassFilter::new(2, vec![0.1, 0.2, 0.1], vec![1.0, -0.5, 0.2]);
        assert_eq!(filter.order, 2);
        assert_eq!(filter.b_coeffs, vec![0.1, 0.2, 0.1]);
        assert_eq!(filter.a_coeffs, vec![1.0, -0.5, 0.2]);
    }

    #[test]
    #[should_panic(expected = "First a coefficient must not be zero")]
    fn test_create_filter_with_invalid_a_coeffs() {
        LowpassFilter::new(2, vec![0.1, 0.2, 0.1], vec![0.0, -0.5, 0.2]);
    }

    #[test]
    fn test_update_first_order() {
        let mut filter = LowpassFilter::new(1, vec![0.2, 0.2], vec![1.0, -0.6]);
        assert_relative_eq!(filter.update(1.0), 0.2, epsilon = 1e-6);
        assert_relative_eq!(filter.update(1.0), 0.52, epsilon = 1e-6);
        assert_relative_eq!(filter.update(1.0), 0.712, epsilon = 1e-6);
    }

    #[test]
    fn test_update_second_order() {
        let mut filter = LowpassFilter::new(2, vec![0.1, 0.2, 0.1], vec![1.0, -0.5, 0.2]);
        assert_relative_eq!(filter.update(1.0), 0.1, epsilon = 1e-6);
        assert_relative_eq!(filter.update(1.0), 0.35, epsilon = 1e-6);
        assert_relative_eq!(filter.update(1.0), 0.555, epsilon = 1e-6);
    }

    #[test]
    fn test_update_settings() {
        let mut filter = LowpassFilter::new(1, vec![0.2, 0.2], vec![1.0, -0.6]);
        assert_relative_eq!(filter.update(1.0), 0.2, epsilon = 1e-6);

        filter.update_settings(2, vec![0.1, 0.2, 0.1], vec![1.0, -0.5, 0.2]);
        assert_eq!(filter.order, 2);
        assert_eq!(filter.b_coeffs, vec![0.1, 0.2, 0.1]);
        assert_eq!(filter.a_coeffs, vec![1.0, -0.5, 0.2]);

        assert_relative_eq!(filter.update(1.0), 0.4, epsilon = 1e-6);
    }
}
