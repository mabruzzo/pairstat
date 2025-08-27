use pyo3::prelude::*;
use pairstat::Accumulator;

#[pyclass]
struct PyAccumulator {
    accum: Accumulator
}

#[pymethods]
impl PyAccumulator {
    /// Reset the tracked accumulator state
    fn reset_data(mut self_: PyRefMut<Self>) {
        self_.accum.reset_data()
    }
}


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _pairstat_rs_pybind(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<PyAccumulator>()?;
    Ok(())
}
