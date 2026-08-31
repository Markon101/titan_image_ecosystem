use anyhow::{bail, Context, Result};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context as ClContext;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_float, cl_uint, CL_BLOCKING};
use std::ptr;

const SOURCE: &str = r#"
inline float swish(float x) { return x / (1.0f + exp(-x)); }

kernel void input_layer(
    global const float *x, global const float *w, global const float *b,
    global float *y, uint pixels, uint inputs, uint outputs,
    uint weight_offset, uint bias_offset) {
  uint gid = get_global_id(0);
  uint p = gid / outputs;
  uint o = gid - p * outputs;
  if (p >= pixels) return;
  float sum = b[bias_offset + o];
  uint row = weight_offset + o * inputs;
  for (uint i = 0; i < inputs; ++i) sum += w[row + i] * x[i * pixels + p];
  y[p * outputs + o] = swish(sum);
}

kernel void residual_layer(
    global const float *x, global const float *w, global const float *b,
    global float *y, uint pixels, uint width,
    uint weight_offset, uint bias_offset) {
  uint gid = get_global_id(0);
  uint p = gid / width;
  uint o = gid - p * width;
  if (p >= pixels) return;
  uint base = p * width;
  float sum = b[bias_offset + o];
  uint row = weight_offset + o * width;
  for (uint i = 0; i < width; ++i) sum += w[row + i] * x[base + i];
  y[base + o] = x[base + o] + 0.5f * swish(sum);
}

kernel void output_heads(
    global const float *x, global const float *w, global const float *b,
    global float *y, uint pixels, uint width,
    uint grounded_weight, uint grounded_bias,
    uint emergent_weight, uint emergent_bias) {
  uint gid = get_global_id(0);
  uint p = gid / 6;
  uint lane = gid - p * 6;
  if (p >= pixels) return;
  uint o = lane % 3;
  uint base = p * width;
  uint weight = lane < 3 ? grounded_weight : emergent_weight;
  uint bias = lane < 3 ? grounded_bias : emergent_bias;
  float sum = b[bias + o];
  uint row = weight + o * width;
  for (uint i = 0; i < width; ++i) sum += w[row + i] * x[base + i];
  y[lane * pixels + p] = sum;
}
"#;

#[derive(Clone)]
pub struct LinearData {
    pub input: usize,
    pub output: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub opencl_c_version: String,
    pub compute_units: u32,
    pub max_work_group_size: usize,
    pub global_mem_bytes: u64,
    pub max_alloc_bytes: u64,
    pub local_mem_bytes: u64,
    pub fp16: bool,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}) | {} | {} | CUs={} WG={} global={}MiB alloc={}MiB local={}KiB fp16={}",
            self.name,
            self.vendor,
            self.version,
            self.opencl_c_version,
            self.compute_units,
            self.max_work_group_size,
            self.global_mem_bytes / 1_048_576,
            self.max_alloc_bytes / 1_048_576,
            self.local_mem_bytes / 1024,
            self.fp16
        )
    }
}

pub struct OpenClMlp {
    _context: ClContext,
    queue: CommandQueue,
    _program: Program,
    input_kernel: Kernel,
    residual_kernel: Kernel,
    heads_kernel: Kernel,
    weights: Buffer<cl_float>,
    biases: Buffer<cl_float>,
    offsets: Vec<(u32, u32)>,
    input_features: usize,
    hidden: usize,
    blocks: usize,
    pub info: DeviceInfo,
}

impl OpenClMlp {
    pub fn new(layers: &[LinearData], blocks: usize) -> Result<Self> {
        if layers.len() != blocks + 3 {
            bail!(
                "OpenCL renderer expected {} layers, got {}",
                blocks + 3,
                layers.len()
            );
        }
        let input_features = layers[0].input;
        let hidden = layers[0].output;
        if hidden == 0
            || layers[1..=blocks]
                .iter()
                .any(|l| l.input != hidden || l.output != hidden)
            || layers[blocks + 1].input != hidden
            || layers[blocks + 1].output != 3
            || layers[blocks + 2].input != hidden
            || layers[blocks + 2].output != 3
        {
            bail!("unsupported OpenCL renderer layer shapes");
        }
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .context("OpenCL device discovery failed")?
            .first()
            .context("no OpenCL GPU device found")?;
        let device = Device::new(device_id);
        let extensions = device.extensions().unwrap_or_default();
        let info = DeviceInfo {
            name: device.name().unwrap_or_else(|_| "unknown".into()),
            vendor: device.vendor().unwrap_or_else(|_| "unknown".into()),
            version: device.version().unwrap_or_else(|_| "unknown".into()),
            opencl_c_version: device
                .opencl_c_version()
                .unwrap_or_else(|_| "unknown".into()),
            compute_units: device.max_compute_units().unwrap_or(0),
            max_work_group_size: device.max_work_group_size().unwrap_or(0),
            global_mem_bytes: device.global_mem_size().unwrap_or(0),
            max_alloc_bytes: device.max_mem_alloc_size().unwrap_or(0),
            local_mem_bytes: device.local_mem_size().unwrap_or(0),
            fp16: extensions.split_whitespace().any(|x| x == "cl_khr_fp16"),
        };
        let context = ClContext::from_device(&device).context("OpenCL context creation failed")?;
        let queue = CommandQueue::create_default(&context, 0)
            .context("OpenCL command queue creation failed")?;
        let program = Program::create_and_build_from_source(&context, SOURCE, "-cl-std=CL1.2")
            .map_err(|e| anyhow::anyhow!("OpenCL renderer kernel build failed: {e}"))?;
        let input_kernel = Kernel::create(&program, "input_layer")?;
        let residual_kernel = Kernel::create(&program, "residual_layer")?;
        let heads_kernel = Kernel::create(&program, "output_heads")?;

        let mut packed_weights = Vec::new();
        let mut packed_biases = Vec::new();
        let mut offsets = Vec::with_capacity(layers.len());
        for layer in layers {
            if layer.weight.len() != layer.input * layer.output || layer.bias.len() != layer.output
            {
                bail!("invalid OpenCL renderer weight payload");
            }
            offsets.push((packed_weights.len() as u32, packed_biases.len() as u32));
            packed_weights.extend_from_slice(&layer.weight);
            packed_biases.extend_from_slice(&layer.bias);
        }
        let mut weights = unsafe {
            Buffer::<cl_float>::create(
                &context,
                CL_MEM_READ_ONLY,
                packed_weights.len(),
                ptr::null_mut(),
            )?
        };
        let mut biases = unsafe {
            Buffer::<cl_float>::create(
                &context,
                CL_MEM_READ_ONLY,
                packed_biases.len(),
                ptr::null_mut(),
            )?
        };
        unsafe {
            queue.enqueue_write_buffer(&mut weights, CL_BLOCKING, 0, &packed_weights, &[])?;
            queue.enqueue_write_buffer(&mut biases, CL_BLOCKING, 0, &packed_biases, &[])?;
        }
        Ok(Self {
            _context: context,
            queue,
            _program: program,
            input_kernel,
            residual_kernel,
            heads_kernel,
            weights,
            biases,
            offsets,
            input_features,
            hidden,
            blocks,
            info,
        })
    }

    pub fn run(&self, features: &[f32], pixels: usize) -> Result<Vec<f32>> {
        if features.len() != self.input_features * pixels {
            bail!("OpenCL renderer feature payload mismatch");
        }
        let context = &self._context;
        let mut input = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_ONLY, features.len(), ptr::null_mut())?
        };
        let hidden_len = pixels
            .checked_mul(self.hidden)
            .context("OpenCL hidden size overflow")?;
        let hidden_a = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, hidden_len, ptr::null_mut())?
        };
        let hidden_b = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, hidden_len, ptr::null_mut())?
        };
        let output = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_WRITE_ONLY, pixels * 6, ptr::null_mut())?
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut input, CL_BLOCKING, 0, features, &[])?;
        }
        let p = pixels as cl_uint;
        let input_features = self.input_features as cl_uint;
        let hidden = self.hidden as cl_uint;
        let (w0, b0) = self.offsets[0];
        unsafe {
            ExecuteKernel::new(&self.input_kernel)
                .set_arg(&input)
                .set_arg(&self.weights)
                .set_arg(&self.biases)
                .set_arg(&hidden_a)
                .set_arg(&p)
                .set_arg(&input_features)
                .set_arg(&hidden)
                .set_arg(&w0)
                .set_arg(&b0)
                .set_global_work_size(pixels * self.hidden)
                .enqueue_nd_range(&self.queue)?;
        }
        let mut source_is_a = true;
        for index in 0..self.blocks {
            let (wo, bo) = self.offsets[index + 1];
            let (source, destination) = if source_is_a {
                (&hidden_a, &hidden_b)
            } else {
                (&hidden_b, &hidden_a)
            };
            unsafe {
                ExecuteKernel::new(&self.residual_kernel)
                    .set_arg(source)
                    .set_arg(&self.weights)
                    .set_arg(&self.biases)
                    .set_arg(destination)
                    .set_arg(&p)
                    .set_arg(&hidden)
                    .set_arg(&wo)
                    .set_arg(&bo)
                    .set_global_work_size(pixels * self.hidden)
                    .enqueue_nd_range(&self.queue)?;
            }
            source_is_a = !source_is_a;
        }
        let source = if source_is_a { &hidden_a } else { &hidden_b };
        let (gw, gb) = self.offsets[self.blocks + 1];
        let (ew, eb) = self.offsets[self.blocks + 2];
        unsafe {
            ExecuteKernel::new(&self.heads_kernel)
                .set_arg(source)
                .set_arg(&self.weights)
                .set_arg(&self.biases)
                .set_arg(&output)
                .set_arg(&p)
                .set_arg(&hidden)
                .set_arg(&gw)
                .set_arg(&gb)
                .set_arg(&ew)
                .set_arg(&eb)
                .set_global_work_size(pixels * 6)
                .enqueue_nd_range(&self.queue)?;
        }
        let mut result = vec![0.0f32; pixels * 6];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output, CL_BLOCKING, 0, &mut result, &[])?;
        }
        Ok(result)
    }
}
