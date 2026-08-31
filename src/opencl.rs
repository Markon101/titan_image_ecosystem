use anyhow::{bail, Context, Result};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context as ClContext;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_float, cl_uint, CL_BLOCKING, CL_NON_BLOCKING};
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

const NCA_SOURCE: &str = r#"
inline float nca_swish(float x) { return x / (1.0f + exp(-x)); }

kernel void nca_perception(
    global const float *field, global float *perceived,
    uint pixels, uint width, uint height, uint channels) {
  uint gid = get_global_id(0);
  uint feature = gid / pixels;
  uint p = gid - feature * pixels;
  if (p >= pixels || feature >= channels * 8) return;
  uint ring = feature / (channels * 4);
  uint inner = feature - ring * channels * 4;
  uint channel = inner / 4;
  uint filter = inner - channel * 4;
  int dilation = (int)ring + 1;
  int x = (int)(p % width);
  int y = (int)(p / width);
  float sum = 0.0f;
  for (int ky = 0; ky < 3; ++ky) {
    int sy = (y + (ky - 1) * dilation + (int)height) % (int)height;
    for (int kx = 0; kx < 3; ++kx) {
      int sx = (x + (kx - 1) * dilation + (int)width) % (int)width;
      float coefficient = 0.0f;
      if (filter == 0) {
        coefficient = (kx == 1 && ky == 1) ? 1.0f : 0.0f;
      } else if (filter == 1) {
        const float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        coefficient = sobel_x[ky * 3 + kx] * 0.125f;
      } else if (filter == 2) {
        const float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        coefficient = sobel_y[ky * 3 + kx] * 0.125f;
      } else {
        const float laplacian[9] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
        coefficient = laplacian[ky * 3 + kx] * 0.125f;
      }
      sum += coefficient * field[channel * pixels + (uint)sy * width + (uint)sx];
    }
  }
  perceived[feature * pixels + p] = sum;
}

kernel void nca_input_layer(
    global const float *perceived, global const float *macro_context,
    global const float *genome, global const float *weights, global const float *biases,
    global float *hidden_out, uint pixels, uint channels, uint genome_dim,
    uint hidden_width, uint weight_offset, uint bias_offset) {
  uint gid = get_global_id(0);
  uint p = gid / hidden_width;
  uint o = gid - p * hidden_width;
  if (p >= pixels) return;
  uint perception_features = channels * 8;
  uint input_features = perception_features + channels + genome_dim;
  uint row = weight_offset + o * input_features;
  float sum = biases[bias_offset + o];
  for (uint i = 0; i < perception_features; ++i)
    sum += weights[row + i] * perceived[i * pixels + p];
  for (uint i = 0; i < channels; ++i)
    sum += weights[row + perception_features + i] * macro_context[i * pixels + p];
  for (uint i = 0; i < genome_dim; ++i)
    sum += weights[row + perception_features + channels + i] * genome[i];
  hidden_out[p * hidden_width + o] = nca_swish(sum);
}

kernel void nca_residual_layer(
    global const float *hidden_in, global const float *weights, global const float *biases,
    global float *hidden_out, uint pixels, uint hidden_width,
    uint weight_offset, uint bias_offset) {
  uint gid = get_global_id(0);
  uint p = gid / hidden_width;
  uint o = gid - p * hidden_width;
  if (p >= pixels) return;
  uint base = p * hidden_width;
  uint row = weight_offset + o * hidden_width;
  float sum = biases[bias_offset + o];
  for (uint i = 0; i < hidden_width; ++i)
    sum += weights[row + i] * hidden_in[base + i];
  hidden_out[base + o] = hidden_in[base + o] + 0.5f * nca_swish(sum);
}

kernel void nca_output_layer(
    global const float *hidden, global const float *weights, global const float *biases,
    global const float *clock_masks, global float *output,
    uint pixels, uint channels, uint hidden_width, uint clock_index, float gain,
    uint weight_offset, uint bias_offset) {
  uint gid = get_global_id(0);
  uint p = gid / channels;
  uint o = gid - p * channels;
  if (p >= pixels) return;
  uint base = p * hidden_width;
  uint row = weight_offset + o * hidden_width;
  float sum = biases[bias_offset + o];
  for (uint i = 0; i < hidden_width; ++i)
    sum += weights[row + i] * hidden[base + i];
  output[o * pixels + p] = tanh(sum) * clock_masks[clock_index * pixels + p] * gain;
}
"#;

pub struct OpenClNca {
    _context: ClContext,
    queue: CommandQueue,
    _program: Program,
    perception_kernel: Kernel,
    input_kernel: Kernel,
    residual_kernel: Kernel,
    output_kernel: Kernel,
    weights: Buffer<cl_float>,
    biases: Buffer<cl_float>,
    clock_masks: Buffer<cl_float>,
    field: Buffer<cl_float>,
    macro_context: Buffer<cl_float>,
    genome: Buffer<cl_float>,
    perceived: Buffer<cl_float>,
    hidden_a: Buffer<cl_float>,
    hidden_b: Buffer<cl_float>,
    output: Buffer<cl_float>,
    offsets: [(u32, u32); 3],
    size: usize,
    pixels: usize,
    channels: usize,
    genome_dim: usize,
    hidden: usize,
    gain: f32,
    clock_mask_count: usize,
}

impl OpenClNca {
    pub fn new(
        layers: &[LinearData],
        size: usize,
        channels: usize,
        gain: f32,
        clock_masks: &[Vec<f32>],
    ) -> Result<Self> {
        if layers.len() != 3 || clock_masks.is_empty() {
            bail!("OpenCL NCA requires three linear layers and clock masks");
        }
        let hidden = layers[0].output;
        let expected_spatial = channels.checked_mul(9).context("NCA channel overflow")?;
        let genome_dim = layers[0]
            .input
            .checked_sub(expected_spatial)
            .context("invalid OpenCL NCA input width")?;
        if size == 0
            || channels == 0
            || hidden == 0
            || layers[1].input != hidden
            || layers[1].output != hidden
            || layers[2].input != hidden
            || layers[2].output != channels
        {
            bail!("unsupported OpenCL NCA layer shapes");
        }
        let pixels = size.checked_mul(size).context("NCA field size overflow")?;
        if clock_masks.iter().any(|mask| mask.len() != pixels) {
            bail!("OpenCL NCA clock-mask shape mismatch");
        }
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .context("OpenCL NCA device discovery failed")?
            .first()
            .context("no OpenCL GPU device found for NCA")?;
        let device = Device::new(device_id);
        let context =
            ClContext::from_device(&device).context("OpenCL NCA context creation failed")?;
        let queue = CommandQueue::create_default(&context, 0)
            .context("OpenCL NCA command queue creation failed")?;
        let program = Program::create_and_build_from_source(&context, NCA_SOURCE, "-cl-std=CL1.2")
            .map_err(|e| anyhow::anyhow!("OpenCL NCA kernel build failed: {e}"))?;
        let perception_kernel = Kernel::create(&program, "nca_perception")?;
        let input_kernel = Kernel::create(&program, "nca_input_layer")?;
        let residual_kernel = Kernel::create(&program, "nca_residual_layer")?;
        let output_kernel = Kernel::create(&program, "nca_output_layer")?;

        let mut packed_weights = Vec::new();
        let mut packed_biases = Vec::new();
        let mut offsets = Vec::with_capacity(3);
        for layer in layers {
            if layer.weight.len() != layer.input * layer.output || layer.bias.len() != layer.output
            {
                bail!("invalid OpenCL NCA weight payload");
            }
            offsets.push((packed_weights.len() as u32, packed_biases.len() as u32));
            packed_weights.extend_from_slice(&layer.weight);
            packed_biases.extend_from_slice(&layer.bias);
        }
        let masks = clock_masks.iter().flatten().copied().collect::<Vec<_>>();
        let field_len = channels
            .checked_mul(pixels)
            .context("NCA field allocation overflow")?;
        let perceived_len = field_len
            .checked_mul(8)
            .context("NCA perception allocation overflow")?;
        let hidden_len = hidden
            .checked_mul(pixels)
            .context("NCA hidden allocation overflow")?;
        let mut weights = create_buffer(&context, CL_MEM_READ_ONLY, packed_weights.len())?;
        let mut biases = create_buffer(&context, CL_MEM_READ_ONLY, packed_biases.len())?;
        let mut mask_buffer = create_buffer(&context, CL_MEM_READ_ONLY, masks.len())?;
        unsafe {
            queue.enqueue_write_buffer(&mut weights, CL_BLOCKING, 0, &packed_weights, &[])?;
            queue.enqueue_write_buffer(&mut biases, CL_BLOCKING, 0, &packed_biases, &[])?;
            queue.enqueue_write_buffer(&mut mask_buffer, CL_BLOCKING, 0, &masks, &[])?;
        }
        let field = create_buffer(&context, CL_MEM_READ_ONLY, field_len)?;
        let macro_context = create_buffer(&context, CL_MEM_READ_ONLY, field_len)?;
        let genome = create_buffer(&context, CL_MEM_READ_ONLY, genome_dim)?;
        let perceived = create_buffer(&context, CL_MEM_READ_WRITE, perceived_len)?;
        let hidden_a = create_buffer(&context, CL_MEM_READ_WRITE, hidden_len)?;
        let hidden_b = create_buffer(&context, CL_MEM_READ_WRITE, hidden_len)?;
        let output = create_buffer(&context, CL_MEM_WRITE_ONLY, field_len)?;
        Ok(Self {
            _context: context,
            queue,
            _program: program,
            perception_kernel,
            input_kernel,
            residual_kernel,
            output_kernel,
            weights,
            biases,
            clock_masks: mask_buffer,
            field,
            macro_context,
            genome,
            perceived,
            hidden_a,
            hidden_b,
            output,
            offsets: [offsets[0], offsets[1], offsets[2]],
            size,
            pixels,
            channels,
            genome_dim,
            hidden,
            gain,
            clock_mask_count: clock_masks.len(),
        })
    }

    pub fn run(
        &mut self,
        field: &[f32],
        macro_context: &[f32],
        genome: &[f32],
        clock_index: usize,
    ) -> Result<Vec<f32>> {
        let field_len = self.channels * self.pixels;
        if field.len() != field_len
            || macro_context.len() != field_len
            || genome.len() != self.genome_dim
            || clock_index >= self.clock_mask_count
        {
            bail!("OpenCL NCA input payload mismatch");
        }
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.field, CL_NON_BLOCKING, 0, field, &[])?;
            self.queue.enqueue_write_buffer(
                &mut self.macro_context,
                CL_NON_BLOCKING,
                0,
                macro_context,
                &[],
            )?;
            self.queue
                .enqueue_write_buffer(&mut self.genome, CL_NON_BLOCKING, 0, genome, &[])?;
        }
        let pixels = self.pixels as cl_uint;
        let width = self.size as cl_uint;
        let height = self.size as cl_uint;
        let channels = self.channels as cl_uint;
        let genome_dim = self.genome_dim as cl_uint;
        let hidden = self.hidden as cl_uint;
        let clock = clock_index as cl_uint;
        let gain = self.gain;
        unsafe {
            ExecuteKernel::new(&self.perception_kernel)
                .set_arg(&self.field)
                .set_arg(&self.perceived)
                .set_arg(&pixels)
                .set_arg(&width)
                .set_arg(&height)
                .set_arg(&channels)
                .set_global_work_size(self.pixels * self.channels * 8)
                .enqueue_nd_range(&self.queue)?;
            let (wi, bi) = self.offsets[0];
            ExecuteKernel::new(&self.input_kernel)
                .set_arg(&self.perceived)
                .set_arg(&self.macro_context)
                .set_arg(&self.genome)
                .set_arg(&self.weights)
                .set_arg(&self.biases)
                .set_arg(&self.hidden_a)
                .set_arg(&pixels)
                .set_arg(&channels)
                .set_arg(&genome_dim)
                .set_arg(&hidden)
                .set_arg(&wi)
                .set_arg(&bi)
                .set_global_work_size(self.pixels * self.hidden)
                .enqueue_nd_range(&self.queue)?;
            let (wh, bh) = self.offsets[1];
            ExecuteKernel::new(&self.residual_kernel)
                .set_arg(&self.hidden_a)
                .set_arg(&self.weights)
                .set_arg(&self.biases)
                .set_arg(&self.hidden_b)
                .set_arg(&pixels)
                .set_arg(&hidden)
                .set_arg(&wh)
                .set_arg(&bh)
                .set_global_work_size(self.pixels * self.hidden)
                .enqueue_nd_range(&self.queue)?;
            let (wo, bo) = self.offsets[2];
            ExecuteKernel::new(&self.output_kernel)
                .set_arg(&self.hidden_b)
                .set_arg(&self.weights)
                .set_arg(&self.biases)
                .set_arg(&self.clock_masks)
                .set_arg(&self.output)
                .set_arg(&pixels)
                .set_arg(&channels)
                .set_arg(&hidden)
                .set_arg(&clock)
                .set_arg(&gain)
                .set_arg(&wo)
                .set_arg(&bo)
                .set_global_work_size(self.pixels * self.channels)
                .enqueue_nd_range(&self.queue)?;
        }
        let mut output = vec![0.0f32; field_len];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.output, CL_BLOCKING, 0, &mut output, &[])?;
        }
        Ok(output)
    }
}

fn create_buffer(context: &ClContext, flags: u64, elements: usize) -> Result<Buffer<cl_float>> {
    unsafe {
        Buffer::<cl_float>::create(context, flags, elements, ptr::null_mut()).map_err(Into::into)
    }
}
