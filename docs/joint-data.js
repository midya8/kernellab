"use strict";
globalThis.KERNEL_JOINT_CONTROLS = [
  {
    name: "opposite-sides",
    description:
      "Same marginal variances as repeated-location control, but negative posterior cross covariance. Difference uncertainty is larger than the diagonal-only estimate.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.0,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[-1.0], [1.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "repeated-latent",
    description:
      "Same marginal variances as opposite-sides control. Two queries of the identical latent value cancel exactly in a difference.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.0,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[-1.0], [-1.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "repeated-noisy",
    description:
      "Identical latent value cancels, but two independent future observation noises do not. Difference noise variance is twice the noise variance.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.2,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[-1.0], [-1.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "nearby-positive",
    description:
      "Nearby queries on the same side share positive posterior covariance. Diagonal-only uncertainty overstates their difference.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.0,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[0.8], [1.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "training-query",
    description:
      "Noiseless training query is deterministic under fixed assumptions; a separate unobserved query remains random.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.0,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[0.0], [1.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "zero-weights",
    description:
      "The zero linear functional is deterministic, not an optimizer or feasibility certificate.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.0,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[-1.0], [1.0]],
      weights: [[0.0, 0.0]],
    },
  },
  {
    name: "shared-observation-noise",
    description:
      "Two contrasts reusing a future observation share its noise. Contrast covariance adds noise times W times transpose(W).",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.2,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[-1.0], [-1.0]],
      weights: [
        [1.0, 0.0],
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "training-jitter",
    description:
      "Explicit jitter changes conditioning but is not future observation noise. Repeated latent differences still vanish.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.0,
          jitter: 0.01,
          prior_mean: 0.0,
        },
      },
      queries: [[0.0], [0.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "signed-multidimensional",
    description:
      "Signed linear forms over a two-feature query set retain every pair contribution and cross-contrast covariance.",
    input: {
      training: {
        rows: [
          [-1.0, 0.0],
          [0.0, 1.0],
          [1.0, -1.0],
        ],
        targets: [0.5, -0.2, 0.8],
        kernel: {
          length_scale: 0.8,
          signal_variance: 1.3,
          noise_variance: 0.1,
          jitter: 1e-8,
          prior_mean: 0.3,
        },
      },
      queries: [
        [-0.5, 0.2],
        [0.4, 0.5],
        [1.0, -0.8],
      ],
      weights: [
        [1.0, -2.0, 0.5],
        [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
      ],
    },
  },
  {
    name: "far-training",
    description:
      "Training-query covariance underflows here; posterior returns to the supplied prior, not a calibrated confidence statement.",
    input: {
      training: {
        rows: [[100.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.0,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[-1.0], [1.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "tiny-signal",
    description:
      "Small signal variance remains explicit. The numerical gate is a diagnostic, not a relative-error guarantee near zero.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [0.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1e-12,
          noise_variance: 0.0,
          jitter: 0.0,
          prior_mean: 0.0,
        },
      },
      queries: [[-1.0], [1.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
  {
    name: "large-mean",
    description:
      "Changing target and supplied mean moves linear means but does not change fixed-kernel covariance.",
    input: {
      training: {
        rows: [[0.0]],
        targets: [1000000.0],
        kernel: {
          length_scale: 1.0,
          signal_variance: 1.0,
          noise_variance: 0.0,
          jitter: 0.0,
          prior_mean: -1000000.0,
        },
      },
      queries: [[-1.0], [1.0]],
      weights: [
        [1.0, -1.0],
        [0.5, 0.5],
      ],
    },
  },
];
